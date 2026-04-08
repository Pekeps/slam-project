"""Stereo visual SLAM with loop closures and online pose-graph optimization.

Front-end: frame-to-frame stereo VO.
Back-end: a scipy-based SE(3) pose graph with sequential odometry edges
plus loop-closure edges detected by descriptor matching + PnP verification
against older keyframes.

Two trajectory arrays are maintained:

* ``raw_poses``  — the pure VO trajectory, never touched by optimization.
  Kept around for the before/after comparison on the final static plot.
* ``live_poses`` — the currently-believed trajectory, updated whenever the
  pose graph is re-optimized.

Ground-truth poses, if present, are loaded purely for evaluation/plotting.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from slam_utils import (
    form_T,
    invert_T,
    load_image_paths,
    load_poses_kitti,
    batch_invert_T,
    rot_to_angle_axis_batch,
    batch_se3_exp,
)
from slam_video import SlamVideoWriter
from stereo import (
    StereoFrame,
    get_stereo_frame,
    read_kitti_stereo_calib,
)


# === POLUT ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def paths_for_sequence(sequence: str) -> dict:
    """Return the calib / poses / image directory paths for a KITTI sequence."""
    return {
        "calib_path": os.path.join(
            DATA_DIR,
            "data_odometry_calib",
            "dataset",
            "sequences",
            sequence,
            "calib.txt",
        ),
        "poses_path": os.path.join(
            DATA_DIR, "data_odometry_poses", "dataset", "poses", f"{sequence}.txt"
        ),
        "left_dir": os.path.join(
            DATA_DIR, "data_odometry_gray", "dataset", "sequences", sequence, "image_0"
        ),
        "right_dir": os.path.join(
            DATA_DIR, "data_odometry_gray", "dataset", "sequences", sequence, "image_1"
        ),
    }


@dataclass
class SlamConfig:
    max_frames: int = 5000
    detector: str = "orb"  # "orb" (fast, binary), "sift" (slower, float), or "akaze" (binary, nonlinear scale space)
    n_features: int = 3000
    ratio_test: float = 0.70
    pnp_reproj_err_px: float = 3.0
    pnp_confidence: float = 0.999
    min_pnp_matches: int = 10
    max_epipolar_err_px: float = 1.5
    min_depth: float = 0.1
    max_depth: float = 200.0
    map_downsample: int = 5

    # Keyframe insertion
    keyframe_interval: int = 10  # insert a KF at least every N frames

    # Loop closure
    loop_skip_recent: int = 20  # ignore the N most-recent KFs when searching
    loop_min_matches: int = 80  # raw descriptor-match count to attempt verification
    loop_min_inliers: int = 50  # PnP inliers required to accept a loop
    loop_max_rel_trans: float = 100.0  # max recovered relative translation (m)
    loop_max_raw_dist: float = 500.0  # max raw-trajectory distance between KFs (m)

    # Pose-graph optimization
    pg_max_nfev: int = 500
    pg_ftol: float = 1e-4  # relative cost tolerance (1e-4 ≈ sub-mm residuals)
    pg_xtol: float = 1e-4  # relative parameter tolerance

    # Video visualization (optional)
    video_path: Optional[str] = None
    video_fps: int = 20


@dataclass
class Keyframe:
    id: int
    frame_idx: int
    T_w_c: np.ndarray  # mutable — updated by the pose graph
    kps: List[cv2.KeyPoint]
    desc: np.ndarray
    pts3d_cam: np.ndarray  # stored in camera frame so the map moves with T_w_c


@dataclass
class PoseGraph:
    """Lightweight SE(3) pose graph optimized with scipy's TRF solver.

    Nodes are initialized with ``add_vertex`` and store a full 4x4 pose.
    Each optimization step parameterizes the update as a local SE(3)
    increment around the current pose estimate.
    """

    vertices: List[np.ndarray] = field(default_factory=list)  # 4x4 T_w_c
    edges: List[Tuple[int, int, np.ndarray]] = field(default_factory=list)
    fixed: set = field(default_factory=set)

    def add_vertex(self, T_w_c: np.ndarray) -> int:
        self.vertices.append(T_w_c.copy())
        return len(self.vertices) - 1

    def fix(self, idx: int) -> None:
        self.fixed.add(idx)

    def add_edge(self, i: int, j: int, T_ij_meas: np.ndarray) -> None:
        self.edges.append((i, j, T_ij_meas.copy()))

    def optimize(self, max_nfev: int, ftol: float, xtol: float) -> dict:
        """Minimize the pose-graph cost over free vertex increments."""
        n = len(self.vertices)
        n_edges = len(self.edges)
        n_res = 6 * n_edges
        free = [k for k in range(n) if k not in self.fixed]
        free_indices = np.asarray(free, dtype=np.int64)
        free_lookup = {k: i for i, k in enumerate(free)}
        n_free = len(free)
        n_params = 6 * n_free

        # Snapshot the linearization point as a single (N, 4, 4) array so we
        # can do batched matmuls against it in the residual function.
        vertices0 = np.stack(self.vertices)  # (N, 4, 4)

        # Precompute per-edge measurement matrices and their inverses once.
        edge_i = np.fromiter((e[0] for e in self.edges), dtype=np.int64, count=n_edges)
        edge_j = np.fromiter((e[1] for e in self.edges), dtype=np.int64, count=n_edges)
        T_meas = np.stack([e[2] for e in self.edges])  # (E, 4, 4)
        T_meas_inv = batch_invert_T(T_meas)  # (E, 4, 4)

        # Sparsity pattern: each edge (i, j) only touches xi_i and xi_j, so
        # the Jacobian is block-sparse with 6x6 blocks. scipy.TRF uses this
        # to compute finite-difference columns in groups.
        sparsity = lil_matrix((n_res, n_params), dtype=bool)
        for e_idx in range(n_edges):
            r0 = 6 * e_idx
            for v in (int(edge_i[e_idx]), int(edge_j[e_idx])):
                if v in free_lookup:
                    p0 = 6 * free_lookup[v]
                    sparsity[r0 : r0 + 6, p0 : p0 + 6] = True

        def residuals(xi_flat: np.ndarray) -> np.ndarray:
            # Scatter the free xi increments into a full (N, 6) array with
            # zeros at the fixed vertices — exp(0) = I, so the fixed
            # vertices are automatically held at vertices0 after the matmul.
            xi_all = np.zeros((n, 6), dtype=np.float64)
            xi_all[free_indices] = xi_flat.reshape(n_free, 6)

            # Batched right-perturbation: T_k = T_k0 @ exp(xi_k)
            exp_all = batch_se3_exp(xi_all)  # (N, 4, 4)
            poses = vertices0 @ exp_all  # (N, 4, 4)

            # Gather endpoints for every edge, then batched error computation.
            Ti = poses[edge_i]  # (E, 4, 4)
            Tj = poses[edge_j]  # (E, 4, 4)
            Ti_inv = batch_invert_T(Ti)  # (E, 4, 4)
            T_pred = Ti_inv @ Tj  # (E, 4, 4)
            T_err = T_meas_inv @ T_pred  # (E, 4, 4)

            # 6-vector residual per edge: [translation; angle-axis]
            trans = T_err[:, :3, 3]  # (E, 3)
            rvec = rot_to_angle_axis_batch(T_err[:, :3, :3])  # (E, 3)
            # Interleave into a flat (6*E,) vector in the same order the
            # sparsity pattern assumes (6 residuals per edge, in sequence).
            res = np.empty(n_res, dtype=np.float64)
            res[0::6] = trans[:, 0]
            res[1::6] = trans[:, 1]
            res[2::6] = trans[:, 2]
            res[3::6] = rvec[:, 0]
            res[4::6] = rvec[:, 1]
            res[5::6] = rvec[:, 2]
            return res

        x0 = np.zeros(n_params, dtype=np.float64)
        cost_initial = float(0.5 * float(np.sum(residuals(x0) ** 2)))

        result = least_squares(
            residuals,
            x0,
            method="trf",
            jac_sparsity=sparsity.tocsr(),
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=1e-8,
        )

        # Bake the optimized increments back into the vertex poses. Using
        # the batched exp keeps the update itself fast too.
        xi_all_final = np.zeros((n, 6), dtype=np.float64)
        xi_all_final[free_indices] = result.x.reshape(n_free, 6)
        exp_final = batch_se3_exp(xi_all_final)
        vertices_final = vertices0 @ exp_final  # (N, 4, 4)
        for k in range(n):
            self.vertices[k] = vertices_final[k]

        return {
            "cost_initial": cost_initial,
            "cost_final": float(result.cost),
            "nfev": int(result.nfev),
            "message": result.message,
        }


class StereoSlam:
    def __init__(
        self,
        calib_path: str,
        left_dir: str,
        right_dir: str,
        poses_path: Optional[str] = None,
        config: Optional[SlamConfig] = None,
    ):
        self.config = config or SlamConfig()
        self.calib = read_kitti_stereo_calib(calib_path)
        left_all = load_image_paths(left_dir)
        right_all = load_image_paths(right_dir)

        # Align left and right by file stem (not by list index), then keep
        # only the frames present in both. Some KITTI extractions (notably
        # seq 08) have scattered missing left frames — naively pairing by
        # index would silently misalign the stereo pair and destroy the
        # triangulation. Filename-based intersection fixes this. For
        # normal sequences where both sides are complete and contiguous,
        # this is a no-op.
        def _stem(p: str) -> str:
            return os.path.splitext(os.path.basename(p))[0]

        left_by_stem = {_stem(p): p for p in left_all}
        right_by_stem = {_stem(p): p for p in right_all}
        common_stems = sorted(set(left_by_stem) & set(right_by_stem))
        n_dropped = (len(left_all) - len(common_stems)) + (
            len(right_all) - len(common_stems)
        )
        if n_dropped:
            print(
                f"warning: dropping {n_dropped} unmatched frame(s) "
                f"(left={len(left_all)}, right={len(right_all)}, "
                f"matched={len(common_stems)})"
            )
        self.left_paths = [left_by_stem[s] for s in common_stems][
            : self.config.max_frames
        ]
        self.right_paths = [right_by_stem[s] for s in common_stems][
            : self.config.max_frames
        ]

        # GT poses are indexed by the (full) sequence frame stem, so we
        # must filter them with the same stems to stay aligned with the
        # kept image pairs after dropping gaps.
        self.gt_poses: Optional[List[np.ndarray]] = None
        if poses_path is not None:
            gt_all = load_poses_kitti(poses_path)
            if gt_all is not None:
                gt_filtered = [
                    gt_all[int(s)] for s in common_stems if int(s) < len(gt_all)
                ]
                self.gt_poses = gt_filtered[: self.config.max_frames]

        det = self.config.detector.lower()
        if det == "orb":
            self.detector = cv2.ORB_create(nfeatures=self.config.n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif det == "sift":
            self.detector = cv2.SIFT_create(nfeatures=self.config.n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif det == "akaze":
            # AKAZE has no nfeatures cap; feature count is governed by `threshold`
            # (lower → more features). Default MLDB descriptor is binary → Hamming.
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError(
                f"Unknown detector {self.config.detector!r}; expected 'orb', 'sift', or 'akaze'."
            )

        # Front-end state
        self.raw_poses: List[np.ndarray] = []  # per-frame poses from VO only
        self.live_poses: List[np.ndarray] = []  # per-frame, updated on each PGO
        self.keyframes: List[Keyframe] = []
        self.pose_graph = PoseGraph()
        self.loop_edges: List[Tuple[int, int]] = []  # (i, j) KF id pairs
        self.pnp_failures = 0

        # World-frame map cache (rebuilt from keyframes on demand)
        self._map_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._map_dirty = True

        # Video writer (optional). Bounds are taken from GT if available so
        # the map view is stable throughout playback.
        self.video_writer: Optional[SlamVideoWriter] = None
        if self.config.video_path:
            bounds = None
            if self.gt_poses is not None and len(self.gt_poses) > 0:
                gt_xz = np.array(
                    [[T[0, 3], T[2, 3]] for T in self.gt_poses], dtype=np.float64
                )
                xmin, xmax = float(gt_xz[:, 0].min()), float(gt_xz[:, 0].max())
                zmin, zmax = float(gt_xz[:, 1].min()), float(gt_xz[:, 1].max())
                pad = 0.1 * max(xmax - xmin, zmax - zmin, 10.0)
                bounds = (xmin - pad, xmax + pad, zmin - pad, zmax + pad)
            self.video_writer = SlamVideoWriter(
                path=self.config.video_path,
                fps=self.config.video_fps,
                bounds_xz=bounds,
            )

    # -------- front-end --------

    def _track_pose(
        self,
        prev_X_w: np.ndarray,
        prev_desc: np.ndarray,
        curr_kps: List[cv2.KeyPoint],
        curr_desc: np.ndarray,
    ) -> Optional[np.ndarray]:
        if (
            prev_desc is None
            or len(prev_desc) < 2
            or curr_desc is None
            or len(curr_desc) == 0
        ):
            return None
        knn = self.matcher.knnMatch(curr_desc, prev_desc, k=2)
        pts_2d = []
        pts_3d = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= self.config.ratio_test * n.distance:
                continue
            pts_2d.append(curr_kps[m.queryIdx].pt)
            pts_3d.append(prev_X_w[m.trainIdx])
        if len(pts_2d) < self.config.min_pnp_matches:
            return None
        pts_2d = np.asarray(pts_2d, dtype=np.float64)
        pts_3d = np.asarray(pts_3d, dtype=np.float64)
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            self.calib.K,
            np.zeros(5),
            reprojectionError=self.config.pnp_reproj_err_px,
            confidence=self.config.pnp_confidence,
        )
        if not success:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return invert_T(form_T(R, tvec))

    def _insert_keyframe(
        self, frame_idx: int, T_w_c: np.ndarray, sf: StereoFrame
    ) -> Keyframe:
        kf = Keyframe(
            id=len(self.keyframes),
            frame_idx=frame_idx,
            T_w_c=T_w_c.copy(),
            kps=sf.kps,
            desc=sf.desc.copy(),
            pts3d_cam=sf.pts3d_cam.copy(),
        )
        self.keyframes.append(kf)

        v_id = self.pose_graph.add_vertex(kf.T_w_c)
        if v_id != kf.id:
            raise RuntimeError("KF id / pose graph vertex id mismatch")

        if len(self.keyframes) == 1:
            # First KF fixes the gauge of the world frame.
            self.pose_graph.fix(kf.id)
        else:
            prev_kf = self.keyframes[-2]
            # Odometry edge = pure-VO relative motion between the two KF frames.
            # Derived from raw_poses (not from kf.T_w_c values) so the edge is
            # invariant to whatever PGO has done to the world frame.
            T_rel = (
                invert_T(self.raw_poses[prev_kf.frame_idx])
                @ self.raw_poses[kf.frame_idx]
            )
            self.pose_graph.add_edge(prev_kf.id, kf.id, T_rel)

        self._map_dirty = True
        return kf

    # -------- live pose propagation & map cache --------

    def _compute_live_pose(self, frame_idx: int, T_w_c_raw: np.ndarray) -> np.ndarray:
        """Rebase a raw-VO pose onto the nearest preceding keyframe's current pose."""
        if not self.keyframes:
            return T_w_c_raw.copy()
        kf_frame_indices = np.array([kf.frame_idx for kf in self.keyframes])
        pos = int(np.searchsorted(kf_frame_indices, frame_idx, side="right") - 1)
        if pos < 0:
            kf = self.keyframes[0]
        else:
            kf = self.keyframes[pos]
        if frame_idx == kf.frame_idx:
            return kf.T_w_c.copy()
        T_kf_old = self.raw_poses[kf.frame_idx]
        T_kf_new = kf.T_w_c
        T_rel = invert_T(T_kf_old) @ T_w_c_raw
        return T_kf_new @ T_rel

    def _rebuild_live_poses(self) -> None:
        """Recompute live poses for every frame seen so far."""
        self.live_poses = [
            self._compute_live_pose(k, self.raw_poses[k])
            for k in range(len(self.raw_poses))
        ]

    def _optimize_pose_graph(self) -> None:
        info = self.pose_graph.optimize(
            max_nfev=self.config.pg_max_nfev,
            ftol=self.config.pg_ftol,
            xtol=self.config.pg_xtol,
        )
        # Bake optimized poses back into the keyframe objects.
        for kf in self.keyframes:
            kf.T_w_c = self.pose_graph.vertices[kf.id]
        self._map_dirty = True
        print(
            f"    PGO: cost {info['cost_initial']:.3f} -> {info['cost_final']:.3f} "
            f"({info['nfev']} nfev)"
        )

    def _get_world_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame 3D map points + descriptors, rebuilt lazily from keyframes."""
        if not self._map_dirty and self._map_cache is not None:
            return self._map_cache
        stride = self.config.map_downsample
        if not self.keyframes:
            self._map_cache = (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 32), dtype=np.uint8),
            )
        else:
            all_X: List[np.ndarray] = []
            all_D: List[np.ndarray] = []
            for kf in self.keyframes:
                if len(kf.pts3d_cam) == 0:
                    continue
                X_cam = kf.pts3d_cam[::stride]
                D = kf.desc[::stride]
                X_w = (kf.T_w_c[:3, :3] @ X_cam.T + kf.T_w_c[:3, 3:4]).T
                all_X.append(X_w)
                all_D.append(D)
            if all_X:
                self._map_cache = (
                    np.concatenate(all_X, axis=0),
                    np.concatenate(all_D, axis=0),
                )
            else:
                self._map_cache = (
                    np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 32), dtype=np.uint8),
                )
        self._map_dirty = False
        return self._map_cache

    def _write_video_frame(
        self,
        i: int,
        left_img: np.ndarray,
        sf: StereoFrame,
        total_loop_closures: int,
        gt_xz_full: Optional[np.ndarray],
    ) -> None:
        if self.video_writer is None:
            return

        # Live trajectory so far, in (x, z)
        traj_xz = np.array(
            [[T[0, 3], T[2, 3]] for T in self.live_poses], dtype=np.float64
        )

        # Map points (from the cached keyframe map), projected to (x, z)
        X_w, _ = self._get_world_map()
        map_xz = X_w[:, [0, 2]] if len(X_w) > 0 else np.zeros((0, 2))

        # GT up to current frame
        gt_xz = None
        if gt_xz_full is not None:
            gt_xz = gt_xz_full[: i + 1]

        # Loop closure segments in (x, z) using current KF positions
        loop_segs = []
        for a_id, b_id in self.loop_edges:
            a = self.keyframes[a_id].T_w_c[:3, 3]
            b = self.keyframes[b_id].T_w_c[:3, 3]
            loop_segs.append((np.array([a[0], a[2]]), np.array([b[0], b[2]])))

        overlay_lines = [
            f"Frame {i} / {len(self.left_paths) - 1}",
            f"Keyframes: {len(self.keyframes)}",
            f"Loop closures: {total_loop_closures}",
        ]
        if gt_xz is not None and len(gt_xz) > 0 and len(traj_xz) > 0:
            drift = float(np.linalg.norm(traj_xz[-1] - gt_xz[-1]))
            overlay_lines.append(f"Drift vs GT: {drift:.1f} m")

        self.video_writer.write(
            left_img,
            sf.kps,
            traj_xz,
            map_xz,
            gt_xz=gt_xz,
            loop_segments_xz=loop_segs,
            overlay_lines=overlay_lines,
        )

    # -------- loop closure --------

    def _verify_loop(
        self, curr_kf: Keyframe, old_kf: Keyframe
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Run descriptor match + PnP verification between two keyframes.

        Returns (T_old_curr, inlier_count) on success, where T_old_curr is the
        relative transform from old_kf's frame to curr_kf's frame (i.e. the
        edge measurement for the pose graph), or None on failure.
        """
        if len(old_kf.desc) < 2 or len(curr_kf.desc) < 2:
            return None

        # Raw-trajectory proximity gate. raw_poses is never touched by PGO,
        # so its distances always reflect the pure VO estimate. Spurious
        # matches between visually-similar-but-physically-distant places
        # show up with large raw distances.
        p_old_raw = self.raw_poses[old_kf.frame_idx][:3, 3]
        p_curr_raw = self.raw_poses[curr_kf.frame_idx][:3, 3]
        if (
            float(np.linalg.norm(p_curr_raw - p_old_raw))
            > self.config.loop_max_raw_dist
        ):
            return None

        knn = self.matcher.knnMatch(curr_kf.desc, old_kf.desc, k=2)
        pts_2d = []
        pts_3d_old_cam = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= self.config.ratio_test * n.distance:
                continue
            pts_2d.append(curr_kf.kps[m.queryIdx].pt)
            pts_3d_old_cam.append(old_kf.pts3d_cam[m.trainIdx])

        if len(pts_2d) < self.config.loop_min_matches:
            return None

        pts_2d = np.asarray(pts_2d, dtype=np.float64)
        pts_3d_old_cam = np.asarray(pts_3d_old_cam, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_old_cam,
            pts_2d,
            self.calib.K,
            np.zeros(5),
            reprojectionError=self.config.pnp_reproj_err_px,
            confidence=self.config.pnp_confidence,
        )
        if (
            not success
            or inliers is None
            or len(inliers) < self.config.loop_min_inliers
        ):
            return None

        # cv2 returns T that takes OLD's camera-frame points into CURR's camera frame.
        R, _ = cv2.Rodrigues(rvec)
        T_curr_old = form_T(R, tvec)
        T_old_curr = invert_T(T_curr_old)

        if np.linalg.norm(T_old_curr[:3, 3]) > self.config.loop_max_rel_trans:
            return None  # implausibly large relative translation, likely spurious

        return T_old_curr, int(len(inliers))

    def _detect_loop_closures(
        self, curr_kf: Keyframe
    ) -> List[Tuple[int, np.ndarray, int]]:
        """Search older keyframes for valid loop closures to ``curr_kf``."""
        cutoff = curr_kf.id - self.config.loop_skip_recent
        if cutoff <= 0:
            return []
        detections: List[Tuple[int, np.ndarray, int]] = []
        for old_kf in self.keyframes[:cutoff]:
            out = self._verify_loop(curr_kf, old_kf)
            if out is None:
                continue
            T_rel, inliers = out
            detections.append((old_kf.id, T_rel, inliers))
        return detections

    # -------- driver --------

    # -------- driver --------

    def run(self) -> None:
        prev_X_w: Optional[np.ndarray] = None
        prev_desc: Optional[np.ndarray] = None
        prev_sf: Optional[StereoFrame] = None
        prev_frame_idx = -1
        frames_since_kf = 0
        total_loop_closures = 0

        # Pre-extract GT for the video overlay
        gt_xz_full: Optional[np.ndarray] = None
        if self.gt_poses is not None:
            gt_xz_full = np.array(
                [[T[0, 3], T[2, 3]] for T in self.gt_poses], dtype=np.float64
            )

        print(f"Aloitetaan stereo-SLAM ({len(self.left_paths)} kuvaa)...")

        for i, (lp, rp) in enumerate(zip(self.left_paths, self.right_paths)):
            L = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
            R = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
            if L is None or R is None:
                raise RuntimeError(f"failed to read frame {i}: {lp} / {rp}")

            sf = get_stereo_frame(
                L,
                R,
                self.calib,
                self.detector,
                self.matcher,
                ratio_test=self.config.ratio_test,
                max_epipolar_err_px=self.config.max_epipolar_err_px,
                min_depth=self.config.min_depth,
                max_depth=self.config.max_depth,
            )

            # --- front-end pose tracking (raw VO) ---
            if i == 0:
                T_w_c_raw = np.eye(4)
            else:
                T_w_c_raw = self._track_pose(prev_X_w, prev_desc, sf.kps, sf.desc)
                if T_w_c_raw is None:
                    self.pnp_failures += 1
                    print(f"  frame {i}: PnP failed, holding previous pose")
                    T_w_c_raw = self.raw_poses[-1]

            self.raw_poses.append(T_w_c_raw)
            frames_since_kf += 1

            # --- tracking carry-over lives in the raw VO frame, ALWAYS ---
            # This keeps the front-end decoupled from the back-end. PGO never
            # reaches back into prev_X_w, so raw_poses stays self-consistent.
            if len(sf.pts3d_cam) > 0:
                X_w_raw = (T_w_c_raw[:3, :3] @ sf.pts3d_cam.T + T_w_c_raw[:3, 3:4]).T
                prev_X_w = X_w_raw
                prev_desc = sf.desc
                prev_sf = sf
                prev_frame_idx = i

            # --- compute live pose for current frame (before KF insertion so
            # the look-up hits the previous keyframe) ---
            T_w_c_live_current = self._compute_live_pose(i, T_w_c_raw)

            # --- keyframe insertion + loop closure + online optimization ---
            optimized_this_iter = False
            insert_kf = i == 0 or frames_since_kf >= self.config.keyframe_interval
            if insert_kf and len(sf.pts3d_cam) > 0:
                # Insert with the LIVE pose so the new KF enters the graph in
                # the (already corrected) world frame. The odometry edge is
                # computed from raw_poses inside _insert_keyframe, so it's
                # invariant to the world frame.
                kf = self._insert_keyframe(i, T_w_c_live_current, sf)
                frames_since_kf = 0

                detections = self._detect_loop_closures(kf)
                for old_id, T_rel, inliers in detections:
                    self.pose_graph.add_edge(old_id, kf.id, T_rel)
                    self.loop_edges.append((old_id, kf.id))
                    total_loop_closures += 1
                    print(
                        f"  frame {i}: LOOP KF{old_id} <-> KF{kf.id} "
                        f"(inliers={inliers})"
                    )

                if detections:
                    self._optimize_pose_graph()
                    optimized_this_iter = True

            # --- live pose bookkeeping ---
            if optimized_this_iter:
                # Keyframes moved; rebuild every live pose so the trajectory
                # reflects the correction immediately on the video + plot.
                self._rebuild_live_poses()
            else:
                self.live_poses.append(T_w_c_live_current)

            # --- video frame ---
            if self.video_writer is not None:
                self._write_video_frame(i, L, sf, total_loop_closures, gt_xz_full)

            if i % 50 == 0:
                map_pts, _ = self._get_world_map()
                print(
                    f"Frame {i} valmis. KF={len(self.keyframes)}, "
                    f"loopit={total_loop_closures}, kartta={len(map_pts)}"
                )

        if self.video_writer is not None:
            self.video_writer.close()

        print(
            f"Front-end valmis. Keyframes: {len(self.keyframes)}, "
            f"loop-sulkeumia: {total_loop_closures}, "
            f"PnP-epäonnistumisia: {self.pnp_failures}"
        )


def trajectory_error(
    poses: List[np.ndarray], gt_poses: List[np.ndarray]
) -> Tuple[float, float]:
    n = min(len(poses), len(gt_poses))
    est = np.array([poses[i][:3, 3] for i in range(n)])
    gt = np.array([gt_poses[i][:3, 3] for i in range(n)])
    d = np.linalg.norm(est - gt, axis=1)
    return float(d[-1]), float(np.sqrt(np.mean(d**2)))


def save_kitti_trajectory(poses: List[np.ndarray], path: str) -> None:
    """Write a list of 4x4 poses as a KITTI-format trajectory file.

    Each line is the row-major flattening of the top 3x4 block of the pose
    matrix (12 floats separated by spaces) — the exact format used by KITTI
    ``poses/XX.txt`` and by the official ``evaluate_odometry`` devkit.
    """
    with open(path, "w") as f:
        for T in poses:
            row = T[:3, :].reshape(-1)
            f.write(" ".join(f"{v:.6e}" for v in row) + "\n")


# KITTI odometry benchmark sub-sequence lengths, in meters.
KITTI_SEGMENT_LENGTHS: Tuple[float, ...] = (
    100.0,
    200.0,
    300.0,
    400.0,
    500.0,
    600.0,
    700.0,
    800.0,
)


def kitti_segment_errors(
    poses_est: List[np.ndarray],
    poses_gt: List[np.ndarray],
    lengths: Tuple[float, ...] = KITTI_SEGMENT_LENGTHS,
    step: int = 10,
) -> dict:
    """Compute KITTI-style segment-based drift metrics.

    Mirrors the official ``evaluate_odometry`` devkit logic:

    1. Walk the ground-truth trajectory, accumulating path length ``s_k``.
    2. For each starting frame ``i`` (stride ``step``) and each segment
       length ``L``, find the first frame ``j`` such that ``s_j - s_i >= L``.
    3. Compute the relative transform between frames ``i`` and ``j`` in both
       the estimate and the ground truth, and measure the error as
       ``E = (ΔT_gt)^{-1} (ΔT_est)``.
    4. Translational error ``e_t = |E[:3, 3]| / L`` (a fraction).
       Rotational error ``e_r = arccos((tr R_err - 1)/2) / L`` (rad/m).
    5. Average across every valid (start, length) pair.

    Returns a dict:

    - ``per_length``: ``{L: (t_err_pct, r_err_deg_per_m)}``
    - ``overall``: ``(avg_t_err_pct, avg_r_err_deg_per_m)``
    - ``n_segments``: total count of evaluated segments

    Percentages are in percent translation drift (e.g. 1.5 = 1.5 %/100m),
    and rotations are in degrees per meter traveled. These are the units
    that appear in every published KITTI VO table.
    """
    n = min(len(poses_est), len(poses_gt))
    if n < 2:
        return {
            "per_length": {},
            "overall": (float("nan"), float("nan")),
            "n_segments": 0,
        }

    # Cumulative path length along the GT translations.
    gt_t = np.array([T[:3, 3] for T in poses_gt[:n]], dtype=np.float64)
    diffs = np.diff(gt_t, axis=0)
    step_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step_lens)])  # shape (n,)

    per_length: dict = {}
    all_t: List[float] = []
    all_r: List[float] = []

    for L in lengths:
        t_errs: List[float] = []
        r_errs: List[float] = []
        for i in range(0, n, step):
            target = cum[i] + L
            # np.searchsorted gives the first index where cum >= target
            j = int(np.searchsorted(cum, target))
            if j >= n:
                break  # trajectory too short to reach length L from here

            dT_est = invert_T(poses_est[i]) @ poses_est[j]
            dT_gt = invert_T(poses_gt[i]) @ poses_gt[j]
            E = invert_T(dT_gt) @ dT_est

            t_err = float(np.linalg.norm(E[:3, 3])) / L  # fraction
            trace = float(E[0, 0] + E[1, 1] + E[2, 2])
            cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
            theta = float(np.arccos(cos_theta))
            r_err = theta / L  # rad/m

            t_errs.append(t_err)
            r_errs.append(r_err)

        if t_errs:
            per_length[L] = (
                float(np.mean(t_errs)) * 100.0,  # → %
                float(np.mean(r_errs)) * 180.0 / np.pi,  # → deg/m
            )
            all_t.extend(t_errs)
            all_r.extend(r_errs)

    if all_t:
        overall = (
            float(np.mean(all_t)) * 100.0,
            float(np.mean(all_r)) * 180.0 / np.pi,
        )
    else:
        overall = (float("nan"), float("nan"))

    return {
        "per_length": per_length,
        "overall": overall,
        "n_segments": len(all_t),
    }


def plot_slam_results(
    slam: StereoSlam,
    poses_raw: List[np.ndarray],
    poses_opt: List[np.ndarray],
    out_path: str,
) -> None:
    raw_xy = np.array([T[:3, 3] for T in poses_raw])
    opt_xy = np.array([T[:3, 3] for T in poses_opt])

    plt.figure(figsize=(12, 9))

    # --- 3D map (built from optimized keyframes, so it follows PGO) ---
    X_w, _ = slam._get_world_map()
    if len(X_w) > 0:
        plt.scatter(
            X_w[:, 0], X_w[:, 2], s=0.5, c="lightgray", alpha=0.3, label="3D Map Points"
        )

    # --- ground truth ---
    if slam.gt_poses is not None:
        gt = np.array([T[:3, 3] for T in slam.gt_poses[: len(poses_opt)]])
        plt.plot(gt[:, 0], gt[:, 2], "g-", linewidth=2.0, label="Ground Truth")

    # --- raw stereo VO (no loop closures) ---
    plt.plot(
        raw_xy[:, 0], raw_xy[:, 2], "b:", linewidth=1.0, label="Stereo VO (no loops)"
    )

    # --- optimized trajectory ---
    plt.plot(
        opt_xy[:, 0], opt_xy[:, 2], "r-", linewidth=1.5, label="Stereo SLAM (+ PGO)"
    )

    # --- loop closure edges ---
    for a_id, b_id in slam.loop_edges:
        a = slam.keyframes[a_id].T_w_c[:3, 3]
        b = slam.keyframes[b_id].T_w_c[:3, 3]
        plt.plot([a[0], b[0]], [a[2], b[2]], "m-", linewidth=0.8, alpha=0.6)
    if slam.loop_edges:
        # Add a proxy legend entry for the loop closure lines
        plt.plot(
            [], [], "m-", linewidth=0.8, label=f"Loop closures ({len(slam.loop_edges)})"
        )

    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("Stereo SLAM with Loop Closures + Pose-Graph Optimization")
    plt.legend()
    plt.savefig(out_path, dpi=300)
    print(f"Lopputulos tallennettu: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stereo SLAM on a KITTI sequence.")
    parser.add_argument(
        "--sequence",
        "-s",
        default="00",
        help="KITTI odometry sequence number (e.g. 00, 01, ..., 10).",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory to write <sequence>.png and <sequence>.mp4 into.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable the side-by-side visualization video.",
    )
    parser.add_argument(
        "--detector",
        choices=("orb", "sift", "akaze"),
        default="orb",
        help="Feature detector/descriptor to use (default: orb).",
    )
    args = parser.parse_args()

    sequence = args.sequence
    paths = paths_for_sequence(sequence)
    os.makedirs(args.results_dir, exist_ok=True)
    plot_path = os.path.join(args.results_dir, f"{sequence}.png")
    traj_path = os.path.join(args.results_dir, f"{sequence}.txt")
    video_path = (
        None if args.no_video else os.path.join(args.results_dir, f"{sequence}.mp4")
    )

    config = SlamConfig(video_path=video_path, detector=args.detector)
    slam = StereoSlam(
        calib_path=paths["calib_path"],
        left_dir=paths["left_dir"],
        right_dir=paths["right_dir"],
        poses_path=paths["poses_path"],
        config=config,
    )
    slam.run()

    # Save the optimized trajectory in KITTI format for benchmarking +
    # offline evaluation (compatible with the official evaluate_odometry
    # devkit and with scripts/kitti_summary.py).
    save_kitti_trajectory(slam.live_poses, traj_path)
    print(f"[seq {sequence}] trajectory saved: {traj_path}")

    if slam.gt_poses is not None:
        # Absolute-error numbers (global position distance, not the KITTI
        # metric — useful for a quick eyeball of drift magnitude).
        final_raw, rmse_raw = trajectory_error(slam.raw_poses, slam.gt_poses)
        final_opt, rmse_opt = trajectory_error(slam.live_poses, slam.gt_poses)
        print(
            f"[seq {sequence}] Raw stereo VO:      "
            f"final err {final_raw:7.2f} m, RMSE {rmse_raw:6.2f} m"
        )
        print(
            f"[seq {sequence}] Stereo SLAM (+PGO): "
            f"final err {final_opt:7.2f} m, RMSE {rmse_opt:6.2f} m"
        )

        # KITTI benchmark-protocol metrics: segment-based drift rates over
        # sub-trajectories of 100-800 m. This is the number that appears in
        # every published KITTI VO table.
        kitti_raw = kitti_segment_errors(slam.raw_poses, slam.gt_poses)
        kitti_opt = kitti_segment_errors(slam.live_poses, slam.gt_poses)
        t_raw, r_raw = kitti_raw["overall"]
        t_opt, r_opt = kitti_opt["overall"]
        print(
            f"[seq {sequence}] KITTI raw VO:       "
            f"t_err {t_raw:6.2f} %,  r_err {r_raw*1000:6.3f} deg/km "
            f"({kitti_raw['n_segments']} segs)"
        )
        print(
            f"[seq {sequence}] KITTI SLAM (+PGO):  "
            f"t_err {t_opt:6.2f} %,  r_err {r_opt*1000:6.3f} deg/km "
            f"({kitti_opt['n_segments']} segs)"
        )
        if kitti_opt["per_length"]:
            print(f"[seq {sequence}] KITTI per-length (SLAM):")
            for L, (t, r) in sorted(kitti_opt["per_length"].items()):
                print(
                    f"[seq {sequence}]   L={int(L):4d} m: "
                    f"t_err {t:6.2f} %,  r_err {r*1000:6.3f} deg/km"
                )

    plot_slam_results(slam, slam.raw_poses, slam.live_poses, plot_path)


if __name__ == "__main__":
    main()
