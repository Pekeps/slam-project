"""Stereo visual SLAM with loop closures and online pose-graph optimization.

Front-end: frame-to-frame stereo VO, same shape as stereo_vo.py.
Back-end: a scipy-based SE(3) pose graph with sequential odometry edges
plus loop-closure edges detected by descriptor matching + PnP verification
against older keyframes.

**Pose-graph optimization runs online**: every time one or more loop
closures are detected on a new keyframe, the pose graph is re-optimized
immediately. The optimized keyframe poses are then propagated to the
in-between non-keyframe frames so that subsequent tracking (and the video
visualization) benefits from the correction right away.

Two trajectory arrays are maintained:

* ``raw_poses``  — the pure VO trajectory, never touched by optimization.
  Kept around for the before/after comparison on the final static plot.
* ``live_poses`` — the currently-believed trajectory, updated whenever the
  pose graph is re-optimized. Drives the video visualization.

Ground-truth poses, if present, are loaded purely for evaluation/plotting.
"""

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
    pose_error_6d,
    se3_exp,
)
from slam_video import SlamVideoWriter
from stereo import (
    StereoCalib,
    StereoFrame,
    estimate_stereo_depth,
    read_kitti_stereo_calib,
)


# === POLUT ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SEQUENCE = "00"
CALIB_PATH = os.path.join(
    DATA_DIR, "data_odometry_calib", "dataset", "sequences", SEQUENCE, "calib.txt"
)
POSES_PATH = os.path.join(
    DATA_DIR, "data_odometry_poses", "dataset", "poses", f"{SEQUENCE}.txt"
)
LEFT_DIR = os.path.join(
    DATA_DIR, "data_odometry_gray", "dataset", "sequences", SEQUENCE, "image_0"
)
RIGHT_DIR = os.path.join(
    DATA_DIR, "data_odometry_gray", "dataset", "sequences", SEQUENCE, "image_1"
)


@dataclass
class SlamConfig:
    max_frames: int = 4000
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
    loop_min_inliers: int = 30  # PnP inliers required to accept a loop
    loop_max_rel_trans: float = 50.0  # sanity cap on recovered relative translation (m)

    # Pose-graph optimization
    pg_max_nfev: int = 5000  # plenty of headroom for finite-diff on ~1k params
    pg_ftol: float = 1e-6
    pg_xtol: float = 1e-6

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
        n = len(self.vertices)
        free = [k for k in range(n) if k not in self.fixed]
        free_lookup = {k: i for i, k in enumerate(free)}
        n_free = len(free)
        n_params = 6 * n_free
        n_res = 6 * len(self.edges)

        # Snapshot the linearization point — we parameterize around THESE poses
        # and never mutate them during the optimization. Only at the end do we
        # bake the optimized increments back into self.vertices.
        vertices0 = [T.copy() for T in self.vertices]

        # Sparsity pattern: each edge (i, j) only touches xi_i and xi_j.
        sparsity = lil_matrix((n_res, n_params), dtype=bool)
        for e_idx, (i, j, _) in enumerate(self.edges):
            r0 = 6 * e_idx
            for v in (i, j):
                if v in free_lookup:
                    p0 = 6 * free_lookup[v]
                    sparsity[r0 : r0 + 6, p0 : p0 + 6] = True

        def residuals(xi_flat: np.ndarray) -> np.ndarray:
            xi_free = xi_flat.reshape(n_free, 6)
            poses = [T for T in vertices0]  # fixed linearization point
            for i, k in enumerate(free):
                poses[k] = vertices0[k] @ se3_exp(xi_free[i])

            res = np.empty(n_res, dtype=np.float64)
            for e_idx, (i, j, T_ij_meas) in enumerate(self.edges):
                T_ij_pred = invert_T(poses[i]) @ poses[j]
                T_err = invert_T(T_ij_meas) @ T_ij_pred
                res[6 * e_idx : 6 * e_idx + 6] = pose_error_6d(T_err)
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

        # Bake the optimized increments into the vertex poses.
        xi_free = result.x.reshape(n_free, 6)
        for i, k in enumerate(free):
            self.vertices[k] = vertices0[k] @ se3_exp(xi_free[i])

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
        self.left_paths = load_image_paths(left_dir)[: self.config.max_frames]
        self.right_paths = load_image_paths(right_dir)[: self.config.max_frames]
        if len(self.left_paths) != len(self.right_paths):
            raise RuntimeError(
                f"left/right image count mismatch: "
                f"{len(self.left_paths)} vs {len(self.right_paths)}"
            )
        self.gt_poses: Optional[List[np.ndarray]] = None
        if poses_path is not None:
            gt = load_poses_kitti(poses_path)
            if gt is not None:
                self.gt_poses = gt[: self.config.max_frames]

        self.detector = cv2.ORB_create(nfeatures=self.config.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

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

    def _compute_live_pose(
        self, frame_idx: int, T_w_c_raw: np.ndarray
    ) -> np.ndarray:
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

            sf = estimate_stereo_depth(
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
                X_w_raw = (
                    T_w_c_raw[:3, :3] @ sf.pts3d_cam.T + T_w_c_raw[:3, 3:4]
                ).T
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
    slam = StereoSlam(
        calib_path=CALIB_PATH,
        left_dir=LEFT_DIR,
        right_dir=RIGHT_DIR,
        poses_path=POSES_PATH,
        config=SlamConfig(video_path="kitti_slam_video.mp4"),
    )
    slam.run()

    if slam.gt_poses is not None:
        final_raw, rmse_raw = trajectory_error(slam.raw_poses, slam.gt_poses)
        final_opt, rmse_opt = trajectory_error(slam.live_poses, slam.gt_poses)
        print(
            f"Raw stereo VO:       final err {final_raw:7.2f} m, RMSE {rmse_raw:6.2f} m"
        )
        print(
            f"Stereo SLAM (+PGO):  final err {final_opt:7.2f} m, RMSE {rmse_opt:6.2f} m"
        )

    plot_slam_results(slam, slam.raw_poses, slam.live_poses, "kitti_loop_result.png")


if __name__ == "__main__":
    main()
