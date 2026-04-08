"""Stereo visual odometry on KITTI odometry sequences.

Frame-to-frame tracking: at each new frame, the stereo pair is
triangulated into metric 3D points in the current left-camera frame,
and the pose is recovered by matching the *previous* frame's stereo
points (already in world coordinates) against the *current* frame's
left-image observations via PnP RANSAC. No ground-truth scale recovery
is used — stereo gives us metric directly.

Ground-truth poses, if present, are loaded purely for the final
comparison plot.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam_utils import (
    SparseMap,
    form_T,
    invert_T,
    load_image_paths,
    load_poses_kitti,
)
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
class StereoVOConfig:
    max_frames: int = 4000
    n_features: int = 3000
    ratio_test: float = 0.70
    pnp_reproj_err_px: float = 3.0
    pnp_confidence: float = 0.999
    min_pnp_matches: int = 10
    max_epipolar_err_px: float = 1.5
    min_depth: float = 0.1
    max_depth: float = 200.0
    map_downsample: int = 5  # stride when adding points to the visualisation map


class StereoVisualOdometry:
    def __init__(
        self,
        calib_path: str,
        left_dir: str,
        right_dir: str,
        poses_path: Optional[str] = None,
        config: Optional[StereoVOConfig] = None,
    ):
        self.config = config or StereoVOConfig()

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
        self.smap = SparseMap()  # accumulated map, visualisation only
        self.pnp_failures = 0

    def _track_pose(
        self,
        prev_X_w: np.ndarray,
        prev_desc: np.ndarray,
        curr_kps: List[cv2.KeyPoint],
        curr_desc: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Match current left features to prev frame's stereo points and run PnP.

        Returns the camera-to-world transform ``T_w_c`` for the current
        frame, or ``None`` if not enough matches / PnP failed.
        """
        if prev_desc is None or len(prev_desc) == 0:
            return None
        if curr_desc is None or len(curr_desc) == 0:
            return None
        if len(prev_desc) < 2:
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
        T_c_w = form_T(R, tvec)
        return invert_T(T_c_w)

    def run(self) -> List[np.ndarray]:
        """Run the full stereo VO loop and return a list of T_w_c per frame."""
        poses_w_c: List[np.ndarray] = []
        prev_X_w: Optional[np.ndarray] = None
        prev_desc: Optional[np.ndarray] = None

        print(f"Aloitetaan stereo-VO ({len(self.left_paths)} kuvaa)...")

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

            if i == 0:
                T_w_c = np.eye(4)
            else:
                T_w_c = self._track_pose(prev_X_w, prev_desc, sf.kps, sf.desc)
                if T_w_c is None:
                    self.pnp_failures += 1
                    print(f"  frame {i}: PnP failed, holding previous pose")
                    T_w_c = poses_w_c[-1]

            poses_w_c.append(T_w_c)

            # Lift this frame's stereo 3D points into the world frame and use
            # them as the PnP target for the next iteration.
            if len(sf.pts3d_cam) > 0:
                X_w_curr = (T_w_c[:3, :3] @ sf.pts3d_cam.T + T_w_c[:3, 3:4]).T
                prev_X_w = X_w_curr
                prev_desc = sf.desc
                stride = self.config.map_downsample
                self.smap.add(X_w_curr[::stride], sf.desc[::stride])
            # If stereo returned nothing, carry over the previous track state.

            if i % 50 == 0:
                print(
                    f"Frame {i} valmis. Stereopisteitä: {len(sf.kps)}, "
                    f"karttapisteitä yht.: {len(self.smap.X_w)}"
                )

        print(
            f"Stereo-VO valmis. PnP-epäonnistumisia: "
            f"{self.pnp_failures}/{len(poses_w_c)}"
        )
        return poses_w_c


def trajectory_error(
    poses_w_c: List[np.ndarray], gt_poses: List[np.ndarray]
) -> Tuple[float, float]:
    """Final-position error and RMSE of translation between estimated and GT."""
    n = min(len(poses_w_c), len(gt_poses))
    est = np.array([poses_w_c[i][:3, 3] for i in range(n)])
    gt = np.array([gt_poses[i][:3, 3] for i in range(n)])
    diffs = np.linalg.norm(est - gt, axis=1)
    final_err = float(diffs[-1])
    rmse = float(np.sqrt(np.mean(diffs**2)))
    return final_err, rmse


def plot_stereo_results(
    vo: StereoVisualOdometry,
    poses_w_c: List[np.ndarray],
    out_path: str,
) -> None:
    est = np.array([T[:3, 3] for T in poses_w_c])

    plt.figure(figsize=(12, 8))

    # --- PIIRRETÄÄN PISTEKARTTA ---
    X_w, _ = vo.smap.as_arrays()
    if len(X_w) > 0:
        plt.scatter(
            X_w[:, 0], X_w[:, 2], s=0.5, c="gray", alpha=0.3, label="3D Map Points"
        )

    # --- PIIRRETÄÄN GT (vain vertailua varten) ---
    if vo.gt_poses is not None:
        gt = np.array([T[:3, 3] for T in vo.gt_poses[: len(poses_w_c)]])
        plt.plot(gt[:, 0], gt[:, 2], "g-", linewidth=2, label="Ground Truth (GT)")

    # --- PIIRRETÄÄN STEREO-VO REITTI ---
    plt.plot(est[:, 0], est[:, 2], "b-", linewidth=1.5, label="Stereo VO Trajectory")

    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("Stereo Visual Odometry vs. Ground Truth")
    plt.legend()

    plt.savefig(out_path, dpi=300)
    print(f"Lopputulos tallennettu: {out_path}")


def main() -> None:
    vo = StereoVisualOdometry(
        calib_path=CALIB_PATH,
        left_dir=LEFT_DIR,
        right_dir=RIGHT_DIR,
        poses_path=POSES_PATH,
    )
    poses = vo.run()

    if vo.gt_poses is not None:
        final_err, rmse = trajectory_error(poses, vo.gt_poses)
        print(f"Final position error vs GT: {final_err:.2f} m")
        print(f"Translation RMSE vs GT:     {rmse:.2f} m")

    plot_stereo_results(vo, poses, "kitti_stereo_result.png")


if __name__ == "__main__":
    main()
