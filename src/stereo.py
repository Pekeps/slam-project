"""Stereo support: calibration loading and sparse stereo depth estimation.

The depth estimator detects features independently in the left and right
images, ratio-tests their descriptor matches, applies a rectified-pair
sanity filter (small epipolar residual + positive disparity), then
triangulates the survivors with the KITTI projection matrices to get
metric 3D points in the LEFT camera frame.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class StereoCalib:
    K: np.ndarray  # 3x3 intrinsics (shared by both rectified cameras)
    P0: np.ndarray  # 3x4 left projection in left-camera frame
    P1: np.ndarray  # 3x4 right projection in left-camera frame
    baseline: float  # meters, positive
    fx: float  # K[0, 0]


@dataclass
class StereoFrame:
    kps: List[cv2.KeyPoint]  # keypoints in the LEFT image
    desc: np.ndarray  # Nx32 (ORB) descriptors aligned with kps
    pts3d_cam: np.ndarray  # Nx3 3D points in the LEFT camera frame


def read_kitti_stereo_calib(calib_path: str) -> StereoCalib:
    """Parse a KITTI odometry-format calib.txt and return left+right intrinsics.

    KITTI's P1 stores [-fx*baseline, 0, 0] in its right column, so the
    baseline (in meters) is recovered as -P1[0, 3] / fx.
    """
    P0 = None
    P1 = None
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P0:"):
                P0 = np.array(
                    [float(x) for x in line.split()[1:]], dtype=np.float64
                ).reshape(3, 4)
            elif line.startswith("P1:"):
                P1 = np.array(
                    [float(x) for x in line.split()[1:]], dtype=np.float64
                ).reshape(3, 4)
    if P0 is None or P1 is None:
        raise ValueError(f"P0 and/or P1 not found in {calib_path}")
    K = P0[:, :3]
    fx = float(K[0, 0])
    baseline = float(-P1[0, 3] / fx)
    return StereoCalib(K=K, P0=P0, P1=P1, baseline=baseline, fx=fx)


def estimate_stereo_depth(
    left_img: np.ndarray,
    right_img: np.ndarray,
    calib: StereoCalib,
    detector,
    matcher,
    ratio_test: float = 0.7,
    max_epipolar_err_px: float = 1.5,
    min_depth: float = 0.1,
    max_depth: float = 200.0,
) -> StereoFrame:
    """Detect features in both stereo images and triangulate metric 3D points.

    Returns a StereoFrame whose entries are all index-aligned: ``kps[i]``,
    ``desc[i]`` and ``pts3d_cam[i]`` describe the same physical point.
    """
    kps_L, desc_L = detector.detectAndCompute(left_img, None)
    kps_R, desc_R = detector.detectAndCompute(right_img, None)

    if desc_L is None or desc_R is None or len(kps_L) == 0 or len(kps_R) == 0:
        return _empty_stereo_frame()

    # knnMatch needs at least k=2 candidates per query
    if len(kps_R) < 2:
        return _empty_stereo_frame()

    knn = matcher.knnMatch(desc_L, desc_R, k=2)

    pts_L = []
    pts_R = []
    keep_idx = []  # indices into kps_L / desc_L that survive the ratio test
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance >= ratio_test * n.distance:
            continue
        pts_L.append(kps_L[m.queryIdx].pt)
        pts_R.append(kps_R[m.trainIdx].pt)
        keep_idx.append(m.queryIdx)

    if len(pts_L) == 0:
        return _empty_stereo_frame()

    pts_L = np.asarray(pts_L, dtype=np.float64)
    pts_R = np.asarray(pts_R, dtype=np.float64)

    # Rectified-pair sanity filter: epipolar lines are horizontal, so the
    # vertical coordinate should agree, and disparity (uL - uR) must be
    # strictly positive for a point in front of the camera.
    epi_err = np.abs(pts_L[:, 1] - pts_R[:, 1])
    disparity = pts_L[:, 0] - pts_R[:, 0]
    sane = (epi_err < max_epipolar_err_px) & (disparity > 0)
    if not np.any(sane):
        return _empty_stereo_frame()

    pts_L = pts_L[sane]
    pts_R = pts_R[sane]
    keep_idx = [keep_idx[i] for i in np.flatnonzero(sane)]

    # Triangulate in the LEFT camera frame using the KITTI P0/P1.
    X_h = cv2.triangulatePoints(calib.P0, calib.P1, pts_L.T, pts_R.T)
    X = (X_h[:3, :] / (X_h[3, :] + 1e-12)).T  # Nx3

    Z = X[:, 2]
    depth_ok = (Z > min_depth) & (Z < max_depth) & np.isfinite(Z)
    if not np.any(depth_ok):
        return _empty_stereo_frame()

    X = X[depth_ok]
    keep_idx = [keep_idx[i] for i in np.flatnonzero(depth_ok)]

    kept_kps = [kps_L[i] for i in keep_idx]
    kept_desc = desc_L[keep_idx]

    return StereoFrame(kps=kept_kps, desc=kept_desc, pts3d_cam=X)


def _empty_stereo_frame() -> StereoFrame:
    return StereoFrame(
        kps=[],
        desc=np.zeros((0, 32), dtype=np.uint8),
        pts3d_cam=np.zeros((0, 3), dtype=np.float64),
    )
