"""Shared helpers used by the stereo SLAM pipeline.

Two groups of utilities live here:

* KITTI-specific I/O (``load_image_paths``, ``load_poses_kitti``) and a few
  small rigid-transform primitives (``form_T``, ``invert_T``).
* SE(3) Lie-group operations used by the pose-graph optimizer:
  ``skew``, ``se3_exp``, ``pose_error_6d`` (scalar versions) plus their
  batched counterparts ``rigid_inverse_batch``, ``se3_exp_batch``, and
  ``rot_to_angle_axis_batch`` for the vectorised residual function.
"""

import glob
import os
from typing import List, Optional

import cv2
import numpy as np


# === KITTI I/O ===


def load_image_paths(img_dir: str) -> List[str]:
    paths = sorted(
        glob.glob(os.path.join(img_dir, "*.png"))
        + glob.glob(os.path.join(img_dir, "*.jpg"))
    )
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {img_dir}")
    return paths


def load_poses_kitti(poses_path: str) -> Optional[List[np.ndarray]]:
    if not os.path.exists(poses_path):
        return None
    poses = []
    with open(poses_path, "r") as f:
        for line in f:
            T = np.fromstring(line, dtype=np.float64, sep=" ").reshape(3, 4)
            T = np.vstack([T, [0, 0, 0, 1]])
            poses.append(T)
    return poses


# === Rigid-transform primitives ===


def form_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# === SE(3) Lie group helpers (scalar) ===
# xi = [rho (3,), phi (3,)] in R^6, where rho is translation and phi is
# the so(3) axis-angle (magnitude = angle, direction = rotation axis).


def skew(v: np.ndarray) -> np.ndarray:
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map se(3) -> SE(3). Input is 6-vec [rho; phi]."""
    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))
    phi_hat = skew(phi)

    T = np.eye(4, dtype=np.float64)
    if theta < 1e-10:
        # First-order expansion around identity
        T[:3, :3] = np.eye(3) + phi_hat
        V = np.eye(3) + 0.5 * phi_hat
    else:
        phi_hat_sq = phi_hat @ phi_hat
        s = np.sin(theta)
        c = np.cos(theta)
        T[:3, :3] = (
            np.eye(3)
            + (s / theta) * phi_hat
            + ((1.0 - c) / (theta * theta)) * phi_hat_sq
        )
        V = (
            np.eye(3)
            + ((1.0 - c) / (theta * theta)) * phi_hat
            + ((theta - s) / (theta**3)) * phi_hat_sq
        )
    T[:3, 3] = V @ rho
    return T


def pose_error_6d(T_err: np.ndarray) -> np.ndarray:
    """Turn an SE(3) error (identity = zero error) into a 6-vec residual.

    Uses the translation directly and Rodrigues angle-axis for rotation.
    Not the full SE(3) log, but exactly zero at the minimum and well-behaved
    for small residuals, which is what scipy's least-squares cares about.
    """
    R_err = T_err[:3, :3]
    t_err = T_err[:3, 3]
    rvec, _ = cv2.Rodrigues(R_err)
    return np.concatenate([t_err, rvec.ravel()])


# === Batched SE(3) helpers ===
# Pure-numpy operations over leading batch dimensions. Used by the pose-graph
# optimizer to avoid the Python-level per-edge loop, which is the dominant
# cost of each residual evaluation at graph sizes >~100 vertices.


def rigid_inverse_batch(T: np.ndarray) -> np.ndarray:
    """Batched closed-form rigid inverse for SE(3) matrices.

    For ``T = [[R, t], [0, 1]]`` the inverse is ``[[R.T, -R.T @ t], [0, 1]]``,
    which avoids the general 4x4 matrix inverse and is much faster.

    Parameters
    ----------
    T : np.ndarray, shape (..., 4, 4)

    Returns
    -------
    np.ndarray, shape (..., 4, 4)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_T = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = R_T
    out[..., :3, 3:4] = -(R_T @ t)
    out[..., 3, 3] = 1.0
    return out


def se3_exp_batch(xi: np.ndarray) -> np.ndarray:
    """Batched SE(3) exponential map.

    Parameters
    ----------
    xi : np.ndarray, shape (..., 6), layout [rho; phi]
        ``rho`` (first three entries) is the translation part; ``phi`` (last
        three entries) is the axis-angle rotation part.

    Returns
    -------
    np.ndarray, shape (..., 4, 4)
    """
    rho = xi[..., :3]
    phi = xi[..., 3:]
    batch_shape = xi.shape[:-1]

    # Skew-symmetric matrix of phi (the hat operator applied to each element
    # of the batch).
    phi_hat = np.zeros(batch_shape + (3, 3), dtype=xi.dtype)
    phi_hat[..., 0, 1] = -phi[..., 2]
    phi_hat[..., 0, 2] = phi[..., 1]
    phi_hat[..., 1, 0] = phi[..., 2]
    phi_hat[..., 1, 2] = -phi[..., 0]
    phi_hat[..., 2, 0] = -phi[..., 1]
    phi_hat[..., 2, 1] = phi[..., 0]

    phi_hat_sq = phi_hat @ phi_hat

    theta = np.linalg.norm(phi, axis=-1)  # (...,)
    small = theta < 1e-8
    theta_safe = np.where(small, 1.0, theta)
    sin_t = np.sin(theta_safe)
    cos_t = np.cos(theta_safe)

    # Taylor expansions for the small-angle branch:
    #   sin(t)/t          ≈ 1
    #   (1-cos(t))/t^2    ≈ 1/2
    #   (t - sin(t))/t^3  ≈ 1/6
    a = np.where(small, 1.0, sin_t / theta_safe)
    b = np.where(small, 0.5, (1.0 - cos_t) / (theta_safe * theta_safe))
    d = np.where(small, 1.0 / 6.0, (theta_safe - sin_t) / (theta_safe**3))

    # Broadcast scalar coefficients against 3x3 phi_hat
    a = a[..., np.newaxis, np.newaxis]
    b = b[..., np.newaxis, np.newaxis]
    d = d[..., np.newaxis, np.newaxis]

    I3 = np.eye(3, dtype=xi.dtype)
    R = I3 + a * phi_hat + b * phi_hat_sq
    V = I3 + b * phi_hat + d * phi_hat_sq

    V_rho = np.einsum("...ij,...j->...i", V, rho)  # (..., 3)

    out = np.zeros(batch_shape + (4, 4), dtype=xi.dtype)
    out[..., :3, :3] = R
    out[..., :3, 3] = V_rho
    out[..., 3, 3] = 1.0
    return out


def rot_to_angle_axis_batch(R: np.ndarray) -> np.ndarray:
    """Batched SO(3) log map via the angle-axis representation.

    Uses the standard Rodrigues-inverse formula with a small-angle fallback.
    Degenerates near rotation angle ``pi``; this is acceptable for
    pose-graph residuals which should never see rotations that large once
    the estimate is anywhere near the optimum.

    Parameters
    ----------
    R : np.ndarray, shape (..., 3, 3)

    Returns
    -------
    np.ndarray, shape (..., 3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = np.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Asymmetric part of R: equals 2*sin(theta) * axis
    asym = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )

    sin_theta = np.sin(theta)
    safe_sin = np.where(np.abs(sin_theta) < 1e-12, 1.0, sin_theta)
    # rvec = (theta / (2 sin theta)) * asym
    # For small theta: asym ≈ 2*theta*axis, so 0.5 * asym ≈ theta * axis = rvec
    small = theta < 1e-6
    scale = np.where(small, 0.5, theta / (2.0 * safe_sin))
    return asym * scale[..., np.newaxis]
