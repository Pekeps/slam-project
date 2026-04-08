import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Frame:
    img: np.ndarray
    kps: list
    desc: Optional[np.ndarray]


class SparseMap:
    def __init__(self):
        self.X_w = []
        self.desc = []

    def add(self, X_w: np.ndarray, desc: np.ndarray):
        for i in range(len(X_w)):
            self.X_w.append(X_w[i])
            self.desc.append(desc[i])

    def as_arrays(self):
        if len(self.X_w) == 0:
            return np.zeros((0, 3)), np.zeros((0, 32), dtype=np.uint8)
        return np.asarray(self.X_w, dtype=np.float64), np.asarray(
            self.desc, dtype=np.uint8
        )


def read_kitti_K(calib_path: str, cam: str = "P0"):
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith(cam + ":"):
                vals = np.array([float(x) for x in line.split()[1:]], dtype=np.float64)
                P = vals.reshape(3, 4)
                K = P[:, :3]
                return K, P
    raise ValueError(f"{cam} not found in {calib_path}")


def load_image_paths(img_dir: str) -> List[str]:
    paths = sorted(
        glob.glob(os.path.join(img_dir, "*.png"))
        + glob.glob(os.path.join(img_dir, "*.jpg"))
    )
    if len(paths) == 0:
        raise FileNotFoundError(f"Kuvia ei löytynyt polusta: {img_dir}")
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


def triangulate_world(K, T_w_c0, T_w_c1, q0, q1):
    T_c0_w, T_c1_w = invert_T(T_w_c0), invert_T(T_w_c1)
    P0, P1 = K @ T_c0_w[:3, :], K @ T_c1_w[:3, :]
    X_h = cv2.triangulatePoints(P0, P1, q0.T, q1.T)
    return (X_h[:3, :] / (X_h[3, :] + 1e-12)).T


def reproj_err_and_depth(K, T_w_c, X_w, q_px):
    T_c_w = invert_T(T_w_c)
    X_c = (T_c_w[:3, :3] @ X_w.T + T_c_w[:3, 3:4]).T
    x = (K @ X_c.T).T
    proj = x[:, :2] / x[:, 2:3]
    return np.linalg.norm(proj - q_px, axis=1), X_c[:, 2]
