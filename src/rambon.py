import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SEQUENCE = "00"
CALIB_PATH = os.path.join(
    DATA_DIR, "data_odometry_calib", "dataset", "sequences", SEQUENCE, "calib.txt"
)
POSES_PATH = os.path.join(
    DATA_DIR, "data_odometry_poses", "dataset", "poses", f"{SEQUENCE}.txt"
)
IMAGES_DIR = os.path.join(
    DATA_DIR, "data_odometry_gray", "dataset", "sequences", SEQUENCE, "image_0"
)

MAX_FRAMES = 400
BUILD_MAP_FRAMES = 300
DETECTOR_TYPE = "ORB"  # "ORB" tai "SIFT"
N_FEATURES = 3000
RATIO_TEST = 0.70
E_RANSAC_PROB = 0.999
E_RANSAC_THRESH_PX = 1.0
PNP_REPROJ_ERR_PX = 3.0
PNP_CONFIDENCE = 0.999
REPROJ_THRESH_PX = 3.0
MAX_DEPTH = 200.0
MIN_DEPTH = 0.1


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


# === APUFUNKTIOT ===
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


# === VISUAL ODOMETRY LUOKKA ===
class VisualOdometry:
    def __init__(self, calib_path: str, images_dir: str, poses_path: str):
        self.K, _ = read_kitti_K(calib_path)
        self.D = np.zeros(5)
        self.image_paths = load_image_paths(images_dir)[:MAX_FRAMES]
        self.gt_poses = load_poses_kitti(poses_path)
        if self.gt_poses:
            self.gt_poses = self.gt_poses[:MAX_FRAMES]
        self.detector = (
            cv2.ORB_create(nfeatures=N_FEATURES)
            if DETECTOR_TYPE == "ORB"
            else cv2.SIFT_create()
        )
        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING if DETECTOR_TYPE == "ORB" else cv2.NORM_L2
        )

    def frame(self, i):
        img = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE)
        kps, desc = self.detector.detectAndCompute(img, None)
        return Frame(img, kps, desc)

    def get_matches(self, f0, f1):
        matches = self.matcher.knnMatch(f0.desc, f1.desc, k=2)
        good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
        q0 = np.float32([f0.kps[m.queryIdx].pt for m in good])
        q1 = np.float32([f1.kps[m.trainIdx].pt for m in good])
        d1 = np.array([f1.desc[m.trainIdx] for m in good])
        return q0, q1, d1

    def get_pose_E(self, q0, q1):
        E, mask = cv2.findEssentialMat(
            q0,
            q1,
            self.K,
            method=cv2.RANSAC,
            prob=E_RANSAC_PROB,
            threshold=E_RANSAC_THRESH_PX,
        )
        _, R, t, mask_pose = cv2.recoverPose(E, q0, q1, self.K, mask=mask)
        return R, t, mask_pose

    def step_scale(self, i):
        if not self.gt_poses:
            return 1.0
        return np.linalg.norm(self.gt_poses[i][:3, 3] - self.gt_poses[i - 1][:3, 3])


def localize_pnp(vo, smap, img_gray):
    X_w, D_map = smap.as_arrays()
    kps, desc = vo.detector.detectAndCompute(img_gray, None)
    if desc is None or len(X_w) < 50:
        return None

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        desc.astype(np.uint8), D_map.astype(np.uint8), k=2
    )
    pts2d, pts3d = [], []
    for m, n in matches:
        if m.distance < RATIO_TEST * n.distance:
            pts2d.append(kps[m.queryIdx].pt)
            pts3d.append(X_w[m.trainIdx])

    if len(pts2d) < 10:
        return None
    success, rvec, tvec, _ = cv2.solvePnPRansac(
        np.array(pts3d), np.array(pts2d), vo.K, vo.D
    )
    if not success:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return invert_T(form_T(R, tvec))


def plot_final_results(vo, poses_w_c, loc_poses, smap, start_frame):
    # 1. Haetaan VO-reitti (se ketjutettu reitti)
    est_vo = np.array([T[:3, 3] for T in poses_w_c])

    # 2. Haetaan PnP-lokalisointipisteet
    loc_xy = []
    for T in loc_poses:
        if T is not None:
            loc_xy.append(T[:3, 3])
    loc_xy = np.array(loc_xy)

    plt.figure(figsize=(12, 8))

    # --- PIIRRETÄÄN PISTEKARTTA ---
    X_w, _ = smap.as_arrays()
    if len(X_w) > 0:
        # Otetaan vain joka 5. piste, jotta kuvaaja ei hidastu liikaa
        plt.scatter(
            X_w[::5, 0], X_w[::5, 2], s=0.5, c="gray", alpha=0.3, label="3D Map Points"
        )

    # --- PIIRRETÄÄN GROUND TRUTH (GT) ---
    if vo.gt_poses is not None:
        gt = np.array([T[:3, 3] for T in vo.gt_poses])
        plt.plot(gt[:, 0], gt[:, 2], "g-", linewidth=2, label="Ground Truth (GT)")

    # --- PIIRRETÄÄN VO-REITTI ---
    plt.plot(est_vo[:, 0], est_vo[:, 2], "b--", linewidth=1, label="VO Trajectory")

    # --- PIIRRETÄÄN PNP-LOKALISOINTI ---
    if len(loc_xy) > 0:
        plt.scatter(loc_xy[:, 0], loc_xy[:, 2], c="red", s=5, label="PnP Localization")

    # Merkitään kohta, jossa kartoitus loppui ja PnP alkoi
    boundary = est_vo[start_frame]
    plt.scatter(
        boundary[0],
        boundary[2],
        marker="X",
        s=100,
        c="black",
        label="Localization Start",
    )

    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("Visual Odometry vs. PnP Localization vs. Ground Truth")
    plt.legend()

    # Tallennetaan ja näytetään
    plt.savefig("kitti_final_result.png", dpi=300)
    print("Lopputulos tallennettu: kitti_final_result.png")
    # plt.show()


def main():
    vo = VisualOdometry(CALIB_PATH, IMAGES_DIR, POSES_PATH)
    smap = SparseMap()
    poses_w_c = [np.eye(4)]
    f0 = vo.frame(0)

    print(f"Aloitetaan VO ja kartoitus ({len(vo.image_paths)} kuvaa)...")

    for i in range(1, len(vo.image_paths)):
        f1 = vo.frame(i)
        q0, q1, d1 = vo.get_matches(f0, f1)
        R, t, inliers = vo.get_pose_E(q0, q1)

        T_w_c1 = poses_w_c[-1] @ invert_T(form_T(R, t * vo.step_scale(i)))
        poses_w_c.append(T_w_c1)

        # Kartan rakentaminen (triangulaatio)
        if i < BUILD_MAP_FRAMES and inliers is not None:
            inl = inliers.ravel().astype(bool)
            X_w = triangulate_world(
                vo.K, poses_w_c[-2], poses_w_c[-1], q0[inl], q1[inl]
            )
            err0, z0 = reproj_err_and_depth(vo.K, poses_w_c[-2], X_w, q0[inl])
            err1, z1 = reproj_err_and_depth(vo.K, poses_w_c[-1], X_w, q1[inl])

            keep = (
                (z0 > MIN_DEPTH)
                & (z0 < MAX_DEPTH)
                & (z1 > MIN_DEPTH)
                & (z1 < MAX_DEPTH)
                & (err0 < REPROJ_THRESH_PX)
                & (err1 < REPROJ_THRESH_PX)
            )
            smap.add(X_w[keep][::2], d1[inl][keep][::2])

        if i % 50 == 0:
            print(f"Frame {i} valmis. Karttapisteitä: {len(smap.X_w)}")
        f0 = f1

    # Lokalisointi PnP:llä
    print("Aloitetaan PnP-lokalisointi...")
    loc_poses = []
    for i in range(BUILD_MAP_FRAMES, len(vo.image_paths)):
        loc_poses.append(localize_pnp(vo, smap, vo.frame(i).img))

    # Piirretään tulokset
    plot_final_results(vo, poses_w_c, loc_poses, smap, BUILD_MAP_FRAMES)


if __name__ == "__main__":
    main()
