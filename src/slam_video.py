"""Side-by-side video visualization of a running SLAM pipeline.

Each frame of the output video shows:

* **Left pane** — a top-down (X vs Z) view of the map: 3D map points, the
  currently-believed trajectory, optional ground truth, loop-closure edges,
  and the current camera position as a moving marker.
* **Right pane** — the current left-camera image with detected keypoints
  overlaid, plus small overlay text (frame number, KF count, loop count,
  drift vs GT).

The writer takes pre-extracted (x, z) arrays for trajectories/map points so
that the caller can apply any projection it wants without this helper
needing to know about the SLAM data model.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class SlamVideoWriter:
    def __init__(
        self,
        path: str,
        fps: int = 20,
        pane_w: int = 960,
        pane_h: int = 720,
        bounds_xz: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Parameters
        ----------
        path
            Output mp4 file path.
        fps
            Video frame rate.
        pane_w, pane_h
            Dimensions of each of the two panes (map on left, camera on right).
        bounds_xz
            (x_min, x_max, z_min, z_max) world bounds for the static map view.
            If None, the bounds expand dynamically each frame to fit the
            current data. Static bounds make the video much easier to follow.
        """
        self.pane_w = pane_w
        self.pane_h = pane_h
        self.total_w = pane_w * 2
        self.total_h = pane_h
        self.bounds_xz = bounds_xz

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (self.total_w, self.total_h))
        if not self.writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {path}")

    # ---------- geometry ----------

    def _get_bounds(
        self,
        trajectory_xz: np.ndarray,
        gt_xz: Optional[np.ndarray],
    ) -> Tuple[float, float, float, float]:
        if self.bounds_xz is not None:
            return self.bounds_xz
        parts = []
        if gt_xz is not None and len(gt_xz) > 0:
            parts.append(gt_xz)
        if len(trajectory_xz) > 0:
            parts.append(trajectory_xz)
        if not parts:
            return (-20.0, 20.0, -20.0, 20.0)
        combined = np.concatenate(parts, axis=0)
        xmin, xmax = float(combined[:, 0].min()), float(combined[:, 0].max())
        zmin, zmax = float(combined[:, 1].min()), float(combined[:, 1].max())
        if xmax - xmin < 10.0:
            mid = 0.5 * (xmin + xmax)
            xmin, xmax = mid - 5.0, mid + 5.0
        if zmax - zmin < 10.0:
            mid = 0.5 * (zmin + zmax)
            zmin, zmax = mid - 5.0, mid + 5.0
        pad = 0.1 * max(xmax - xmin, zmax - zmin)
        return (xmin - pad, xmax + pad, zmin - pad, zmax + pad)

    def _world_to_px(
        self,
        pts_xz: np.ndarray,
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Vectorised world (x, z) -> left-pane pixel (px, py)."""
        xmin, xmax, zmin, zmax = bounds
        margin = 40
        usable_w = self.pane_w - 2 * margin
        usable_h = self.pane_h - 2 * margin
        world_w = max(xmax - xmin, 1e-6)
        world_h = max(zmax - zmin, 1e-6)
        scale = min(usable_w / world_w, usable_h / world_h)
        cx = self.pane_w / 2.0
        cy = self.pane_h / 2.0
        xmid = 0.5 * (xmin + xmax)
        zmid = 0.5 * (zmin + zmax)
        px = cx + (pts_xz[:, 0] - xmid) * scale
        py = cy - (pts_xz[:, 1] - zmid) * scale  # flip z so +Z points up on screen
        return np.stack([px, py], axis=1).astype(np.int32)

    # ---------- panes ----------

    def _render_left(
        self,
        trajectory_xz: np.ndarray,
        map_xz: np.ndarray,
        gt_xz: Optional[np.ndarray],
        loop_segments_xz: Sequence[Tuple[np.ndarray, np.ndarray]],
        current_xz: Optional[np.ndarray],
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray:
        pane = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
        cv2.rectangle(
            pane, (1, 1), (self.pane_w - 2, self.pane_h - 2), (60, 60, 60), 1
        )

        # Map points as a single vectorised pixel poke (fast)
        if len(map_xz) > 0:
            mp_px = self._world_to_px(map_xz, bounds)
            mask = (
                (mp_px[:, 0] >= 0)
                & (mp_px[:, 0] < self.pane_w)
                & (mp_px[:, 1] >= 0)
                & (mp_px[:, 1] < self.pane_h)
            )
            mp_px = mp_px[mask]
            pane[mp_px[:, 1], mp_px[:, 0]] = (110, 110, 110)

        # Ground truth trajectory (green)
        if gt_xz is not None and len(gt_xz) > 1:
            gt_px = self._world_to_px(gt_xz, bounds)
            cv2.polylines(
                pane,
                [gt_px.reshape(-1, 1, 2)],
                False,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )

        # Loop closure edges (magenta)
        for a_xz, b_xz in loop_segments_xz:
            seg = np.array([[a_xz[0], a_xz[1]], [b_xz[0], b_xz[1]]], dtype=np.float64)
            seg_px = self._world_to_px(seg, bounds)
            cv2.line(
                pane,
                tuple(seg_px[0]),
                tuple(seg_px[1]),
                (200, 0, 200),
                1,
                cv2.LINE_AA,
            )

        # Estimated (live) trajectory (orange)
        if len(trajectory_xz) > 1:
            tr_px = self._world_to_px(trajectory_xz, bounds)
            cv2.polylines(
                pane,
                [tr_px.reshape(-1, 1, 2)],
                False,
                (0, 128, 255),
                2,
                cv2.LINE_AA,
            )

        # Current camera position marker (yellow dot with black outline)
        if current_xz is not None:
            cur_px = self._world_to_px(current_xz.reshape(1, 2), bounds)[0]
            cv2.circle(pane, tuple(cur_px), 7, (0, 255, 255), -1)
            cv2.circle(pane, tuple(cur_px), 9, (0, 0, 0), 2)

        # Header
        cv2.putText(
            pane,
            "Map + trajectory (top-down, X vs Z)",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        # Legend line
        legend_y = self.pane_h - 14
        cv2.putText(
            pane,
            "GT (green)  VO/SLAM (orange)  loops (magenta)",
            (12, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return pane

    def _render_right(
        self,
        img_gray: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        overlay_lines: Sequence[str],
    ) -> np.ndarray:
        if img_gray.ndim == 2:
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            img = img_gray.copy()

        if keypoints:
            for kp in keypoints:
                x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                cv2.circle(img, (x, y), 2, (0, 255, 255), -1)

        h, w = img.shape[:2]
        # Scale to fit pane while preserving aspect ratio
        scale = self.pane_w / w
        new_h = int(round(h * scale))
        pane = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
        if new_h > self.pane_h:
            scale = self.pane_h / h
            new_w = int(round(w * scale))
            img_scaled = cv2.resize(img, (new_w, self.pane_h))
            left_pad = (self.pane_w - new_w) // 2
            pane[:, left_pad : left_pad + new_w] = img_scaled
        else:
            img_scaled = cv2.resize(img, (self.pane_w, new_h))
            top_pad = (self.pane_h - new_h) // 2
            pane[top_pad : top_pad + new_h, :] = img_scaled

        cv2.rectangle(
            pane, (1, 1), (self.pane_w - 2, self.pane_h - 2), (60, 60, 60), 1
        )

        y = 28
        for line in overlay_lines:
            cv2.putText(
                pane,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 24

        cv2.putText(
            pane,
            "Left camera + keypoints (yellow)",
            (12, self.pane_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return pane

    # ---------- public API ----------

    def write(
        self,
        camera_img: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        trajectory_xz: np.ndarray,
        map_xz: np.ndarray,
        gt_xz: Optional[np.ndarray] = None,
        loop_segments_xz: Sequence[Tuple[np.ndarray, np.ndarray]] = (),
        overlay_lines: Sequence[str] = (),
    ) -> None:
        bounds = self._get_bounds(trajectory_xz, gt_xz)
        current = trajectory_xz[-1] if len(trajectory_xz) > 0 else None
        left = self._render_left(
            trajectory_xz, map_xz, gt_xz, loop_segments_xz, current, bounds
        )
        right = self._render_right(camera_img, keypoints, overlay_lines)
        frame = cv2.hconcat([left, right])
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
