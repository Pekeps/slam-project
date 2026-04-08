#!/usr/bin/env python
"""Aggregate KITTI-protocol error metrics across all saved trajectories.

Reads every ``results/<SEQ>.txt`` trajectory file, pairs it with the
ground-truth from ``data/data_odometry_poses/dataset/poses/<SEQ>.txt``,
computes the segment-based KITTI drift metrics (translational %, rotational
deg/km), and prints both a per-sequence table and an overall average.

Usage
-----

    python scripts/kitti_summary.py
    python scripts/kitti_summary.py --results-dir results
    python scripts/kitti_summary.py --sequences 00 02 05 07 09

The overall row is the plain arithmetic mean of per-sequence metrics. This
matches how papers typically report a KITTI summary line.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np


# Make the ``src/`` package importable whether this script is launched from
# the project root or from the scripts/ directory itself.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from slam_utils import load_poses_kitti  # noqa: E402
from stereo_slam import kitti_segment_errors  # noqa: E402


def load_kitti_trajectory(path: str) -> List[np.ndarray]:
    """Parse a 12-float-per-line KITTI trajectory file into 4x4 matrices."""
    poses: List[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = np.fromstring(line, dtype=np.float64, sep=" ")
            if vals.size != 12:
                raise ValueError(
                    f"{path}: expected 12 floats per line, got {vals.size}"
                )
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = vals.reshape(3, 4)
            poses.append(T)
    return poses


def gt_path_for(sequence: str) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "data",
        "data_odometry_poses",
        "dataset",
        "poses",
        f"{sequence}.txt",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--results-dir",
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Directory containing <seq>.txt trajectory files.",
    )
    ap.add_argument(
        "--sequences",
        nargs="+",
        default=[f"{i:02d}" for i in range(11)],
        help="Sequence IDs to include (default: 00..10).",
    )
    args = ap.parse_args()

    print(
        "seq | frames |   t_err (%) |   r_err (deg/km) |   segs"
    )
    print("----+--------+-------------+------------------+--------")

    t_list: List[float] = []
    r_list: List[float] = []
    frame_total = 0
    seg_total = 0

    for seq in args.sequences:
        traj_path = os.path.join(args.results_dir, f"{seq}.txt")
        gt_path = gt_path_for(seq)

        if not os.path.isfile(traj_path):
            print(f" {seq} | (missing results/{seq}.txt)")
            continue
        if not os.path.isfile(gt_path):
            print(f" {seq} | (missing GT {gt_path})")
            continue

        est = load_kitti_trajectory(traj_path)
        gt = load_poses_kitti(gt_path)
        if gt is None:
            print(f" {seq} | (GT load failed)")
            continue

        n_frames = min(len(est), len(gt))
        metrics = kitti_segment_errors(est, gt)
        t_err, r_err = metrics["overall"]
        n_segs = metrics["n_segments"]

        if np.isnan(t_err):
            print(
                f" {seq} | {n_frames:6d} |       (no segments long enough)"
            )
            continue

        print(
            f" {seq} | {n_frames:6d} | {t_err:10.3f}  | {r_err*1000:15.3f}  | {n_segs:6d}"
        )
        t_list.append(t_err)
        r_list.append(r_err)
        frame_total += n_frames
        seg_total += n_segs

    print("----+--------+-------------+------------------+--------")
    if t_list:
        t_mean = float(np.mean(t_list))
        r_mean = float(np.mean(r_list))
        print(
            f" avg| {frame_total:6d} | {t_mean:10.3f}  | {r_mean*1000:15.3f}  | {seg_total:6d}"
        )
    else:
        print(" avg| (no sequences produced metrics)")


if __name__ == "__main__":
    main()
