#!/usr/bin/env python
"""Aggregate KITTI-protocol error metrics across all saved trajectories.

Reads every ``results/<SEQ>.txt`` trajectory file (plus its ``.meta.txt``
runtime sidecar, if present), pairs it with the ground-truth from
``data/data_odometry_poses/dataset/poses/<SEQ>.txt``, computes the
segment-based KITTI drift metrics (translational %, rotational deg/km),
and prints both a per-sequence table and an overall average.

Usage
-----

    python scripts/kitti_summary.py
    python scripts/kitti_summary.py --results-dir results
    python scripts/kitti_summary.py --sequences 00 02 05 07 09
    python scripts/kitti_summary.py --output results/benchmark.txt

With ``--output`` the summary is written to both the file and stdout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Dict, List, Optional

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


def load_meta(path: str) -> Dict[str, float]:
    """Parse a sidecar `.meta.txt` file. Returns an empty dict if missing."""
    if not os.path.isfile(path):
        return {}
    out: Dict[str, float] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts
            try:
                out[key] = float(val)
            except ValueError:
                continue
    return out


def gt_path_for(sequence: str) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "data",
        "data_odometry_poses",
        "dataset",
        "poses",
        f"{sequence}.txt",
    )


def build_report(sequences: List[str], results_dir: str) -> str:
    """Build the multi-line benchmark report as a single string."""
    lines: List[str] = []

    header = (
        "seq | frames |  t_err (%) |  r_err (deg/m)  | runtime (s) | ms/frame | segs "
    )
    sep = (
        "----+--------+------------+-----------------+-------------+----------+------"
    )
    lines.append(
        f"# Stereo SLAM KITTI benchmark — generated "
        f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("#")
    lines.append(
        "# translational error: percent drift over 100-800 m sub-trajectories"
    )
    lines.append(
        "# rotational error:    degrees per metre over the same sub-trajectories"
    )
    lines.append(
        "# runtime:              wall-clock seconds for slam.run() only (front-end + online PGO)"
    )
    lines.append(
        "# ms/frame:             average per-frame processing time in milliseconds"
    )
    lines.append("#")
    lines.append(header)
    lines.append(sep)

    t_list: List[float] = []
    r_list: List[float] = []
    frame_total = 0
    seg_total = 0
    wall_total = 0.0
    valid_count = 0

    for seq in sequences:
        traj_path = os.path.join(results_dir, f"{seq}.txt")
        meta_path = os.path.join(results_dir, f"{seq}.meta.txt")
        gt_path = gt_path_for(seq)

        if not os.path.isfile(traj_path):
            lines.append(f" {seq} | (missing {os.path.basename(traj_path)})")
            continue
        if not os.path.isfile(gt_path):
            lines.append(f" {seq} | (missing GT {gt_path})")
            continue

        est = load_kitti_trajectory(traj_path)
        gt = load_poses_kitti(gt_path)
        if gt is None:
            lines.append(f" {seq} | (GT load failed)")
            continue

        meta = load_meta(meta_path)
        wall = meta.get("wall_seconds", float("nan"))

        n_frames = min(len(est), len(gt))
        metrics = kitti_segment_errors(est, gt)
        t_err, r_err = metrics["overall"]
        n_segs = metrics["n_segments"]

        # Per-frame time derived from wall time and the actual frame count,
        # so the meta file only needs to store the one scalar that isn't
        # already implied by the trajectory file.
        spf_ms = (wall / n_frames * 1000.0) if (not np.isnan(wall) and n_frames > 0) else float("nan")

        def fmt(x: float, width: int, prec: int = 2) -> str:
            return "   n/a" if np.isnan(x) else f"{x:{width}.{prec}f}"

        if np.isnan(t_err):
            lines.append(
                f" {seq} | {n_frames:6d} |     n/a    |        n/a      "
                f"| {fmt(wall, 10):>10}  | {fmt(spf_ms, 7, 2):>7}  | {n_segs:5d}"
            )
            continue

        lines.append(
            f" {seq} | {n_frames:6d} | "
            f"{t_err:9.3f}  | {r_err:13.5f}   | {fmt(wall, 10, 2):>10}  | "
            f"{fmt(spf_ms, 7, 2):>7}  | {n_segs:5d}"
        )
        t_list.append(t_err)
        r_list.append(r_err)
        frame_total += n_frames
        seg_total += n_segs
        if not np.isnan(wall):
            wall_total += wall
        valid_count += 1

    lines.append(sep)
    if t_list:
        t_mean = float(np.mean(t_list))
        r_mean = float(np.mean(r_list))
        avg_spf_ms = (wall_total / frame_total * 1000.0) if frame_total else float("nan")
        # "avg" row: means of the quantities that average meaningfully
        # across sequences. Columns that are strictly totals get "—".
        lines.append(
            f" avg|    —   | "
            f"{t_mean:9.3f}  | {r_mean:13.5f}   |      —      | "
            f"{avg_spf_ms:7.2f}  |   —  "
        )
        # "tot" row: sums of frames, wall time, and segment counts.
        # t_err/r_err/ms-per-frame don't sum meaningfully so they get "—".
        lines.append(
            f" tot| {frame_total:6d} |     —      |       —         "
            f"| {wall_total:10.2f}  |    —     | {seg_total:5d}"
        )
    else:
        lines.append(" avg| (no sequences produced metrics)")

    lines.append("")
    lines.append(f"# {valid_count} sequence(s) evaluated")
    return "\n".join(lines) + "\n"


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
    ap.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional file to write the report to (in addition to stdout).",
    )
    args = ap.parse_args()

    report = build_report(args.sequences, args.results_dir)
    sys.stdout.write(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nbenchmark saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
