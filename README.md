# slam-project

Stereo visual SLAM on the KITTI odometry benchmark. A feature-based,
frame-to-frame front-end tracks the camera with ORB + stereo depth + PnP,
and a scipy-based SE(3) pose-graph back-end closes loops online to
redistribute accumulated drift.

Architecture follows ORB-SLAM2 at a high level (ORB features, stereo
triangulation, PnP tracking, keyframe insertion, loop closures,
pose-graph optimization) with a few deliberate simplifications: brute-force
descriptor matching instead of a BoW vocabulary, pose-graph-only backend
(no bundle adjustment), and a single-threaded pipeline.

Implementation notes, per-function math, and further-reading links live in
[`docs/stereo_slam.md`](docs/stereo_slam.md).

---

## Repository layout

```
src/
  stereo_slam.py    main pipeline + CLI entry point
  stereo.py         stereo calibration loader + sparse stereo depth
  slam_utils.py     KITTI I/O, rigid-transform primitives, SE(3) helpers
  slam_video.py     side-by-side visualization video writer
scripts/
  run_all_sequences.sh   parallel runner for sequences 00..10
docs/
  stereo_slam.md    implementation notes + math references
results/
  <seq>.png         trajectory vs GT comparison plot
  <seq>.mp4         camera + map/trajectory visualization video
  logs/<seq>.log    per-sequence stdout
data/                KITTI odometry dataset (you must download this)
requirements.txt     Python dependencies
```

---

## Requirements

- Python 3.12+ (tested on CPython 3.14)
- A POSIX shell (Linux/macOS). The runner script is Bash.
- Roughly **22 GB free disk** for the KITTI grayscale stereo archive plus
  another ~1 GB for the generated videos if you run all 11 sequences.

The KITTI odometry dataset itself is not in the repo (too large). You need
to download three archives from the
[KITTI odometry benchmark page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php):

| Archive | Size | What's in it |
|---|---|---|
| `data_odometry_calib.zip` | ~1 MB | Camera intrinsics / projection matrices |
| `data_odometry_poses.zip` | ~4 MB | Ground-truth poses for sequences 00–10 |
| `data_odometry_gray.zip`  | ~22 GB | Left + right grayscale images (sequences 00–21) |

Extract all three under `data/` so the layout is:

```
data/
  data_odometry_calib/dataset/sequences/<SEQ>/calib.txt
  data_odometry_poses/dataset/poses/<SEQ>.txt
  data_odometry_gray/dataset/sequences/<SEQ>/image_0/*.png    (left)
  data_odometry_gray/dataset/sequences/<SEQ>/image_1/*.png    (right)
```

The pipeline reads `image_0` (left) for tracking and `image_1` (right)
for stereo depth. `<SEQ>` is a two-digit sequence number (00, 01, …).

---

## Install

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The only runtime dependencies are `numpy`, `scipy`, `opencv-python`, and
`matplotlib`. They are all pinned to compatible-release versions in
`requirements.txt`.

---

## Run one sequence

```bash
python src/stereo_slam.py --sequence 00
```

Outputs go to `results/00.png` and `results/00.mp4`:

- `<seq>.png` — top-down trajectory plot with ground truth (green), raw
  stereo VO (dotted blue), loop-closed SLAM (red), loop-closure edges
  (magenta), and the sparse map as gray points.
- `<seq>.mp4` — side-by-side visualization video: left pane is the
  top-down map + trajectory + loop closures filling in as the pipeline
  progresses; right pane is the left camera image with ORB features
  highlighted.

Additional flags:

```bash
python src/stereo_slam.py --sequence 05 --results-dir some/other/dir
python src/stereo_slam.py --sequence 02 --no-video    # skip the video
python src/stereo_slam.py --help                       # all options
```

---

## Run all KITTI training sequences (00–10) in parallel

```bash
scripts/run_all_sequences.sh
```

Launches 11 processes, one per sequence, each pinned to a single BLAS
thread so they don't oversubscribe the CPU. Per-sequence stdout is
redirected to `results/logs/<seq>.log`. The script waits for every job
to finish and then prints a summary. Expect about 10–15 minutes total on
a ~12-core desktop.

To rerun only a subset, edit the `SEQUENCES=(…)` list at the top of the
script, or just run `python src/stereo_slam.py --sequence XX` by hand.

---

## Interpreting the output

The on-terminal summary per run looks like:

```
[seq 00] Raw stereo VO:       final err    75.98 m, RMSE    90.05 m
[seq 00] Stereo SLAM (+PGO):  final err     4.34 m, RMSE    23.03 m
```

- **Raw stereo VO** is the front-end trajectory with no back-end
  corrections — it drifts linearly with travel distance.
- **Stereo SLAM (+PGO)** is the same trajectory after loop closures have
  been detected and the pose graph has been re-optimized online.

Sequences with loop closures (00, 02, 05, 06, 07, 09) show a visible
drop between these two rows. Sequences without loops (01, 03, 04, 08, 10)
report identical numbers — there's nothing for the back-end to correct
without a revisit.

The generated `results/<seq>.png` plot overlays the raw trajectory, the
optimized trajectory, and the ground truth for a direct visual
comparison.
