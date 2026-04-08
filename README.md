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
- Roughly **22 GB free disk** for the KITTI grayscale stereo archive plus
  another ~1 GB for the generated videos if you run all 11 sequences.

The KITTI odometry dataset itself is not in the repo. You need
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

Runtime dependencies are `numpy`, `scipy`, `opencv-python`, and `matplotlib`. 

---

## Run one sequence

```bash
python src/stereo_slam.py --sequence 00
```

---

## Run all KITTI training sequences (00–10)

```bash
scripts/run_all_sequences.sh
```

To rerun only a subset, edit the `SEQUENCES=(…)` list at the top of the
script, or just run `python src/stereo_slam.py --sequence XX` by hand.

