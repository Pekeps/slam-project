# slam-project

Visual SLAM exploration on the KITTI odometry benchmark.

- `src/rambon.py` — monocular VO with sparse mapping and PnP relocalization. Uses ground-truth poses to recover the monocular translation scale.
- `src/stereo_vo.py` — stereo visual odometry pipeline. Frame-to-frame tracking via stereo triangulation + PnP. Fully metric, no ground-truth scale dependency.
- `src/stereo.py` — stereo calibration loader and sparse stereo depth estimator (the building block used by `stereo_vo.py`).
- `src/slam_utils.py` — shared helpers (Frame/SparseMap, KITTI loaders, geometry).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the **calib**, **poses**, and **grayscale (`data_odometry_gray`)** archives from the [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and extract them under `data/` so you end up with:

```
data/data_odometry_calib/dataset/sequences/00/calib.txt
data/data_odometry_poses/dataset/poses/00.txt
data/data_odometry_gray/dataset/sequences/00/image_0/*.png   (left, used by mono)
data/data_odometry_gray/dataset/sequences/00/image_1/*.png   (right, used by stereo)
```

Only sequence 00 is tracked; the other 21 sequences and `image_1/` are gitignored to keep the repo small. You can change the sequence by editing `SEQUENCE` at the top of `src/rambon.py`.

## Run

Monocular VO (uses GT scale):

```bash
python src/rambon.py     # → kitti_final_result.png
```

Stereo VO (metric, no GT scale):

```bash
python src/stereo_vo.py  # → kitti_stereo_result.png
```

Both pipelines save a trajectory + sparse map + ground-truth comparison plot. `stereo_vo.py` also reports the final-position error and translation RMSE against the ground truth for quick quality feedback.
