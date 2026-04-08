# Stereo SLAM — Implementation Notes

This document walks through the math and code of the stereo SLAM pipeline
in [`src/stereo_slam.py`](../src/stereo_slam.py) and its supporting modules
[`src/slam_utils.py`](../src/slam_utils.py) and
[`src/stereo.py`](../src/stereo.py). For each function we give the
signature, what it does, the relevant math in LaTeX, and links to
resources that explain the operation in more depth.

---

## Reference paper

The overall architecture is a simplified version of:

> **Mur-Artal, R. and Tardós, J. D.** (2017).
> *"ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo, and
> RGB-D Cameras"*, IEEE Transactions on Robotics 33(5), 1255–1262.
> [arXiv:1610.06475](https://arxiv.org/abs/1610.06475) ·
> [Official PDF](https://webdiis.unizar.es/~raulmur/orbslam/)

Like ORB-SLAM2 we use:

- **ORB** features for both frame-to-frame tracking and loop-closure
  candidate matching.
- **Stereo triangulation** to recover metric depth so no
  ground-truth scale is needed.
- **PnP** between a reference's 3D points and the next frame's 2D
  observations to obtain the camera pose.
- **Keyframes** as the units of the back-end, chosen by a motion-based
  heuristic.
- **Loop closures** detected by descriptor matching + geometric
  verification (PnP RANSAC), then enforced as edges in a pose graph.
- **Pose-graph optimization** to redistribute accumulated drift after a
  loop closes.

Simplifications vs ORB-SLAM2:

- No bag-of-words (DBoW2) place recognition; we brute-force match new
  keyframes against older ones.
- Pose-graph-only back-end; no local or full bundle adjustment.
- No separate tracking / mapping / loop-closing threads; everything runs
  in a single loop.

Other core references used below:

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd
  ed., Cambridge University Press, 2004. [Book site](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- Barfoot, T., *State Estimation for Robotics*, Cambridge University
  Press, 2017. [Free PDF](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf)
- Solà, J., Deray, J., Atchuthan, D., *"A micro Lie theory for state
  estimation in robotics"*, arXiv 2018.
  [arXiv:1812.01537](https://arxiv.org/abs/1812.01537)
- Grisetti, G., Kümmerle, R., Stachniss, C., Burgard, W., *"A Tutorial on
  Graph-Based SLAM"*, IEEE Intelligent Transportation Systems Magazine,
  2010. [PDF](https://www.cs.cmu.edu/~kaess/pub/Grisetti10titsmag.pdf)
- Kümmerle et al., *"g2o: a general framework for graph optimization"*,
  ICRA 2011. [PDF](https://www2.informatik.uni-freiburg.de/~stachnis/pdf/kuemmerle11icra.pdf)

---

## Mathematical preliminaries

### Rigid body transformations, $SE(3)$

A camera pose is a rigid-body transformation $T \in SE(3)$ written as a
$4\times 4$ homogeneous matrix

$$
T = \begin{bmatrix} R & t \\ 0^\top & 1 \end{bmatrix}, \qquad R \in SO(3),\; t \in \mathbb{R}^3
$$

where $R$ is a rotation and $t$ is a translation. A point $\mathbf{X}$
expressed in camera coordinates is mapped into world coordinates by
$\mathbf{X}_w = R\,\mathbf{X}_c + t$, or equivalently, with the homogeneous
embedding $\tilde{\mathbf{X}} = [x, y, z, 1]^\top$, by
$\tilde{\mathbf{X}}_w = T\,\tilde{\mathbf{X}}_c$. We write $T_w^c$ for
"pose of camera in world", i.e. the matrix that takes camera-frame
points into the world frame.

**Inverse:** If $T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$ then

$$
T^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix}
$$

*Resource:* Hartley & Zisserman, Ch. 2. Barfoot, Ch. 7.

### Lie algebra $\mathfrak{se}(3)$ and the exponential map

The tangent space of $SE(3)$ at the identity is
$\mathfrak{se}(3) \cong \mathbb{R}^6$. We parameterize a small rigid-body
increment by $\xi = (\rho, \phi) \in \mathbb{R}^6$, where $\rho$ is a
translation-like part and $\phi$ is an axis-angle rotation vector (the
magnitude $\theta = \lVert\phi\rVert$ is the rotation angle, and
$\phi/\theta$ is the axis).

The **hat operator** lifts $\xi$ to the Lie algebra $\mathfrak{se}(3)$
as a $4\times 4$ matrix

$$
\xi^\wedge = \begin{bmatrix} \phi^\wedge & \rho \\ 0^\top & 0 \end{bmatrix}, \qquad \phi^\wedge = \begin{bmatrix} 0 & -\phi_3 & \phi_2 \\ \phi_3 & 0 & -\phi_1 \\ -\phi_2 & \phi_1 & 0 \end{bmatrix}
$$

and the **exponential map** $\exp: \mathfrak{se}(3) \to SE(3)$ sends
$\xi$ to a rigid-body transformation via

$$
\exp(\xi^\wedge) = \begin{bmatrix} R(\phi) & V(\phi)\,\rho \\ 0^\top & 1 \end{bmatrix}
$$

where (using Rodrigues' formula with $\theta = \lVert\phi\rVert$)

$$
R(\phi) = I + \frac{\sin\theta}{\theta}\,\phi^\wedge + \frac{1-\cos\theta}{\theta^2}\,(\phi^\wedge)^2
$$

$$
V(\phi) = I + \frac{1-\cos\theta}{\theta^2}\,\phi^\wedge + \frac{\theta - \sin\theta}{\theta^3}\,(\phi^\wedge)^2
$$

For small $\theta$, both formulas reduce to first-order expansions:
$R \approx I + \phi^\wedge$, $V \approx I + \tfrac{1}{2}\phi^\wedge$.

*Resources:*
- Solà et al., [A micro Lie theory](https://arxiv.org/abs/1812.01537),
  especially §3 and the SE(3) section of §7.
- Barfoot, [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf), Ch. 7.
- Ethan Eade, [Lie Groups for 2D and 3D Transformations](https://ethaneade.com/lie.pdf) — concise PDF notes.

### Camera projection model (pinhole)

A 3D point $\mathbf{X}_c = (X, Y, Z)^\top$ in camera coordinates projects
onto the image plane as

$$
\begin{bmatrix} u \\ v \end{bmatrix} = \pi(K, \mathbf{X}_c) = \begin{bmatrix} f_x X/Z + c_x \\ f_y Y/Z + c_y \end{bmatrix}
$$

with the intrinsic matrix

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

For a world point $\mathbf{X}_w$ and camera pose $T_w^c$, the projection
is $\pi\!\left(K,\; (T_w^c)^{-1}\,\tilde{\mathbf{X}}_w\right)$.

The **projection matrix** in homogeneous form is

$$
P = K\,\left[R \mid t\right]
$$

which takes a 3D point $\tilde{\mathbf{X}}_w = [X, Y, Z, 1]^\top$ to a
homogeneous image point $\tilde{\mathbf{x}} = P\,\tilde{\mathbf{X}}_w$.

*Resources:*
- Hartley & Zisserman, Ch. 6.
- OpenCV docs, [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

---

## Module: `stereo.py`

### `StereoCalib` (dataclass)

```python
@dataclass
class StereoCalib:
    K: np.ndarray      # 3x3 shared intrinsics (rectified)
    P0: np.ndarray     # 3x4 left projection
    P1: np.ndarray     # 3x4 right projection
    baseline: float    # meters
    fx: float          # K[0, 0]
```

KITTI stores two projection matrices $P_0$ and $P_1$ in `calib.txt`.
Because the stereo pair is rectified to share intrinsics, $K = P_0[:,\!:\!3]$
for both cameras. The baseline $b$ is encoded in $P_1$:

$$
P_1 = K\,\left[I \mid t\right] = \begin{bmatrix} f_x & 0 & c_x & -f_x b \\ 0 & f_y & c_y & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
$$

so the positive baseline in meters is $b = -P_1[0, 3] / f_x$.

*Resource:* KITTI [odometry benchmark page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### `read_kitti_stereo_calib(calib_path) -> StereoCalib`

Parses a KITTI `calib.txt`, pulls the `P0:` and `P1:` lines, and builds a
`StereoCalib`. Sets $K = P_0[:\!,\!:\!3]$, $f_x = K[0, 0]$, and
$b = -P_1[0, 3] / f_x$.

### `StereoFrame` (dataclass)

```python
@dataclass
class StereoFrame:
    kps: List[cv2.KeyPoint]  # features in LEFT image
    desc: np.ndarray         # Nx32 ORB descriptors (uint8)
    pts3d_cam: np.ndarray    # Nx3 metric points in LEFT camera frame
```

All three arrays are index-aligned: `kps[i]`, `desc[i]`, and
`pts3d_cam[i]` describe the same physical point.

### `estimate_stereo_depth(left_img, right_img, calib, detector, matcher, ratio_test, max_epipolar_err_px, min_depth, max_depth) -> StereoFrame`

Sparse stereo feature triangulation. The pipeline is:

1. Detect ORB features in both images independently.
2. Match left-to-right descriptors with **Lowe's ratio test**:
   keep $m$ iff $m.\text{distance} < \alpha \cdot n.\text{distance}$, where
   $n$ is the second-nearest neighbor. With $\alpha = 0.7$ this rejects
   ambiguous matches.
3. **Rectified-pair sanity filter.** Since the stereo pair is rectified
   the epipolar lines are horizontal, so for a correct match we require
   $$|v_L - v_R| < \varepsilon \quad\text{and}\quad u_L - u_R > 0$$
   where $\varepsilon$ is ~1 px and the second condition requires
   positive disparity (point in front of the camera).
4. **Triangulation** via [`cv2.triangulatePoints(P0, P1, ptsL, ptsR)`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c).
   Given two projection matrices $P_0, P_1$ and a corresponding pair of
   image points $\mathbf{x}_L, \mathbf{x}_R$, OpenCV solves the
   Direct Linear Transform (DLT):
   $$
   A\,\tilde{\mathbf{X}} = 0, \qquad A = \begin{bmatrix} u_L P_0^{(3)} - P_0^{(1)} \\ v_L P_0^{(3)} - P_0^{(2)} \\ u_R P_1^{(3)} - P_1^{(1)} \\ v_R P_1^{(3)} - P_1^{(2)} \end{bmatrix}
   $$
   where $P_k^{(i)}$ is the $i$-th row of $P_k$. The 3D point
   $\tilde{\mathbf{X}}$ is the right null-space of $A$, found via SVD.
5. **Depth gating.** Reject points whose depth $Z$ is outside
   $[\mathrm{min\_depth},\, \mathrm{max\_depth}]$.

**Analytic form of stereo triangulation** for a rectified pair:

$$
Z = \frac{f_x \cdot b}{u_L - u_R}, \quad X = \frac{(u_L - c_x)\,Z}{f_x}, \quad Y = \frac{(v_L - c_y)\,Z}{f_y}
$$

The OpenCV DLT call is a (mildly) more numerically robust generalization
of this closed form.

*Resources:*
- Hartley & Zisserman, Ch. 12 (triangulation), §12.2 for the DLT method.
- [OpenCV tutorial: Epipolar geometry](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html).
- Lowe, D., [Distinctive Image Features from Scale-Invariant Keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), IJCV 2004 — origin of the ratio test.
- Rublee et al., [ORB: an efficient alternative to SIFT or SURF](http://www.gwylab.com/download/ORB_2012.pdf), ICCV 2011.

---

## Module: `slam_utils.py`

### `load_image_paths(img_dir) -> List[str]`

Globs `*.png` / `*.jpg` under `img_dir` and returns them sorted.

### `load_poses_kitti(poses_path) -> Optional[List[np.ndarray]]`

Reads KITTI-format ground truth poses. Each line of the file contains
the 12 entries of a $3\times 4$ row-major pose matrix. Returns a list of
$4\times 4$ homogeneous matrices (or `None` if the file doesn't exist).

### `form_T(R, t) -> np.ndarray`

Assemble a $4\times 4$ homogeneous transform from a $3\times 3$ rotation
$R$ and a 3-vector $t$:

$$
T = \begin{bmatrix} R & t \\ 0^\top & 1 \end{bmatrix}
$$

### `invert_T(T) -> np.ndarray`

Closed-form rigid inverse — avoids a full matrix inverse:

$$
T^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0^\top & 1 \end{bmatrix}
$$

### `skew(v) -> np.ndarray`

Computes the hat operator $v^\wedge$ for $v \in \mathbb{R}^3$:

$$
v^\wedge = \begin{bmatrix} 0 & -v_3 & v_2 \\ v_3 & 0 & -v_1 \\ -v_2 & v_1 & 0 \end{bmatrix}
$$

This is the matrix that implements the cross product as a matrix
multiplication: $v^\wedge\,w = v \times w$.

### `se3_exp(xi) -> np.ndarray`

Implements the $\mathfrak{se}(3) \to SE(3)$ exponential map described in
the preliminaries. Given $\xi = (\rho, \phi) \in \mathbb{R}^6$ it returns
the $4\times 4$ matrix

$$
\exp(\xi^\wedge) = \begin{bmatrix} R(\phi) & V(\phi)\,\rho \\ 0^\top & 1 \end{bmatrix}
$$

using Rodrigues' formulas for $R(\phi)$ and $V(\phi)$, with a
first-order small-angle fallback when $\theta = \lVert\phi\rVert$ is
numerically small.

This function is the workhorse of pose-graph optimization: it lets us
parameterize a local update to a pose by a minimal 6-vector.

*Resources:*
- [Solà et al., §5.2–5.3](https://arxiv.org/abs/1812.01537).
- Barfoot, eqs. (7.66)–(7.80).
- [Strasdat's TUM slides on Lie groups](https://drive.google.com/file/d/0B9rLLz1XQKmaZTlQdE51eGxvN2s/view).

### `pose_error_6d(T_err) -> np.ndarray`

Converts an SE(3) error matrix (identity = zero error) into a 6-vector
residual:

$$
r(T_\text{err}) = \begin{bmatrix} t_\text{err} \\ \log_{SO(3)}(R_\text{err}) \end{bmatrix} \in \mathbb{R}^6
$$

where $\log_{SO(3)}$ extracts the angle-axis representation via
`cv2.Rodrigues`. This is *not* the exact $\log_{SE(3)}$ (which couples
translation and rotation through $V^{-1}$), but it is zero exactly when
$T_\text{err} = I$ and is well-behaved for small errors — which is all
least-squares residuals need.

*Resource:* [Solà et al., §5.3](https://arxiv.org/abs/1812.01537).

---

## Module: `stereo_slam.py`

### `SlamConfig` (dataclass)

Holds all the tunable knobs: feature count, ratio test threshold, PnP
parameters, keyframe cadence, loop closure gates, pose graph tolerances,
and video output options.

### `Keyframe` (dataclass)

```python
@dataclass
class Keyframe:
    id: int
    frame_idx: int
    T_w_c: np.ndarray       # world pose, mutable (updated by PGO)
    kps: List[cv2.KeyPoint]
    desc: np.ndarray        # Nx32 descriptors
    pts3d_cam: np.ndarray   # Nx3, stored in THIS keyframe's camera frame
```

Note that `pts3d_cam` is in the keyframe's **camera frame**, not the
world frame. This is a deliberate choice: when PGO moves `T_w_c`, the
world positions of the points follow automatically without a separate
re-projection step.

---

### `PoseGraph`

A minimal SE(3) pose graph backend built on
[`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

#### `add_vertex(T_w_c) -> int`

Appends a $4\times 4$ pose as a new vertex, returns its integer id.

#### `add_edge(i, j, T_ij_meas) -> None`

Adds a constraint between vertices $i$ and $j$ with the measured
relative transform $T_{ij}^\text{meas}$. By convention
$T_{ij}^\text{meas}$ is the **pose of $j$ in $i$'s frame**, i.e. it
satisfies $T_{ij}^\text{meas} = (T_w^{c_i})^{-1}\,T_w^{c_j}$ when the
measurement is perfect.

#### `fix(idx) -> None`

Marks a vertex as fixed — it is held constant during optimization.
Fixing at least one vertex removes the gauge freedom (otherwise the
whole graph could slide arbitrarily in $SE(3)$ without changing the
cost).

#### `optimize(max_nfev, ftol, xtol) -> dict`

This is the heart of the back-end. It minimizes

$$
\mathcal{F}(T_1, \ldots, T_N) = \frac{1}{2}\,\sum_{(i,j)\in\mathcal{E}} \left\lVert \log_{SE(3)}\!\left(T_{ij}^\text{meas\,-1}\,T_i^{-1}\,T_j\right)\right\rVert^2
$$

over all non-fixed vertices. Here $\mathcal{E}$ is the edge set (the
odometry chain plus any loop closures) and the inner term is the
6-vector error between the predicted relative pose
$T_i^{-1}\,T_j$ and the measurement $T_{ij}^\text{meas}$, via
`pose_error_6d`.

##### Local $SE(3)$ parameterization

The unknowns are a 6-vector increment $\xi_k \in \mathbb{R}^6$ per
free vertex, and each vertex pose is parameterized around a fixed
linearization point $T_k^{(0)}$ (the *current* keyframe pose at the
moment `optimize` is called) as

$$
T_k(\xi_k) = T_k^{(0)}\,\exp(\xi_k^\wedge)
$$

This is a **right perturbation**: $\xi$ lives in the local
(body-fixed) frame of $T_k^{(0)}$. The optimizer searches over $\xi$,
and at the initial guess $\xi = 0$, $T_k = T_k^{(0)}$. After convergence
we absorb the increments:

$$
T_k^{(0)} \leftarrow T_k^{(0)}\,\exp(\xi_k^{*\,\wedge})
$$

and the vertices are updated to the optimized poses.

##### Why odometry edges have zero initial residual

We take care to insert each new keyframe in a way that makes the
odometry edge residual exactly zero at the linearization point. The edge
measurement is computed directly from `raw_poses`:

$$
T_{ij}^\text{meas} = (T_w^{c_i,\text{raw}})^{-1}\,T_w^{c_j,\text{raw}}
$$

and the new keyframe's pose is initialized by propagating forward from
the previous (possibly PGO-updated) keyframe along the same raw-VO
relative motion. A direct calculation shows that with these choices,
$T_i^{-1}\,T_j = T_{ij}^\text{meas}$ at $\xi = 0$, so the error vanishes
identically on all odometry edges. Only **loop closure edges** have
nonzero initial residuals — which is exactly where the information
comes from.

##### Sparsity pattern for the Jacobian

Scipy's TRF solver can exploit sparsity in the Jacobian to compute
finite differences efficiently. For a pose graph, the residual of edge
$(i, j)$ only depends on vertices $i$ and $j$, so the Jacobian has block
structure: each edge contributes a $6\times 6$ block on $\xi_i$ and a
$6\times 6$ block on $\xi_j$. We build this sparsity mask as a
`scipy.sparse.lil_matrix` and pass it via `jac_sparsity=...`.

*Resources:*
- Grisetti et al., [A Tutorial on Graph-Based SLAM](https://www.cs.cmu.edu/~kaess/pub/Grisetti10titsmag.pdf) — a gentle, thorough derivation of the entire pose-graph machinery.
- Kümmerle et al., [g2o](https://www2.informatik.uni-freiburg.de/~stachnis/pdf/kuemmerle11icra.pdf) — the reference pose-graph backend.
- Triggs et al., [Bundle Adjustment — A Modern Synthesis](https://lear.inrialpes.fr/pubs/2000/TMHF00/Triggs-va99.pdf) — the canonical optimization reference.
- [SciPy `least_squares` docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) — including notes on `method='trf'` and `jac_sparsity`.

---

### `StereoSlam`

The main driver class. Instantiating it loads the calibration, stereo
image paths, optional GT, creates an ORB detector + BFMatcher, and
initializes the pose graph and map cache.

#### `_track_pose(prev_X_w, prev_desc, curr_kps, curr_desc) -> T_w_c | None`

Estimates the world pose of the current frame by matching the current
left-image descriptors against the *previous frame's* 3D world points
and running PnP RANSAC.

The pipeline is:

1. Match current descriptors (query) against `prev_desc` (train) with
   ratio-test BFMatcher.
2. Build 2D–3D correspondences:
   $\{(\mathbf{u}_k, \mathbf{X}_w^k)\}$.
3. Call [`cv2.solvePnPRansac`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e),
   which minimizes reprojection error subject to a RANSAC inlier set:
   $$
   (R^*, t^*) = \arg\min_{R,t} \sum_{k \in \mathcal{I}} \left\lVert \pi(K, R\,\mathbf{X}_w^k + t) - \mathbf{u}_k \right\rVert^2
   $$
   where $\pi$ is the pinhole projection and $\mathcal{I}$ is the RANSAC
   inlier set.
4. Convert the returned world-to-camera transform to $T_w^c$ via
   `invert_T(form_T(R, t))`.

Returns `None` if there are too few matches or PnP fails — in which
case the caller holds the previous pose for that frame.

*Resources:*
- [OpenCV tutorial: solvePnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
- Lepetit et al., [EPnP: an accurate $O(n)$ solution to the PnP problem](https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Documents/team_lepetit/publications/lepetit_ijcv09.pdf), IJCV 2009 — one of the P3P/PnP solvers that OpenCV ships.
- Hartley & Zisserman, §7.1.

#### `_insert_keyframe(frame_idx, T_w_c_live, sf) -> Keyframe`

Creates a new `Keyframe` containing the current stereo features and
points, appends it, adds a corresponding pose-graph vertex, fixes the
first one, and adds the odometry edge from the previous keyframe.

**Critical detail:** the odometry edge is computed directly from
`raw_poses`, not from the current `T_w_c` fields of the keyframes:

```python
T_rel = invert_T(raw_poses[prev_kf.frame_idx]) @ raw_poses[kf.frame_idx]
```

This keeps the edge invariant to any past pose-graph corrections. The
keyframe's own `T_w_c` is initialized from the *live* (possibly
corrected) pose of the current frame, so at $\xi = 0$ we have

$$
T_{\text{prev}}^{-1}\,T_{\text{new}} \;=\; T_{ij}^\text{meas}
$$

and the odometry-edge initial residual is zero.

#### `_verify_loop(curr_kf, old_kf) -> (T_old_curr, inliers) | None`

Tests whether two keyframes correspond to the same physical place.

1. **Raw-trajectory proximity gate.** If the pure-VO world positions of
   the two keyframes differ by more than `loop_max_raw_dist` (default 100 m),
   reject. For a *true* loop closure the two raw positions differ only by
   accumulated drift (tens of meters on KITTI); for a spurious match
   between visually similar but physically distant places they differ by
   the actual geometric distance (hundreds of meters).

2. **Descriptor matching.** BFMatcher knnMatch of `curr_kf.desc` against
   `old_kf.desc` with ratio test.

3. **Match-count gate.** Require `loop_min_matches` surviving matches.

4. **PnP verification.** Run `solvePnPRansac` with old's
   `pts3d_cam` (3D points in old's camera frame) and curr's pixel
   observations. This estimates the relative pose
   $T_{\text{curr}}^{\text{old}}$ that would explain the observations if
   the two keyframes are truly seeing the same scene. Require
   `loop_min_inliers` RANSAC inliers.

5. **Translation magnitude gate.** If the recovered relative translation
   exceeds `loop_max_rel_trans` (default 15 m), reject: for a physically
   genuine loop the two cameras should be within metres of each other.

On success, returns the inverse $T_{\text{old}}^{\text{curr}} = (T_{\text{curr}}^{\text{old}})^{-1}$ — the **pose of the current
keyframe in the old keyframe's frame**, which is the correct convention
for a pose graph edge from `old_id` to `curr_id`.

#### `_detect_loop_closures(curr_kf) -> List[(id, T_old_curr, inliers)]`

Iterates over all keyframes older than `loop_skip_recent` behind the
current one and calls `_verify_loop` on each. Returns all verified loop
closures (a single new keyframe can match multiple older keyframes — this
is expected and the extra constraints are useful).

#### `_compute_live_pose(frame_idx, T_w_c_raw) -> np.ndarray`

Rebases a raw-VO pose onto the nearest preceding keyframe's current
(post-PGO) pose. For a frame $k$ with preceding keyframe at frame $i_k$:

$$
T_{w,k}^\text{live} = T_{w,i_k}^\text{live}\,\cdot\,(T_{w,i_k}^\text{raw})^{-1}\,T_{w,k}^\text{raw}
$$

i.e. "take the raw relative motion from keyframe $i_k$ to frame $k$ and
paste it onto the corrected world position of keyframe $i_k$". This is
what turns PGO corrections into a globally consistent trajectory
including the non-keyframe frames in between.

#### `_rebuild_live_poses() -> None`

After a PGO event, rebuilds the entire `live_poses` list by calling
`_compute_live_pose` on every raw pose seen so far. The live trajectory
snaps to reflect the correction.

#### `_optimize_pose_graph() -> None`

Calls `PoseGraph.optimize`, copies the optimized poses back into the
keyframe objects, and marks the map cache dirty so the next render
rebuilds it.

#### `_get_world_map() -> (X_w, desc)`

Lazily builds the world-frame sparse map by iterating over all keyframes
and transforming their `pts3d_cam` through their *current* `T_w_c`:

$$
\mathbf{X}_w^{(k,i)} = R_w^{c_k}\,\mathbf{X}_{c_k}^{i} + t_w^{c_k}
$$

Because the map is derived from `kf.T_w_c` every time, any PGO update
to the keyframes automatically moves the map with them.

#### `_write_video_frame(i, left_img, sf, total_loop_closures, gt_xz_full) -> None`

Assembles the information needed by `SlamVideoWriter`: the
live-trajectory $(x, z)$ array, the current world map as a $(M, 2)$
array, ground truth up to frame $i$, loop closure line segments, and an
overlay of textual status fields.

#### `run() -> None`

The main loop. For each frame $i$:

1. Load left + right image, run `estimate_stereo_depth`.
2. If $i = 0$, set $T_w^c = I$; otherwise call `_track_pose` against the
   previous frame's 3D points (in the **raw VO frame** — the front-end
   is deliberately decoupled from the back-end).
3. Append to `raw_poses`.
4. Update the tracking carry-over (`prev_X_w`, `prev_desc`) using the
   raw pose, so `prev_X_w` stays in a single consistent coordinate
   frame.
5. Compute the **live** pose for the current frame by rebasing on the
   latest keyframe (`_compute_live_pose`).
6. If we should insert a keyframe (cadence triggered), do so with the
   *live* pose (so the new KF enters the graph in the corrected world
   frame), then search for loop closures. If any are found, add the
   edges and run online PGO.
7. If PGO ran, rebuild all live poses. Otherwise append the new live
   pose.
8. Write a video frame if recording is enabled.

---

### Standalone functions

#### `trajectory_error(poses, gt_poses) -> (final_err, rmse)`

For two same-length lists of $4\times 4$ poses, extract the translation
parts and compute the final-pose error
$\lVert t_{\text{est},N} - t_{\text{gt},N}\rVert$ and the
trajectory translation RMSE
$\sqrt{\tfrac{1}{N}\sum_k \lVert t_{\text{est},k} - t_{\text{gt},k}\rVert^2}$.

Note this is the *aligned* RMSE assuming the trajectories start in the
same coordinate frame (as they do here: both start at identity). For a
*rigid-aligned* absolute trajectory error as in KITTI benchmarking one
would first Umeyama-align the trajectories.

*Resources:*
- Geiger et al., [KITTI odometry evaluation metrics](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- Umeyama, S., [Least-squares estimation of transformation parameters between two point patterns](https://web.stanford.edu/class/cs273/refs/umeyama.pdf), IEEE TPAMI 1991.

#### `plot_slam_results(slam, poses_raw, poses_opt, out_path) -> None`

Produces the final static matplotlib comparison plot: 3D map points,
ground truth (if available), raw stereo-VO trajectory (no loops),
optimized stereo-SLAM trajectory, and loop closure line segments drawn
between the corrected keyframe positions.

---

## Further reading

### Textbooks and chapters

- **Hartley, R. and Zisserman, A.** *Multiple View Geometry in Computer
  Vision*, 2nd ed., CUP 2004.
  [Book site](https://www.robots.ox.ac.uk/~vgg/hzbook/). The standard
  reference for triangulation, fundamental/essential matrix,
  projections, and calibration.
- **Barfoot, T.** *State Estimation for Robotics*, CUP 2017.
  [Free PDF](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf).
  Chapter 7 (Matrix Lie Groups) is the definitive textbook treatment of
  $SE(3)$.
- **Scaramuzza, D. and Fraundorfer, F.**, *"Visual Odometry: Part I &
  II"*, IEEE Robotics & Automation Magazine, 2011.
  [Part I PDF](https://rpg.ifi.uzh.ch/docs/VO_Part_I_Scaramuzza.pdf) ·
  [Part II PDF](https://rpg.ifi.uzh.ch/docs/VO_Part_II_Scaramuzza.pdf).
  An excellent tutorial-level introduction to VO.

### Tutorial papers

- **Solà, J., Deray, J., Atchuthan, D.**, [*"A micro Lie theory for
  state estimation in robotics"*](https://arxiv.org/abs/1812.01537),
  arXiv 2018. The clearest concise treatment of $\mathfrak{se}(3)$
  parameterizations for SLAM practitioners.
- **Grisetti, G. et al.**, [*"A Tutorial on Graph-Based SLAM"*](https://www.cs.cmu.edu/~kaess/pub/Grisetti10titsmag.pdf),
  IEEE ITSM 2010. Walks through the pose-graph formulation from scratch.
- **Strasdat, H.**, *"Local Accuracy and Global Consistency for Efficient
  Visual SLAM"*, PhD thesis, Imperial College London, 2012. Contains a
  very readable derivation of SE(3) Jacobians for pose-graph problems.
- **Ethan Eade**, [*"Lie Groups for 2D and 3D Transformations"*](https://ethaneade.com/lie.pdf) — a 10-page
  cheat-sheet on $SO(3)$/$SE(3)$ that every SLAM implementer should
  read.

### Foundational papers

- **Mur-Artal, R. and Tardós, J. D.**, [*"ORB-SLAM2"*](https://arxiv.org/abs/1610.06475), T-RO 2017. The primary
  inspiration for this implementation.
- **Mur-Artal, R., Montiel, J.M.M., Tardós, J. D.**, [*"ORB-SLAM: a
  versatile and accurate monocular SLAM system"*](https://arxiv.org/abs/1502.00956), T-RO 2015.
- **Rublee, E., Rabaud, V., Konolige, K., Bradski, G.**, [*"ORB: an
  efficient alternative to SIFT or SURF"*](http://www.gwylab.com/download/ORB_2012.pdf), ICCV 2011.
- **Nistér, D., Naroditsky, O., Bergen, J.**, *"Visual Odometry"*, CVPR
  2004. [PDF](https://ieeexplore.ieee.org/document/1315094). One of the
  earliest real-time monocular VO papers.
- **Kümmerle, R. et al.**, [*"g2o: a general framework for graph
  optimization"*](https://www2.informatik.uni-freiburg.de/~stachnis/pdf/kuemmerle11icra.pdf), ICRA 2011. The
  industry-standard pose-graph backend whose residual/parameterization
  conventions we follow.
- **Triggs, B. et al.**, [*"Bundle Adjustment — A Modern Synthesis"*](https://lear.inrialpes.fr/pubs/2000/TMHF00/Triggs-va99.pdf),
  Vision Algorithms Workshop 1999. The canonical BA reference; less
  directly relevant since we don't do full BA here, but it's the book on
  the table for any serious SLAM work.

### Library / tool documentation

- [OpenCV `calib3d` module reference](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — `solvePnPRansac`, `triangulatePoints`, `findEssentialMat`, `Rodrigues`, `ORB`, `BFMatcher`.
- [SciPy `least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) — including the `'trf'` method, `jac_sparsity`, and robust `loss` options.
- [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) — sequence list, calibration conventions, and evaluation metrics.
