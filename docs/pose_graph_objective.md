# The Pose-Graph Objective Function

A from-first-principles walk-through of what
`PoseGraph.optimize` in [`src/stereo_slam.py`](../src/stereo_slam.py)
is actually minimizing, and why.

---

## 1. What the optimizer is varying

The unknowns are the **world poses of the $N$ keyframes**:

$$
T_1,\, T_2,\, \ldots,\, T_N \;\in\; SE(3),
\qquad
T_k \;=\; \begin{bmatrix} R_k & t_k \\ 0^\top & 1 \end{bmatrix}
$$

Each $T_k$ is a $4\times 4$ rigid transform representing the pose of
keyframe $k$'s camera in the world frame. In code these live in
`self.vertices` of `PoseGraph`. Vertex 0 is held **fixed** (via
`PoseGraph.fix(0)`) to pin the gauge; every other pose is free to move.

---

## 2. What the pose graph "knows": the edge constraints

Every edge $(i, j)$ in the graph carries a **measurement**
$T_{ij}^{\mathrm{meas}}$, which is the relative transform from frame $i$
to frame $j$ that some independent sensor / process observed:

- **Odometry edges** (sequential KFs) — measurement comes from the raw
  VO front-end:
  $$
  T_{ij}^{\mathrm{meas}} \;=\; \bigl(T_i^{\mathrm{raw}}\bigr)^{-1}\, T_j^{\mathrm{raw}}
  $$
- **Loop closure edges** (non-sequential KFs) — measurement comes from
  PnP verification between the two keyframes; it's the actual observed
  relative pose between the two views of the same place.

These measurements are **fixed inputs** to the optimization; they never
change during the solve.

---

## 3. What it means for an edge to "agree" with a pose estimate

Given the current estimates $T_i,\, T_j$, the pose graph can **predict**
what the relative transform should be:

$$
T_{ij}^{\mathrm{pred}}(T_i, T_j) \;=\; T_i^{-1}\, T_j
$$

This is the relative transform that falls out of the two world poses.
If the estimates are perfect and the measurement is noise-free,
prediction and measurement agree exactly:

$$
T_{ij}^{\mathrm{meas}} \;=\; T_{ij}^{\mathrm{pred}}
\quad\Longleftrightarrow\quad
\underbrace{\bigl(T_{ij}^{\mathrm{meas}}\bigr)^{-1}\, T_i^{-1}\, T_j}_{=\;T_{\mathrm{err}}^{(ij)}}
\;=\; I
$$

The quantity $T_{\mathrm{err}}^{(ij)}$ is the **error transform** for
edge $(i, j)$. It is the identity matrix exactly when the two estimates
satisfy the measurement.

---

## 4. Turning the error transform into a number

We need a scalar "how wrong is this edge" score. The natural choice is
a 6-vector residual, splitting translation from rotation:

$$
r_{ij}(T_i, T_j) \;=\; \begin{bmatrix} t_{\mathrm{err}}^{(ij)} \\[4pt] \log_{SO(3)}\!\bigl(R_{\mathrm{err}}^{(ij)}\bigr) \end{bmatrix}
\;\in\; \mathbb{R}^6
$$

where $t_{\mathrm{err}}$ and $R_{\mathrm{err}}$ are the translation and
rotation parts of $T_{\mathrm{err}}^{(ij)}$, and $\log_{SO(3)}$ is the
angle-axis representation of the rotation error. This residual is
**zero if and only if the edge is perfectly satisfied**, and it is
small and smooth near that point.

This is exactly what `pose_error_6d(T_err)` computes in
[`src/slam_utils.py`](../src/slam_utils.py):

```python
def pose_error_6d(T_err):
    R_err = T_err[:3, :3]
    t_err = T_err[:3, 3]
    rvec, _ = cv2.Rodrigues(R_err)   # log_SO(3) via angle-axis
    return np.concatenate([t_err, rvec.ravel()])
```

---

## 5. The objective function

Stack all the per-edge 6-vectors into a giant residual vector
$\mathbf{r}$ and minimize its squared $\ell^2$ norm:

$$
\boxed{\;\mathcal{F}(T_1, \ldots, T_N) \;=\; \frac{1}{2} \sum_{(i,j) \in \mathcal{E}} \bigl\lVert\, r_{ij}(T_i, T_j) \,\bigr\rVert^2\;}
$$

Writing the residual out in full:

$$
\mathcal{F}(T_1, \ldots, T_N) \;=\; \frac{1}{2}\!\!\sum_{(i,j)\in\mathcal{E}}\!\! \left\lVert\, \operatorname{pose\_error_{6d}}\!\left(\bigl(T_{ij}^{\mathrm{meas}}\bigr)^{-1}\, T_i^{-1}\, T_j\right) \right\rVert^2
$$

The optimizer searches over all free keyframe poses $T_k$ for the
configuration that makes **every edge measurement as satisfied as
possible**, in a least-squares sense. When constraints disagree (they
always do after a loop closure fires, because odometry edges and the
new loop edge are mutually inconsistent given the old drifted
estimates), the optimizer finds the **compromise** that best trades off
squared error across all edges simultaneously.

---

## 6. Why this is the "right" thing to minimize

Three properties together:

1. **Zero at the answer.** If all measurements were perfect and
   consistent, there exists a choice of poses where every edge residual
   is exactly zero and the cost is zero. Real measurements are noisy,
   so the minimum cost is strictly positive, but the minimizer is still
   the best reconstruction given those measurements.

2. **Symmetric treatment of translation and rotation.** The 6-vector
   $[t_{\mathrm{err}};\;\log R_{\mathrm{err}}]$ penalizes position error
   in meters and rotation error in radians. With equal weighting these
   are on comparable scales for typical VO residuals. A more principled
   code would use a per-edge **information matrix**
   $\Omega_{ij}$ and minimize
   $\tfrac{1}{2}\sum r_{ij}^\top \Omega_{ij}\, r_{ij}$; we use the
   identity for both edge types.

3. **Gauss–Markov optimality.** Under the (local) assumption that edge
   residuals are zero-mean Gaussian, the least-squares minimum is the
   **maximum-likelihood estimate** of the poses. That's why SLAM
   back-ends everywhere — g2o, GTSAM, Ceres, iSAM — all reduce to the
   same non-linear least-squares formulation.

---

## 7. How it's actually parameterized in the code

Directly optimizing over $4 \times 4$ matrices is awkward — they have
to stay on the $SE(3)$ manifold (rotations must stay orthogonal,
translations must sit in the right rows, etc.). Instead we use a
**local right-perturbation parameterization**:

$$
T_k(\xi_k) \;=\; T_k^{(0)} \cdot \exp\!\bigl(\xi_k^{\wedge}\bigr)
$$

where:

- $T_k^{(0)}$ is the **linearization point**, snapshotted at the start
  of each `optimize()` call from the current estimate of vertex $k$;
- $\xi_k \in \mathbb{R}^6$ is a small local increment living in the
  Lie algebra $\mathfrak{se}(3)$;
- $\xi_k = (\rho_k,\, \phi_k)$ has a translation-like part $\rho$ and
  an axis-angle rotation part $\phi$;
- $(\,\cdot\,)^\wedge$ is the hat operator lifting a 6-vector to a
  $4\times 4$ matrix in the Lie algebra;
- $\exp(\xi^\wedge)$ is the $\mathfrak{se}(3) \to SE(3)$ exponential
  map, implemented by `se3_exp_batch` in `slam_utils.py`.

In this parameterization the unknowns become the concatenated vector

$$
\Xi \;=\; \bigl(\xi_1,\, \xi_2,\, \ldots,\, \xi_N\bigr) \;\in\; \mathbb{R}^{6N}
$$

and the objective we hand to scipy is

$$
\mathcal{F}(\Xi) \;=\; \frac{1}{2}\!\!\sum_{(i,j)\in\mathcal{E}} \left\lVert\, r_{ij}\!\Bigl(T_i^{(0)}\exp(\xi_i^\wedge),\; T_j^{(0)}\exp(\xi_j^\wedge)\Bigr) \right\rVert^2
$$

with initial guess $\Xi = 0$ ("start at the current poses, no
perturbation").

At $\Xi = 0$, all **odometry** residuals are identically zero by
construction — we chose the measurement to be the relative transform of
the current poses, so prediction matches measurement exactly. Only
**loop closure** edges have non-zero residuals at the initial guess,
which is exactly where all the optimization work gets done. The fixed
vertex 0 has $\xi_0 \equiv 0$ and is excluded from the parameter vector
so the gauge is pinned.

After scipy returns the optimal $\Xi^{*}$, we bake the increments back
into the vertex poses:

$$
T_k^{\mathrm{new}} \;=\; T_k^{(0)}\,\exp\!\bigl(\xi_k^{*\,\wedge}\bigr)
$$

which is exactly what the last loop of `PoseGraph.optimize` does:

```python
vertices_final = vertices0 @ se3_exp_batch(xi_all_final)
for k in range(n):
    self.vertices[k] = vertices_final[k]
```

---

## 8. The correspondence between the formula and the code

| Math | Code (`PoseGraph.optimize`) |
|---|---|
| $T_k^{(0)}$ — linearization point | `vertices0 = np.stack(self.vertices)` |
| $T_{ij}^{\mathrm{meas}}$ — edge measurement | `T_meas` (stacked from `self.edges`) |
| $\bigl(T_{ij}^{\mathrm{meas}}\bigr)^{-1}$ — precomputed inverse | `T_meas_inv = rigid_inverse_batch(T_meas)` |
| $T_k^{(0)}\exp(\xi_k^\wedge)$ — updated pose | `poses = vertices0 @ se3_exp_batch(xi_all)` |
| $T_i^{-1}\, T_j$ — predicted relative | `T_pred = rigid_inverse_batch(Ti) @ Tj` |
| $\bigl(T_{ij}^{\mathrm{meas}}\bigr)^{-1}\, T_i^{-1}\, T_j$ — error | `T_err = T_meas_inv @ T_pred` |
| $r_{ij}$ — 6-vector residual | `trans` + `rot_to_angle_axis_batch(T_err[:, :3, :3])` |
| $\mathcal{F}(\Xi)$ — sum of squares | `least_squares(residuals, x0, method='trf', …)` |

`scipy.optimize.least_squares` takes the residual function, computes
the Jacobian via finite differences (using `jac_sparsity` to exploit
the block-sparse structure — each edge only touches two 6-blocks of
the parameter vector), and runs a trust-region reflective iteration
to find the $\Xi$ that minimizes $\mathcal{F}$.

---

## One-sentence summary

> The pose graph's objective is the squared sum of **disagreements**
> between each edge's observed relative pose and the relative pose
> implied by the two keyframe estimates — minimized over all free
> keyframe poses, with keyframe 0 held fixed to pin the world frame.

---

## Further reading

- **Grisetti, Kümmerle, Stachniss, Burgard**, *"A Tutorial on
  Graph-Based SLAM"*, IEEE ITS Magazine, 2010.
  [PDF](https://www.cs.cmu.edu/~kaess/pub/Grisetti10titsmag.pdf) —
  a gentle, thorough derivation of the entire pose-graph formulation.
- **Kümmerle et al.**, *"g2o: A General Framework for Graph
  Optimization"*, ICRA 2011.
  [PDF](https://www2.informatik.uni-freiburg.de/~stachnis/pdf/kuemmerle11icra.pdf) —
  the reference implementation whose residual and parameterization
  conventions we follow.
- **Solà, Deray, Atchuthan**, *"A Micro Lie Theory for State
  Estimation in Robotics"*, arXiv 2018.
  [arXiv:1812.01537](https://arxiv.org/abs/1812.01537) — the clearest
  concise treatment of $\mathfrak{se}(3)$ parameterizations for SLAM
  practitioners.
- **Barfoot**, *State Estimation for Robotics*, CUP 2017.
  [Free PDF](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) —
  chapter 7 (Matrix Lie Groups) is the definitive textbook treatment.
