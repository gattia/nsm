# Plan: Improving NSM mesh interpolation (cheap numerical fixes)

**Status:** Proposed — not started. Hand-off plan for a side agent.
**Created:** 2026-05-18 · **Revised:** 2026-05-18 (split out of the original
rectified-flow plan; Track B moved to `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md`).
**Repo:** `/dataNAS/people/aagatti/programming/NSM/` (NSM). Sole production
consumer: `nsosim` → the comak gait pipeline.
**Goal:** Improve the quality of NSM point correspondences — the map from the
vertices of one shape to the surface of another — by fixing the numerical
SDF-stepping in `NSM/NSM/mesh/interpolate.py`. No retraining. The current
method produces poor matches (vertex collapse, crossings, off-surface drift),
worst at osteophytes (bone) and on the thin meniscus shells. Better
correspondences directly improve `interp_ref_to_subject_to_osim`
(template→subject warping) used throughout the comak pipeline.

> **Motivating context.** The comak meniscus ligament-attachment fix
> (`comak_gait_simulation/.claude/plans/MENISCUS_LIGAMENT_ATTACHMENT_FIX.md`)
> warps a reference meniscus mesh onto subjects via this correspondence; the
> warp quality was flagged as poor. That plan is **not blocked** on this work —
> labeling tolerates a rough warp — but this is the deeper fix.

> **Scope split.** This plan is the cheap, no-retraining numerical work (former
> "Track A"). Two sibling plans hold the rest:
> - `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` — a learned rectified-flow
>   correspondence operator (former "Track B"). A separate, larger effort to do
>   *after* this one; it reuses this plan's improved stepping as its seed.
> - `NSM_TRAINING_IDEAS.md` — upstream decoder-training changes (Eikonal along
>   paths, normal-smoothness regularizer) that would make this stepping exact.
>   Those require retraining and are out of scope here.

---

## 1. The current method and why it fails

### What it does

Given two shapes as NSM latents `z_A, z_B`: discretize A as a mesh, take its
vertices, and "flow" them onto B's surface while the latent linearly
interpolates `z(t) = slerp(z_A, z_B, t)`. Implemented in
`NSM/NSM/mesh/interpolate.py`:

- `interpolate_points(model, latent1, latent2, n_steps=100, points1, surface_idx, spherical=True)`
  (`interpolate.py:327`) → `interpolate_common` (`:243`).
- The loop (`:269-319`): for each of `n_steps=100` steps, advance the latent by
  `1/100` (slerp), then call `update_positions` **exactly once**.
- `update_positions` (`:182`): one first-order surface projection,
  `x ← x − SDF(x)·n̂`, where `n̂` is the **unit** spatial gradient
  `∇_x SDF / ‖∇_x SDF‖` (`:219-237`). The gradient **magnitude is discarded**;
  flat points (`‖∇SDF‖ < 1e-8`) are left unmoved. Both the SDF value and the
  gradient are evaluated at the **same** latent `z(t)` — there is no latent
  mixing; the points simply lag one step (they sit on `z(t−1)`'s level set when
  the `z(t)` field is queried).
- `sdf_gradients` (`:27`) computes **only** `∇_x SDF` via autograd. It never
  computes `∂SDF/∂z` (the placeholder for it is allocated but always zero).

The endpoint map `x(0) ∈ surface_A → x(1) ∈ surface_B` is the correspondence.

### Why it is poor

The procedure is a crude discretization of the ODE the implicit-function
theorem gives for keeping a point on the moving level set:
`dx/dt ≈ −(∂SDF/∂z · dz/dt) / ‖∇_x SDF‖² · ∇_x SDF`. Concrete defects:

1. **Under-converged projection, with no corrector on the terminal step.** One
   projection per latent increment. The off-surface residual is *not* an
   unbounded accumulation — every step re-projects against the true current
   level set, so the residual is bounded by roughly one step's projection
   error. The real costs are: (a) the **terminal** step at `z_B` has no
   successor to correct it, so the output's off-surface error *is* that last
   residual; (b) a poorly-converged step leaves the next step's Newton
   linearisation anchored at a worse point, feeding tangential slip.
2. **Drops the gradient magnitude.** `x ← x − SDF·n̂` is the true Newton step
   `x − SDF·∇SDF/‖∇SDF‖²` *only if* `‖∇SDF‖ = 1` (Eikonal). The NSM decoder is
   **not** Eikonal-trained (`losses.py` has an Eikonal loss; NSM `CLAUDE.md`
   says it is untested) → the step systematically over/undershoots. Magnitude
   of the error is unknown until measured (see Phase 0).
3. **Never uses `∂SDF/∂z`.** The latent-advection term of the ODE above is
   ignored; the method only re-projects after the fact (corrector with no
   predictor).
4. **No correspondence regularization.** The points-only path (`is_mesh=False`,
   the one `nsosim` uses) has *no* smoothing, no anti-crossing, no
   anti-collapse. The `is_mesh=True` path *does* add per-step Laplacian/Taubin
   smoothing (VTK, via pyvista) — `nsosim` simply does not call it. `nsosim`
   **can adapt to use the mesh path**: it uses points-only for speed, but with
   everything in memory that gap may be small, and if the mesh path gives
   materially better correspondences the extra time is worth it. So
   "points-only" is not a fixed constraint — see Fix 4.
5. **The coupling is emergent, not chosen.** Every step moves points purely
   along the surface normal (closest-point projection). The *tangential*
   degree of freedom — which B-point an A-vertex lands on — is left at zero by
   default. Nothing constrains trajectories to be non-crossing or the endpoint
   map to be bijective. **This plan does not fix defect 5** — that is the
   rectified-flow plan. The fixes here improve off-surface accuracy,
   integration stability, and mildly resist collapse/crossing.

**Thin meniscus shells** (the immediate pain): a meniscus is a thin C-wedge; its
SDF has an interior medial skeleton/ridge where `∇SDF` flips. A vertex near the
ridge gets projected to the wrong wall. A *large* latent step can jump a vertex
clean across the ridge — being accurate w.r.t. the ODE does not help once you
have stepped across a crease. The mitigation is small steps near the ridge,
which motivates adaptive step-sizing (Fix 5).

**Bone osteophytes** (the harder case): A has an osteophyte, B does not →
no continuous bijection exists. The current method resolves this by
*collapsing* many A-vertices onto one B-point — pathological. No Track-A fix
repairs this; it is intrinsic (rectified-flow plan, and its Jacobian detector).

### No way to measure any of this

There are **no correspondence-quality metrics** in the repo — only
*reconstruction* metrics (Chamfer/ASSD/EMD in `reconstruct/recon_evaluation.py`).
Off-surface drift, fold-over, collapse, bijectivity must be built (Phase 0).

---

## 2. Phase 0 — diagnostics and baseline (prerequisite)

Build a correspondence-quality evaluation. None exists today. Every fix in §3 is
gated behind a kwarg and re-scored here.

### 2.1 Test data — selection, sourcing, cache once

Surface extraction (marching cubes) is the slow part; interpolation itself is
fast. **Extract once, cache, then iterate.**

**Subject selection (10 knees, KL-stratified).** Demographics with KL grade:
`/dataNAS/people/aagatti/projects/OAI_DESS/aging_trajectories/data/demographics/0_demographics_baseline.csv`
— one row per knee, columns `id`, `side` (LEFT/RIGHT), `kl`, plus per-region
`osteophytes_fem_*_score` (useful later for the bone pass). Pick **10 knees:
4× KL0, 3× KL1, 3× KL2**. **Exclude KL3/4** for now — those shapes are more
likely badly degenerate and would confound the menisci pilot. This deliberately
spans easy→moderate morphs so the metrics can discriminate between fixes.

**Meshes.** Per-subject mesh folders:
`/dataNAS/people/aagatti/projects/OAI_DESS/meshes/00m/{id}/` — per side and
structure, e.g. `{id}_{SIDE}_femur.vtk`, `_femur_cart.vtk`, `_med_men.vtk`,
`_lat_men.vtk`. The femur NSM model is a **joint 4-surface model** (bone, cart,
med_men, lat_men → `surface_idx` 0/1/2/3, per its `mesh_names`) fit with **one
latent per knee** — so fitting needs **all four** of these meshes per knee, not
just the menisci. "Menisci first" (below) refers to which `surface_idx` is
*interpolated and scored*, not which meshes are used to fit.

**Laterality check (resolve before fitting in bulk).** Traced finding:
`reconstruct_mesh` / `fit_nsm` are **never told the side** and contain **no
flip/mirror code** — and similarity registration cannot mirror. So LEFT/RIGHT
consistency is *not* handled in the fitting path; it must already be baked into
the OAI_DESS mesh files (pre-oriented to right-knee at mesh-generation time —
the `_LEFT_`/`_RIGHT_` in the filename is then just an origin label, not the
geometry orientation). **Verify empirically:** fit one LEFT-labeled and one
RIGHT-labeled knee, reconstruct, and confirm both come out in the same
orientation (medial meniscus on the same side of the lateral). If they do not,
the meshes need pre-flipping before fitting — a blocker to resolve first.

**Latents — fit via nsosim's production path.** Interpolation runs in NSM
latent space, so each selected knee needs a latent. **Fit it the exact way
`nsosim` does** — do not reimplement, and do not look up a (possibly stale)
`latent_codes` row. Call `nsosim.utils.fit_nsm`
(`/dataNAS/people/aagatti/programming/nsosim/nsosim/utils.py`) directly; it
loads the model and runs `NSM.reconstruct.reconstruct_mesh`. Verified production
settings (traced through `comak_1_nsm_fitting.py` → `align_knee_osim_fit_nsm` →
`fit_nsm`, and `config/default_generic_gait.json`):

- **Model:** `nsm_models/568_nsm_femur_bone_cart_men_v0.0.1/model/2000.pth` +
  `model_params_config.json` (under `COMAK_SIMULATION_REQUIREMENTS/`).
  `model_type="triplanar"`, `latent_size=1024`, `objects_per_decoder=4`,
  `mesh_names=[bone, cart, med_men, lat_men]`.
- **Explicit nsosim arguments:** `n_samples_latent_recon=20000`,
  `num_iter=None` (→ uses the model config's `num_iterations_recon=2000`),
  `convergence_patience=10` (an nsosim **override** — the model config's own
  `convergence_patience_recon` is 50; production uses 10),
  `use_hybrid_optimizer=False`, `seed=0`.
- **Inherited from the model config (do not change):** `lr_recon=0.005`,
  `clamp_dist_recon=0.1`, `code_regularization=True` /
  `code_regularization_weight=0.0001` (identity prior), `n_lr_updates_recon=100`,
  `lr_update_factor_recon=1.1`, `convergence_type_recon='recon_loss'`,
  `scale_method='max_rad'`, `scale_jointly=True`, `latent_init_std=0.01`,
  `latent_bound=10`.

Save each fitted latent and the fit's scale/registration transform alongside
the cached reconstruction. (`use_hybrid_optimizer` stays `False` — the
Adam+LBFGS hybrid recipe was evaluated and rejected; ignore it.)

**All four surfaces.** Fit uses all 4 femur-model surfaces; *score and
interpolate* **all four** — bone (`surface_idx` 0), cart (1), med_men (2),
lat_men (3) — and compute every metric **separately per surface**, storing all
results. The surfaces fail differently (bone: osteophyte topology change; cart:
thin sheet; menisci: thin C-wedge with a medial ridge), so per-surface scores
reveal how each fix helps or hurts each one — high-value insight for a small,
one-time compute cost, and it avoids re-running the whole sweep later. Menisci
remain the *priority* read (the active need) but are no longer scoped alone.
The `osteophytes_fem_*_score` columns can additionally tag which KL1/2 knees
carry osteophytes, for richer interpretation of the bone results.

**Cache.** Reconstruct each selected shape via marching cubes, save meshes +
latents to disk. All §3 experiments load these and only run interpolation —
quick, easy to fan out as background scripts. Evaluation runs in
NSM-normalized space (target = `marching_cubes(z_B)`); the real OAI `.vtk`
meshes are available as an external reconstruction-quality check but are not
the primary target.

### 2.2 The warp, both directions

For an ordered latent pair `(z_A, z_B)`: take cached `mesh(z_A)`'s vertices, run
`interpolate_points` to `z_B` → warped point set; `mesh(z_B)` is the target.
**Always also run the reverse**, `z_B → z_A`: different parts of the surfaces
are degenerate in each direction, so both directions are needed to see the full
picture (and the round trip gives bijectivity). 10 latents → 90 ordered pairs,
each warped for all 4 `surface_idx` values.

### 2.3 Metrics — two families

Every metric below is computed **separately for each `surface_idx`** (bone,
cart, med_men, lat_men) and all per-surface results are stored, so the
experiment matrix can show how each fix affects each surface.

**Family 1 — surface fit + mesh health:**
- ASSD warped ↔ target.
- Directed distance percentiles both ways (warped→target, target→warped):
  min/25/50/mean/75/95/max. The **target→warped** direction catches coverage
  gaps — a collapse symptom.
- Off-surface error: `|SDF(x_warped, z_B)|` distribution (cheap; no second
  marching-cubes needed).
- Warped-mesh triangle health: edge-length / area stats (mean/std/min/max),
  aspect ratio, degenerate-triangle count.
- Warped-mesh self-intersection count.

**Family 2 — correspondence-specific (the half that does not exist yet):**
- **Fold-over:** count of warped triangles whose normal flipped vs the source
  mesh (reuse `mesh/triangle_metrics.py` ideas).
- **Bijectivity:** flow A→B→A, per-vertex round-trip distance.
- **Forward–backward disagreement field** — the topology-mismatch signal.

> **On the collapse/injectivity metric.** A standalone nearest-neighbour /
> point-density metric is *largely redundant* here: local collapse already
> shows up in **triangle health** (shrinking edges/areas, degenerate count),
> and distant-patch overlap shows up in the **self-intersection count** and the
> **target→warped** coverage gap. The one thing triangle metrics miss — two
> non-adjacent patches mapping to the same place — is caught by
> self-intersections. So do **not** build a separate KD-tree density metric;
> the cheap `min edge length` / collapsed-edge count falls out of triangle
> health for free and is sufficient.

### 2.4 NFE sensitivity

Run every config at `n_steps ∈ {10, 25, 50, 100, 200}`. This is what
distinguishes a real fix from "just use more steps," and it reveals how stiff
the trajectories are.

### 2.5 Implementation — a reusable metrics module

Build the metrics as a **proper NSM library module**, not a one-off script:
`NSM/NSM/mesh/correspondence_metrics.py`, with clean, documented function APIs —
one function per metric, plus a top-level scorer that takes a warped mesh /
point set + target and returns a results dict. This module is reused for this
project's experiment matrix *and* for diagnosing interpolation in future
projects, so keep it project-agnostic. A thin experiment-runner script (in
`testing/` or a project scratch dir) calls the library, sweeps the config
matrix, and writes the report.

**Deliverable:** a baseline report — current `interpolate_points` scored on the
pilot pairs, both directions, **all four surfaces** (per-surface metrics),
across the NFE grid.

*Optional diagnostic:* the module can also report the `‖∇_x SDF‖` distribution
near the surface (how far the decoder is from Eikonal). It is cheap and
informative but **not gating** — Fix 2 is applied regardless because it is free
and exact; no Eikonal assessment is needed to proceed.

---

## 3. The fixes — a menu of independently-gated options

Each is a localized, kwarg-gated edit to `interpolate.py` so the old behaviour
is reproducible and each option can be tested **alone** and then **composed**.
Re-score with Phase 0 after each. Implement all of them; the experiment matrix
(§3.7) decides which to keep.

### Fix 1 — Per-step convergence loop (corrector)

Replace the single `update_positions` call in `interpolate_common` with an inner
loop: iterate the projection until `max|SDF| < tol` or a small cap (e.g. 5).
Drives points back onto each level set; in particular guarantees the terminal
step converges (defect 1). Refactor `update_positions` to keep tensors on the
GPU across iterations (today it returns `.cpu()` and the caller re-uploads).
- **Compute:** ~2–3× decoder evals (offsettable by reducing `n_steps`).
- **Retrain:** no. **Difficulty:** low.

### Fix 2 — True Newton step (magnitude)

In `update_positions`, step `x ← x − SDF·∇SDF/‖∇SDF‖²` instead of `x − SDF·n̂`
(defect 2) — i.e. multiply the existing step by an extra `1/‖∇SDF‖`. This is the
exact first-order root-finding step; it equals the current step only when
`‖∇SDF‖ = 1`. Keep the flat-point clamp.
- **Compute:** none extra. **Retrain:** no. **Difficulty:** trivial.
- **Expected payoff:** *conditional* — only matters if the decoder is
  non-Eikonal. It is free, exact, and a no-op in the good case → **do it
  regardless**; no Eikonal assessment is needed to justify it. The optional
  `‖∇_x SDF‖` diagnostic (§2.5) explains the size of any observed gain after
  the fact, but is not a prerequisite. Do not expect a headline gain.

### Fix 3 — Latent-advection predictor (`∂SDF/∂z`)

Wire `∂SDF/∂z` into `sdf_gradients` (add `latent.requires_grad_(True)` to the
`autograd.grad` inputs — the decoder is already differentiable in `z`;
`reconstruct/main.py` backprops into latents). Plumb the latent increment `dz`
through `interpolate_common → update_positions`. Use the implicit-function ODE
step `dx = −(∂SDF/∂z·dz)/‖∇_x SDF‖² · ∇_x SDF` as a **predictor**, then
re-project (Fix 1) as **corrector** → a predictor–corrector integrator.
- **Compute:** negligible extra (one combined backward over position + latent).
- **Retrain:** no. **Difficulty:** low–moderate (the `dz` plumbing).
- **Honest framing:** the least-norm predictor moves points purely along the
  normal — the *same* subspace as the corrector — so at a fixed fine
  `n_steps` it does **not** change where points land; it is an
  integration-*accuracy/speed* fix that lets you reach the 100-step-quality
  answer in ~10–20 steps. Its payoff scales with how aggressively you want to
  cut step count (and with rectified-flow data generation). It is also the
  conceptual bridge to that plan: it makes the free tangential degree of
  freedom explicit. Low priority for a one-off warp; do it for speed.

### Fix 4 — Correspondence regularization (two options)

This is the only Fix that touches the tangential DOF, so the only one that can
move bijectivity/collapse — but mildly; it cannot fix a fundamentally wrong
coupling. Two ways to get it, both worth testing:

**Fix 4a — use the existing mesh path (zero new stepping code).** The
`is_mesh=True` path already applies per-step Laplacian/Taubin smoothing via VTK
(pyvista `.smooth()` / `.smooth_taubin()`, `interpolate.py:299-309`). `nsosim`
can adapt to call `interpolate_mesh` instead of `interpolate_points` — it uses
points-only for speed, but with everything in memory that gap may be small and
better correspondences are worth the time. So the cheapest "Fix 4" is just: run
the mesh path and measure. Caveat: VTK smoothing is **full 3D** — it pulls
points off-surface and shrinks; the existing path does not re-project after
smoothing, so it trades off-surface error for regularity.

**Fix 4b — tangent-projected Laplacian on the points path (new code).** The
refined version: per-vertex Laplacian `Δx` from source-mesh connectivity, remove
the normal component (`Δx_tan = Δx − (Δx·n̂)n̂`), nudge `x ← x + α·Δx_tan`, then
re-project (Fix 1). Smoothing *only in the tangent plane* redistributes points
along the surface without the off-surface pull that 4a's 3D smoothing causes.

*Where to compute 4b — keep it on the GPU.* Build the mesh Laplacian once as a
torch **sparse adjacency matrix** (from the source-mesh faces) and apply it as a
sparse matvec each step. The normals `n̂` are already on the GPU (they are the
`∇_x SDF` from the projection step), so the tangent projection is elementwise
on-device. **Do not offload to CPU/VTK per step:** VTK is used elsewhere for
*end-of-pipeline* mesh smoothing (a one-shot operation), but here smoothing runs
inside a 100+-step loop — a per-step CPU round-trip would dominate runtime. VTK
also cannot do tangent-plane-restricted smoothing, which is the whole point of
4b. So 4b is a pure-GPU op; 4a is the CPU/VTK path and already exists.

- **Compute:** 4a — the mesh path's existing overhead. 4b — +20–50% (sparse
  matvec + a re-projection pass), all on GPU.
- **Retrain:** no. **Difficulty:** 4a trivial (a call-site change in `nsosim` +
  measuring); 4b moderate (faces/adjacency threaded through
  `interpolate_points`, build the sparse Laplacian, tune `α` / `n_iter`).

### Fix 5 — Adaptive step-sizing

Detect when a latent increment is too large and subdivide it. **Whole-mesh
re-update design** (chosen for now — simple, fully vectorised; refine later only
if profiling on bone demands it): all points stay on one shared latent schedule;
at a candidate step over `[t, t+Δ]`, estimate the error and, if the *worst*
point exceeds tolerance, reject the step, halve `Δ`, and retry — everyone
together.

*Error estimators (pick one; both cheap):*
- *Residual-driven:* the `max|SDF|` left after the step (free once Fix 1
  exists). Simple, but measures off-surface error, not trajectory error.
- *Richardson:* one full step vs two half-steps; `max‖x_big − x_small‖`
  estimates the local truncation error of the *trajectory*. Costs ~2–3× evals
  on checked steps; the better signal for the ridge-jumping problem.

*Concrete control structure — the three things that make this safe* (and answer
"how do we not get stuck splitting forever?"):
1. **Step-size floor.** Define `Δ_min` (e.g. nominal `1/n_steps` divided by
   `2^max_depth`, `max_depth ≈ 4–5`). Never subdivide below `Δ_min` — accept the
   step and flag it. This is the hard guarantee against an infinite split loop.
2. **Subdivision-depth counter.** Carry an integer recursion depth; each halving
   increments it; refuse to subdivide past `max_depth`. (Equivalent to the
   floor, tracked explicitly so the report can show *where* on the path the
   integrator struggled — a useful diagnostic and a topology-mismatch hint.)
3. **Scale-relative tolerance.** Make tolerance relative to mesh scale, not
   absolute — e.g. a small fraction of the source mesh's **median edge length**
   (for the Richardson `‖Δx‖` estimator), or a small multiple of Fix 1's
   convergence `tol` (for the residual estimator). Scale-relative tolerances
   transfer across menisci and bone without re-tuning.

When a step is accepted at the floor (case 1) without meeting tolerance, that
location is exactly the meniscus ridge / high-curvature region — record it; it
is where small steps were needed and is a candidate for the topology signal.
Per-point sub-stepping is a later optimisation; not now.
- **Compute:** ~2–3× on refined steps. **Retrain:** no. **Difficulty:**
  low–moderate.

### Fix 6 — Batched line-search / quadratic-fit magnitude step

An **alternative implementation** of the magnitude correction (Fix 2), and an
independently-toggleable option. Instead of the analytic Newton scale, evaluate
the SDF at several candidate step scalings `x − α·SDF·n̂` in **one batched,
forward-only pass** (no backward needed to score candidates), fit the *signed*
`SDF(α)` (≈ linear) or `SDF(α)²` (≈ parabola) and jump to its zero/minimum.
- 3 points → exact quadratic; 5 points → robust least-squares fit. The fit
  itself is a batched 3×3 solve — negligible. Cost is the extra forward evals
  (2 extra for 5 vs 3); fine with GPU headroom, scales ~linearly for the large
  bone meshes.
- **Relationship to other fixes:** Fix 6 and Fix 2 are two ways to fill the
  *magnitude* slot — use one or the other, not both. Both compose with Fix 1
  (direction/curvature via re-projection), Fix 3, Fix 4, Fix 5. Fix 6's niche
  over Fix 2: robustness where `‖∇SDF‖` is untrustworthy (off-surface, near the
  meniscus ridge).
- **Compute:** ~5× forward (no backward) for the line search. **Retrain:** no.
  **Difficulty:** low–moderate.

### 3.7 Experiment matrix

Cumulative ladder, not strict one-at-a-time (some cells are incoherent — Fix 3
needs Fix 1; Fix 4 needs Fix 1 to clean up its off-surface error):

`baseline → +Fix2 → +Fix2+Fix1 → +Fix3 → +Fix4 → +Fix5 → all`

Also run the standalone-meaningful configs: **Fix 1 alone**, **Fix 2 alone**,
and **Fix 6 in place of Fix 2** (swap test). Score each config across the NFE
grid (§2.4), both warp directions, all four surfaces (per-surface). Pick the
knee of the quality-vs-(implementation + compute) curve — and note the best
config may differ per surface; that per-surface breakdown is itself a
deliverable.

### 3.8 Final validation

Re-run the comak meniscus warp
(`comak_gait_simulation/tests/meniscus_ligament_attachment/`) with the chosen
configuration and compare warped-mesh quality against baseline.

---

## 4. Honest caveats

- These fixes improve **off-surface accuracy, integration stability, and step
  efficiency**, and Fix 4 mildly resists collapse/crossing. They do **not** fix
  the emergent-coupling / non-bijectivity problem (defect 5) or the osteophyte
  topology mismatch — those are the rectified-flow plan.
- Fix 2 / Fix 6's payoff is conditional on the decoder being non-Eikonal, but
  both are applied regardless (free + exact); the `‖∇_x SDF‖` diagnostic
  (§2.5) is optional and explanatory, not gating. The principled upstream fix —
  Eikonal training along interpolation paths — is in `NSM_TRAINING_IDEAS.md`.
- The meniscus medial ridge is *intrinsic* to a thin-shell SDF; no Track-A fix
  removes it. Small steps near it (Fix 5) keep vertices from jumping walls.

## Code touchpoints

- `NSM/NSM/mesh/interpolate.py` — current method; all fixes
  (`interpolate_common:243`, `update_positions:182`, `sdf_gradients:27`).
- `NSM/NSM/models/triplanar.py:166,330` — `TriplanarDecoder`; multi-surface SDF
  query, autograd source of `∇_x SDF` and `∂SDF/∂z`.
- `NSM/NSM/mesh/triangle_metrics.py` — reuse for fold-over / triangle health.
- `NSM/NSM/reconstruct/recon_evaluation.py` — existing (reconstruction) metrics;
  Phase 0 adds correspondence metrics alongside (new
  `NSM/NSM/mesh/correspondence_metrics.py`).
- `nsosim/nsosim/nsm_fitting.py:545` — sole production caller of
  `interpolate_points`; the consumer this work improves.
- `nsosim/nsosim/utils.py::fit_nsm` — production NSM latent-fitting entry point
  to reuse for Phase 0 (wraps `NSM.reconstruct.reconstruct_mesh`).
- `nsosim/nsosim/nsm_fitting.py::align_knee_osim_fit_nsm` — production wrapper
  (rigid registration + `fit_nsm`); the fuller path if pre-registration matters.
- `comak_gait_simulation/run_simulations/scripts/comak_1_nsm_fitting.py` +
  `comak_gait_simulation/config/default_generic_gait.json` — where the
  production fitting parameters are set.
- Data: `COMAK_SIMULATION_REQUIREMENTS/nsm_models/568_nsm_femur_bone_cart_men_v0.0.1/`
  (`model/2000.pth`, `model_params_config.json` with
  `mesh_names=[bone,cart,med_men,lat_men]`). Tibia/patella models —
  `650_nsm_tibia_v0.0.1`, `648_nsm_patella_v0.0.1` — exist but are not needed
  for the menisci pilot (the femur model carries the menisci).

## Related

- `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` — the learned correspondence operator;
  the deeper fix for defect 5, to do after this plan.
- `NSM_TRAINING_IDEAS.md` — upstream decoder-training levers (Eikonal along
  paths, normal-smoothness regularizer).
- `comak_gait_simulation/.claude/plans/MENISCUS_LIGAMENT_ATTACHMENT_FIX.md` —
  the warp this work improves (not blocked on it).
- ShapeMed-Knee paper: Gatti et al. 2024.
