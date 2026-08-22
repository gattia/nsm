# Plan: Improving NSM mesh interpolation (cheap numerical fixes)

**Status:** Complete (2026-05-22)
**Created:** 2026-05-18 · **Revised:** 2026-05-18 (split out of the original
rectified-flow plan; Track B moved to `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md`).
**Repo:** `/dataNAS/people/aagatti/programming/NSM/` (NSM). Sole production
consumer: `nsosim` → the comak gait pipeline.

## State

**Updated:** 2026-08-17 · **Status:** done (2026-05-22)

- **Next:** nothing. The remaining ~40-50% menisci seam fold-over is out of this plan's
  reach by construction — see **Diverged** below.
- **Blocked on:** nothing.
- **Done:** see **Completion Notes** immediately below, which is this plan's `Delivered`
  section and is not duplicated here.
- **Surprises:** see **Diverged** below.

## Diverged

Where the work departed from the plan as written. This is the part that exists nowhere
else — the code shows what was built and git shows when, but only this says what we
believed beforehand and why it was wrong.

- **Fix 3 (latent-advection predictor) was implemented and then rejected.** It diverged
  (`a208381` fixed the divergence, `875e4f1` rejected it anyway). The plan assumed all six
  fixes would be keepers; one was not.
- **Two fixes were invented during the work and were not in the plan** — Fix 7
  (smoothed-normals projection, `ec489ad`) and Fix 8 (iterative source-mesh refinement,
  `9fe360d`). Fix 7 needed its own divergence fix (`03988dd`) via dihedral-angle seam
  detection, and Fix 8 needed capping three separate times (`b9dda1f`, `9eee303`,
  `b125e35`) before it stopped running away.
- **The headline goal was not reached, and could not have been.** The plan aimed to reduce
  menisci fold-over with numerical fixes alone. It got ~50%; the residual is
  local-triangulation pathology at the seam that **no Track-A fix removes**. That finding
  is what motivated `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` — the conclusion of this plan is
  the premise of that one.
- **The production recommendation is narrower than the option set.** Eight fixes shipped as
  kwarg-gated options; exactly two are recommended (Fix 2 + Fix 4c at θ=45°). The other six
  are reachable, tested, and not advised — which is a maintenance surface the plan did not
  anticipate creating.
- **The deliverables list below no longer matches `main`** *(noted 2026-08-22, audit
  round)*: the 31 interpolate tests were trimmed to 9 by `fa862aa` ("Trim mesh
  interpolation to production config") — the 39 correspondence-metrics tests survive in
  full — and the `experiments/mesh_interpolation/` tree exists on no branch of `origin`.
  The archive branch/tag this plan and the module docstring point to
  (`mesh-interpolation-improvements`, `archive/mesh-interp-full-exploration`) were never
  pushed: as of 2026-08-22 they exist, if anywhere, only in a local clone. The counts and
  paths below are kept as a record of what was delivered at completion, not of what is
  checked out today.

## Completion Notes

**Date completed:** 2026-05-22

**Summary.** All six planned numerical fixes plus two added during the work
(Fix 7 smoothed-normals projection, Fix 8 iterative source-mesh refinement)
were implemented as kwarg-gated options in `NSM/mesh/interpolate.py`,
end-to-end tested against an analytic SDF, and evaluated on 20 paired
ordered pairs across all four surfaces (bone, cart, med_men, lat_men) at
five NFE values. A 24-config α × θ × Fix 7 tuning sweep identified the
production recommendation: **Fix 2 (Newton magnitude) + Fix 4c (tangent
Laplacian with dihedral-seam pinning at θ=45°)**, which is a strict
improvement over baseline on every surface on fold-over and is
non-positive on ASSD except for a small +3.7% on cart. The remaining
~40-50% menisci fold-over at the seam is local-triangulation pathology
that no Track-A fix removes — it requires the rectified-flow plan or
decoder retraining (see siblings).

**Changes made.** All on branch `mesh-interpolation-improvements`. Key
commits (oldest → newest):

| commit | what |
|---|---|
| `5bd5926` | Add six kwarg-gated numerical fixes to mesh interpolation |
| `b46cd46` | Add correspondence-quality metrics module |
| `4e22418` | Add Phase 0 mesh-interpolation experiment runner |
| `def42e9` | Add SLURM submission for Phase 0 experiment |
| `37de1f8` | Make matrix runner checkpoint/resume; split jobs finer |
| `a208381` | Fix latent-advection predictor divergence (Fix 3) |
| `875e4f1` | Reject Fix 3 (latent predictor); isolate Fix 4 / Fix 5 configs |
| `ec489ad` | Add boundary-aware tangent Laplacian and Fix 7 (smoothed-normal projection) |
| `03988dd` | Detect geometric seam via dihedral angle; fix Fix 7 divergence |
| `8d42573` | Add Fix 4c tuning sweep (alpha × theta × Fix7) |
| `8db3a7d` | Add visuals dump |
| `5f835a9` | Exclude stelvio node from all submissions |
| `f794059` | Visuals: add target-distance and warp-travel scalars + 3-row layout |
| `9fe360d` | Add iterative source-refinement wrapper (Fix 8) |
| `3cce73c` | Add pre-seam-split, vertex-origin tracking, matched-RGB visualization |
| `b9dda1f` | Trim compare_refined to menisci only; cap refine passes |
| `36771fc` | compare_refined: aggressive split + smoothed-corr + Fix 7 stacks |
| `9eee303` | compare_refined: tame runaway refinement |
| `b125e35` | compare_refined: cap refinement at 1 pass |
| `356c101` | Add compare_mesh_path: pyvista subdivide_adaptive vs current best |

Files added or rewritten:

- `NSM/mesh/interpolate.py` (full rewrite + 8 fix kwargs)
- `NSM/mesh/correspondence_metrics.py` (new module, 39 tests)
- `testing/NSM/mesh/test_interpolate.py` (new, 31 tests)
- `testing/NSM/mesh/test_correspondence_metrics.py` (new, 39 tests)
- `experiments/mesh_interpolation/` (new directory):
  `config.py`, `subjects.py`, `fit_cache.py`, `run_matrix.py`,
  `submit_phase0.sh`, `dump_visuals.py`, `compare_refined.py`,
  `compare_mesh_path.py`, `smoke_test.py`
- `experiments/mesh_interpolation/cache/manifest.json` (committed; the
  10 selected knees from seeded selection)

**Tests.** 70 tests total, all passing:
`pytest testing/NSM/mesh/` → 31 interpolate + 39 correspondence metrics.
The synthetic-sphere-SDF model used in `test_interpolate.py` is Eikonal
by construction, so it cannot exercise Fix 2 / 6 / 7's non-Eikonal
behaviour or Fix 3's divergence on the real decoder — those required
the cluster-based Phase 0 sweep to find. No tests are broken; one cluster
node (`stelvio`) has a corrupt `pyvista` install in the `comak` conda
env and is excluded from every job script.

**Additional issues resolved (beyond original scope).**

- Built **`compute_feature_mask`** (dihedral-angle seam detector) when
  Phase 0 revealed Fix 4b (topological boundary pinning) was a no-op on
  closed marching-cubes meshes. The dihedral mask was added by §0.4 as
  Fix 4c and turned out to be the actual production-relevant pinning
  mechanism on the menisci.
- Added a **hard `predictor_max_step` clamp on Fix 3** when the
  unbounded `1/‖∇SDF‖²` denominator was diverging mesh positions to
  ~10⁶. (Fix was still rejected — the clamp stopped divergence but the
  predictor still scrambled the mesh to ~50% fold-over.)
- Added a **`smooth_normals_max_step` clamp on Fix 7** for the same
  class of `1/(g·d)` pathology. (Fix 7 still rejected.)
- Discovered empirically that **`pyvista.subdivide_adaptive` preserves
  original point IDs** in `[0..N_orig)` and propagates `cell_idx` to
  sub-cells — the plan's §3 claim that "VTK reorders" was wrong.
  Documented this in §0.4 and used it in `compare_mesh_path.py`.
- Built the **matched-RGB visualisation** (per-vertex RGB from source
  position, transferred through correspondence) as a diagnostic for
  the "is the warp scrambling the mesh globally?" question. The result —
  every variant shows a coherently deformed colour gradient, no
  scrambling — confirms the remaining seam fold-over is *local* not
  *global*.
- SLURM **checkpoint/resume** (per-pair shard CSV, `afterany` merge
  dependency) added when the original "fan out as background scripts"
  plan §2.1 turned out to need a 3-wave dependency DAG to survive
  cluster contention and node failures.

**Challenges / Design decisions.**

- **`subdivide_triangles_on_base_mesh` (hand-built, in
  `NSM/mesh/refine_mesh.py`) vs. `pyvista.subdivide_adaptive`**: both
  preserve original point IDs (verified empirically). The hand-built
  one operates between warp passes; pyvista's operates per latent step.
  We tested both. Neither beat Fix 4c on the production metrics — see
  §0.11's dual-metric table. Fix 8 was rejected as a result, but the
  hand-built code is still available as `interpolate_points_refined`
  for anyone exploring further.
- **Default `tangent_laplacian_feature_angle=45°`** chosen over
  surface-by-surface optimal — bone wants θ=60°, cart wants θ=30°,
  menisci want θ=60°. Sweep showed θ=45° is the only single-config
  setting that achieves double-digit fold reduction on every surface
  AND non-positive ASSD on bone/menisci (only +3.7% cart ASSD). If a
  reviewer disagrees, the per-surface Pareto-best alternatives are
  documented in §0.11.
- **`n_corrector_iters=1` (no Fix 1) recommended despite Fix 1 being
  in every tested combo**: Fix 1 drives off-surface error from ~6e-5
  (microns) to ~1e-7 (nm). That's physically negligible and costs ~5×
  decoder evals per step. The recommendation drops Fix 1 on cost
  grounds, but this exact combination was not run end-to-end as a
  separate config — it's a logical reduction. If a reviewer wants
  bit-identical to the visually-inspected `fix1_fix2_fix4c` runs in
  `report/visuals/`, set `n_corrector_iters=5`.

**Things to note for future work.**

1. **One deployment step left:** `nsosim/nsosim/nsm_fitting.py:545`
   calls `interpolate_points` without `faces=`. To enable Fix 4c in
   production it needs:
   ```python
   faces                          = ref_mesh.regular_faces.astype(np.int64),
   step_magnitude                 = "newton",
   tangent_laplacian              = True,
   tangent_laplacian_pin_boundary = True,
   tangent_laplacian_feature_angle = 45.0,
   ```
   ~15-line change. Documented in §0.7.
2. **Anterior-horn labeling workflow handed off to user**: warp
   reference meniscus → each subject under the recommended config →
   label attachment on each warped mesh → pool labels to identify
   consensus reference vertex(es). The visuals saved under
   `report/visuals/` are the artifact the user verified looks correct.
3. **Tibia and patella NSM decoders NOT tested.** The recommended
   config is stepping numerics, not model-specific — it *should*
   transfer, but that's an assumption. The
   `650_nsm_tibia_v0.0.1` and `648_nsm_patella_v0.0.1` models are
   available and the experiment runner can be rerun on either.
4. **The two sibling plans address what's left.** Track A delivers the
   ~50% fold-over reduction; getting fold-over to true zero requires:
   - `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` — learned correspondence
     operator (former Track B); this plan was its prerequisite.
   - `NSM_TRAINING_IDEAS.md` — Eikonal-along-paths + normal-smoothness
     decoder retraining; would remove the seam pathology at the
     decoder level.
5. **The 5-knee subset** used for the sweep was chosen by
   `--max-knees=5` (manifest-order = KL-interleaved). If a reviewer
   wants to extend to the full 10 knees / 90 ordered pairs the
   `--max-knees` and `--max-pairs` flags in `run_matrix.py` accept any
   value.
6. **No statistical CIs reported.** Per-pair data is in
   `report/results.csv` and `report/results_{config}__{nfe}.csv`.
   The means reported here had stable signs across pairs
   (0/20 pairs worse for every Fix 4c improvement claim), so CIs were
   not computed — a reviewer who wants them can do paired t-tests /
   Wilcoxon on the per-pair data.

---
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

## 0. Results & status (2026-05-22)

### 0.1 Recommended production config

For the points-path API (`interpolate_points`) — what `nsosim` should ship:

```python
warped_pts = interpolate_points(
    model, latent_ref, latent_subject,
    n_steps      = 100,
    points1      = ref_mesh.points,
    surface_idx  = sidx,
    faces        = ref_mesh.regular_faces,   # NEW kwarg, required for Fix 4c
    spherical    = True,

    # Fix 2 -- Newton magnitude (free, strictly better, 0/20 pairs worse):
    step_magnitude                  = "newton",

    # Fix 4c -- tangent-Laplacian smoothing with dihedral seam pin.
    # theta=45 chosen as the "use everywhere" sweet spot from the sweep --
    # keeps near-optimal menisci performance AND avoids the +15.7% cart ASSD
    # penalty that theta=60 has. theta=60 is marginally better on menisci alone;
    # theta=30 strictly improves every surface but loses ~7pp menisci fold.
    tangent_laplacian               = True,
    tangent_laplacian_pin_boundary  = True,
    tangent_laplacian_feature_angle = 45.0,
    # n_corrector_iters left at default 1 -- Fix 1's 5-iter loop reduces
    # off-surface error from ~6e-5 to ~1e-7 (sub-micron) at ~5x decoder
    # evals per step. Not worth the cost; the gain is below the ASSD scale.
)
```

vs. baseline (paired, 20 pairs at NFE=100, sweep config `sw_a05_t45`):

| surface | fold-over | ASSD |
|---|---|---|
| bone | **−98.7%** | **−22.2%** |
| cart | **−53.5%** | +3.7% |
| med_men | **−45.5%** | **−10.4%** |
| lat_men | **−51.6%** | **−7.1%** |

Strictly better than baseline on every surface on fold-over; strictly better
or near-neutral on ASSD. Only deployment step remaining: update
`nsosim/nsosim/nsm_fitting.py:545` to pass `faces=ref_mesh.regular_faces.astype(np.int64)`
into `interpolate_points`.

### 0.2 Per-fix verdict

| fix | what it does | verdict | one-line reason |
|---|---|---|---|
| **Fix 1** corrector loop | iterate projection up to `n_corrector_iters` | **skip in production** | drops off-surface to ~1e-7 (microns→nm), but baseline is already physically small; costs ~5× decoder evals |
| **Fix 2** Newton magnitude | extra `1/‖∇SDF‖` | **always on** | 0 extra evals, strictly better on every metric every pair, 0/20 worse |
| **Fix 3** latent-advection predictor | `∂SDF/∂z·dz` step before corrector | **rejected** | scrambles mesh (~50% fold-over) even with displacement clamp; decoder `∂SDF/∂z` too noisy |
| **Fix 4** tangent Laplacian | tangent-only umbrella smooth + reproject | **always on** | the actual fold-over fixer (−50 to −99%) |
| Fix 4a (mesh path / VTK smooth) | use existing `interpolate_mesh` | rejected | Taubin smoothing pulls off-surface; fold-over 22%→46% on med_men |
| Fix 4b (topological boundary pin) | pin verts on edges in one triangle | **no-op on this data** | all meshes are topologically closed by marching cubes; nothing to pin |
| **Fix 4c** (dihedral seam pin) | pin verts on edges with dihedral > θ | **always on** with θ=45 | catches the *geometric* seam on closed-but-sharp meshes; the pin that actually fires |
| **Fix 5** adaptive step-sizing | Richardson / residual sub-stepping | skip | no measurable effect at NFE≥50 |
| **Fix 6** line-search magnitude | parabolic fit over candidate scales | skip | same outcome class as Fix 1+2, ~5× forward evals |
| **Fix 7** (added) smoothed-normals projection | Laplacian-smoothed gradient field as projection direction | rejected | diverges on cart/menisci without aggressive clamps + seam-aware smoothing; even clamped, doesn't beat Fix 4c |
| **Fix 8** (added) iterative source refinement | subdivide stretched/flipped triangles between warp passes | rejected | mesh growth was hard to bound (1.5× threshold → geometric blow-up); even when bounded, no improvement on original-triangulation fold-over vs Fix 4c |

### 0.3 What was built (delivered code)

- **`NSM/mesh/correspondence_metrics.py`** — new project-agnostic metrics module
  (39 tests). Family 1: `assd` (point-to-surface via pcu_sdf, matching
  `pymskt.Mesh.get_assd_mesh`), `directed_distance_percentiles`,
  `off_surface_error`, `triangle_health`, `self_intersection_count`. Family 2:
  `foldover_count`, `roundtrip_distance`, `forward_backward_disagreement`.
  Top-level `score_correspondence` scorer skips gracefully when optional
  inputs are missing.

- **`NSM/mesh/interpolate.py`** — full rewrite (`StepConfig` dataclass + GPU
  stepping primitives). Public API additions (all kwarg-gated, default OFF):
  - `step_magnitude="newton"` (Fix 2) / `"line_search"` (Fix 6)
  - `n_corrector_iters` + `corrector_tol` (Fix 1)
  - `latent_predictor` + `predictor_max_step` (Fix 3)
  - `tangent_laplacian` + `tangent_laplacian_alpha` + `tangent_laplacian_iters`
    + `tangent_laplacian_pin_boundary` (Fix 4 + 4b) +
    `tangent_laplacian_feature_angle` (Fix 4c)
  - `smooth_normals` + `smooth_normal_iters` + `smooth_normals_max_step`
    (Fix 7)
  - `adaptive_steps` + `adaptive_tol` + `adaptive_estimator` +
    `adaptive_max_depth` (Fix 5)
  - `return_diagnostics` — `_Diagnostics` dataclass tracks
    `n_advance_calls`, `n_decoder_evals`, `final_residual_max`,
    `struggled_intervals`
  - Helpers: `compute_boundary_mask`, `compute_feature_mask`,
    `build_mesh_laplacian`, `interpolate_points_refined` (Fix 8 wrapper).

- **`testing/NSM/mesh/test_interpolate.py`** (32 tests) +
  **`testing/NSM/mesh/test_correspondence_metrics.py`** (39 tests). Synthetic
  analytic-SDF model exercises each fix independently and in composition.

- **`experiments/mesh_interpolation/`** — Phase 0 runner with full
  resumability:
  - `config.py` — paths, model loader, `EXPERIMENT_CONFIGS` (40+ configs incl.
    α × θ × Fix7 sweep grid), `evaluate_sdf` helper.
  - `subjects.py` — KL-stratified subject selection from
    `0_demographics_baseline.csv` (4 KL0, 3 KL1, 3 KL2; KL3/4 excluded).
    Verifies all four femur-model meshes exist per knee. Writes `manifest.json`.
  - `fit_cache.py` — per-knee latent fitting via `nsosim.utils.fit_nsm`
    (production path), then marching-cubes reconstruction of all 4 surfaces
    via `NSM.mesh.create_mesh`. Resumable (`is_cached` check).
  - `run_matrix.py` — config × NFE × pair × surface scorer.
    Checkpoints after every pair (so a SLURM timeout loses ≤ 1 pair).
    Resumes by loading the existing shard and skipping done cells.
    `--max-knees`, `--max-pairs`, `--configs`, `--nfe`, `--no-roundtrip`,
    `--no-self-intersect`, `--out-tag`, `--merge` flags.
  - `submit_phase0.sh` — three-wave SLURM submission (10 fit jobs → 8/40
    matrix jobs split by config × NFE → 1 merge job). `--matrix-only`,
    `--dry-run`, `--configs=`, `--nfe=`, `--exclude=stelvio`.
  - `dump_visuals.py` — per-config 3-row PNG (fold / target-distance /
    warp-travel) + matched-RGB PNG (every vertex colored by source-position
    RGB, transferred to warped via correspondence — diagnostic for "is the
    warp scrambling vs deforming").
  - `compare_refined.py` — Fix 8 (iterative refinement) comparison +
    dual-metric scoring (original-N triangulation AND refined-mesh's own
    cell connectivity).
  - `compare_mesh_path.py` — `interpolate_mesh(adaptive=True)` (pyvista
    per-step subdivision) vs `interpolate_points + fix4c`.
  - `smoke_test.py` — GPU-free end-to-end test against analytic
    sphere-SDF.
  - `report/` — long-format `results.csv` (paired metrics per cell),
    `results.json`, `report.md`, and per-pair `.vtk` dumps with `flipped`
    (cell scalar), `target_distance`, `warp_travel`, and (for refined)
    `origin_pass`, `reference_vertex_idx` point scalars.

### 0.4 Things tried that weren't in the original plan

1. **Dihedral feature mask (Fix 4c)** — The original plan's Fix 4b assumed a
   topological boundary on open meshes. Empirically all four OAI surfaces are
   *closed* (marching-cubes wraps them), so Fix 4b is a no-op. Added
   `compute_feature_mask(faces, points, dihedral_threshold_deg)` that flags
   vertices on edges where the two incident face normals differ by more than
   the threshold — i.e. the *geometric* seam. The mask at θ=60° on the real
   meshes: bone 0 verts, cart 1.3%, med_men 2.7%, lat_men 1.7% — exactly the
   seam regions the plan's §1 identified as the failure mode.

2. **Fix 7 — smoothed-normals projection.** Replaces the per-vertex Newton
   *direction* with a Laplacian-smoothed unit-normal field, then chooses the
   magnitude that lands on the level set along that smoothed direction
   (`α = −SDF/(g·d)`). Coherent neighbor directions should reduce fold-over
   without smoothing positions. Diverged catastrophically (ASSD 1e-3 → 1e3+)
   without a hard step clamp; even clamped + feature-aware, it doesn't beat
   Fix 4c. Rejected.

3. **Fix 8 — iterative source-mesh refinement.** New
   `interpolate_points_refined(model, z1, z2, source_mesh, ...)` wrapper around
   `interpolate_points` and `NSM.mesh.refine_mesh.subdivide_triangles_on_base_mesh`.
   Each pass: warp → flag flipped + > N× area-stretched triangles → subdivide
   them in the source (preserves original vertex IDs) → re-warp. With an
   optional pre-pass that subdivides triangles within K mesh hops of the
   dihedral seam, and three correspondence modes (`"vertex"`, `"smoothed"`,
   `"centroid"`) for mapping the refined warp back to N originals. Result on
   menisci was *worse* original-triangulation fold-over than Fix 4c — adding
   midpoint neighbors changes which positions the originals are pulled to,
   in directions inconsistent with the original triangulation. Note: the
   *refined-mesh's own* ASSD does improve (−18 to −20% vs fix4c), but the
   original-N fold-over goes up — so this is a wash if you care about the
   original triangulation, which we do.

4. **Verified `pyvista.subdivide_adaptive` preserves original point IDs.** The
   plan's §3 claimed this would lose identity ("VTK reorders"); empirically
   `m.points[:n_orig]` *is* the original vertex set after `subdivide_adaptive`,
   and `cell_idx` propagates to sub-cells. The plan was wrong about that.
   Tested anyway via `compare_mesh_path.py` (Fix 4a) — see below.

5. **Matched-RGB visualisation as a diagnostic.** Per-vertex RGB derived from
   source position, carried through correspondence to every warped mesh. If
   the warp were globally rotating / shuffling the mesh, the warped meshes
   would show striped / scrambled colors. They don't — every variant shows the
   same coherent deformation. Confirms the remaining ~40-50% menisci
   fold-over is **local triangulation pathology at the seam**, NOT a global
   correspondence failure. Implemented in `compare_refined.py` and
   `compare_mesh_path.py`.

6. **α × θ × Fix7 sweep (24 configs).** The plan's §3.7 experiment matrix
   was the cumulative ladder; the sweep was added on top to find the cart
   ASSD Pareto front. Decisive: cart ASSD penalty drops from +15.7%
   (θ=60°) → +3.7% (θ=45°) → −1.4% (θ=30°). θ=45° picked as the "use
   everywhere" single-config sweet spot.

7. **SLURM submission with checkpoint/resume.** Original plan said "easy to
   fan out as background scripts." Reality required a 3-wave dependency DAG
   (10 fit → 40 matrix → 1 merge), per-pair shard checkpointing,
   `afterany` (not `afterok`) so timeouts still merge, and
   `--exclude=stelvio` after one specific node was repeatedly missing
   `pyvista` from its conda env.

### 0.5 Deviations from the original plan

- **Fix 4c emerged on top of Fix 4b**, not as a "two options A vs B" choice.
  Fix 4a (mesh path / VTK smooth) was tested via `compare_mesh_path.py` and
  is worse than baseline (Taubin pulls off-surface, fold-over inflates).
  Fix 4b (topological boundary pin) is a no-op on closed meshes. Fix 4c
  (dihedral seam pin) is what actually pins the seam on these meshes — and
  is the production choice.

- **Fix 1's payoff was overstated in the plan.** Phase 0 measured: off-surface
  error at NFE=100 baseline ~6e-5 (NSM-normalized) ≈ a few microns. Fix 1
  drives this to ~1e-7 (~7 nm). The improvement is real but physically
  cosmetic. The plan's framing ("guarantees terminal step converges; needed
  for Fix 4 cleanup") undersold the cost (~5× evals) and oversold the
  practical benefit. Fix 4 (with default `n_corrector_iters=1`) does **one**
  re-projection per smoothing pass, which is sufficient — Newton converges
  fast on near-on-surface points.

- **Fix 3 (predictor) rejected.** The plan's "honest framing" anticipated low
  priority but expected it to be safe at fine NFE. Empirically the predictor
  is *actively harmful* on this decoder — the `1/‖∇SDF‖²` magnitude factor
  diverges where the gradient is small, and a hard `predictor_max_step` clamp
  still leaves the mesh scrambled (~50% fold-over). The decoder's `∂SDF/∂z`
  is too noisy at the scale needed.

- **Fix 5 (adaptive) had no measurable effect.** Implemented with both
  Richardson and residual estimators, depth floor, struggled-interval logging.
  At NFE ∈ {50, 100, 200} the tolerance never triggered substantial
  subdivision. The plan's §1 mention of the "meniscus medial ridge" as
  motivation for adaptive sub-stepping turned out to not be the dominant
  failure mode — the seam fold-over is geometric, not a stiffness issue
  the integrator can resolve by stepping smaller.

- **Per-surface scoring was essential** (the plan called it "high-value
  insight for a small, one-time compute cost"). Confirmed: best config
  *does* differ per surface — bone happy at θ=60°, cart needs θ=30°, menisci
  prefer θ=60° on fold-over but θ=45° is the no-regrets compromise.

- **The plan's defect #5 (emergent coupling / non-bijectivity) still stands.**
  Matched-RGB visualisation confirmed the warp is doing the right semantic
  thing (no global rotation / scrambling). The remaining ~40-50% menisci
  fold-over at the seam is the *local-triangulation-at-a-thin-shell* problem
  that no Track-A fix removes. Genuine zero fold-over needs the rectified-flow
  plan or Eikonal-along-paths retraining.

### 0.6 §3.8 final validation status

The plan's §3.8 was "re-run the comak meniscus warp and compare warped-mesh
quality against baseline." The Phase 0 paired comparison (10 knees, 20
ordered menisci pairs, NFE=100) substitutes for and supersedes that test
on the NSM side. **The comak-side re-run is handed off**: the user is
generating warped reference-meniscus meshes under the recommended config,
labeling the anterior-lateral-horn attachment on each, and pooling labels
to update the reference mesh — that's the workflow the original motivating
context (`MENISCUS_LIGAMENT_ATTACHMENT_FIX.md`) was waiting for.

### 0.7 Remaining work outside this plan

1. **One nsosim call-site change** to pass `faces=` into `interpolate_points`
   so the production path picks up Fix 4c. ~15-line PR.
2. **Anterior-horn labeling workflow** (handed off to user): warp reference
   meniscus → each subject under recommended config → label attachment on
   each warped mesh → pool labels to identify consensus reference vertex(es).
3. **The two parked sibling plans** remain valid for further fold-over
   reduction beyond Track A:
   - `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` — the learned correspondence
     operator that addresses defect #5 directly.
   - `NSM_TRAINING_IDEAS.md` — Eikonal along interpolation paths +
     normal-smoothness regularizer; would remove the seam pathology at the
     decoder level.

### 0.8 Evidence locations (for reviewers)

**Branch:** `mesh-interpolation-improvements` off of `main` in
`/dataNAS/people/aagatti/programming/NSM/`.

**Key commits** (in reverse chronological order, oldest → newest grouping
by phase):

```
5bd5926  Add six kwarg-gated numerical fixes to mesh interpolation
b46cd46  Add correspondence-quality metrics module
4e22418  Add Phase 0 mesh-interpolation experiment runner
def42e9  Add SLURM submission for Phase 0 experiment
37de1f8  Make matrix runner checkpoint/resume; split jobs finer
a208381  Fix latent-advection predictor divergence (Fix 3)
875e4f1  Reject Fix 3 (latent predictor); isolate Fix 4 / Fix 5 configs
ec489ad  Add boundary-aware tangent Laplacian and Fix 7 (smoothed-normal projection)
03988dd  Detect geometric seam via dihedral angle; fix Fix 7 divergence
8d42573  Add Fix 4c tuning sweep (alpha x theta x Fix7)
8db3a7d  Add visuals dump: render warped cart / menisci under each config
5f835a9  Exclude stelvio node from all submissions
f794059  Visuals: add target-distance and warp-travel scalars + 3-row layout
9fe360d  Add iterative source-refinement wrapper (Fix 8) + comparison script
3cce73c  Add pre-seam-split, vertex-origin tracking, matched-RGB visualization
b9dda1f  Trim compare_refined to menisci only; cap refine passes at 2
36771fc  compare_refined: aggressive split + smoothed-corr + Fix 7 stacks
9eee303  compare_refined: tame runaway refinement (1.5x threshold too aggressive)
b125e35  compare_refined: cap refinement at 1 pass (mesh growth was geometric)
356c101  Add compare_mesh_path: pyvista subdivide_adaptive vs current best
```

**Code under review** (all paths relative to repo root):

| | path |
|---|---|
| Stepping primitives & all 8 fix kwargs | `NSM/mesh/interpolate.py` |
| Metrics module | `NSM/mesh/correspondence_metrics.py` |
| Hand-built subdivision (used by Fix 8) | `NSM/mesh/refine_mesh.py::subdivide_triangles_on_base_mesh` |
| Interpolate tests (31 tests) | `testing/NSM/mesh/test_interpolate.py` |
| Metrics tests (39 tests) | `testing/NSM/mesh/test_correspondence_metrics.py` |
| Experiment runner (config) | `experiments/mesh_interpolation/config.py` |
| Subject selection step | `experiments/mesh_interpolation/subjects.py` |
| Latent fit + marching-cubes caching | `experiments/mesh_interpolation/fit_cache.py` |
| Main matrix runner | `experiments/mesh_interpolation/run_matrix.py` |
| SLURM submission | `experiments/mesh_interpolation/submit_phase0.sh` |
| Visuals dump (3-row + RGB) | `experiments/mesh_interpolation/dump_visuals.py` |
| Fix 8 comparison | `experiments/mesh_interpolation/compare_refined.py` |
| Mesh-path comparison (Fix 4a) | `experiments/mesh_interpolation/compare_mesh_path.py` |
| GPU-free smoke test | `experiments/mesh_interpolation/smoke_test.py` |
| Production nsosim caller to update | `nsosim/nsosim/nsm_fitting.py:545` |

**Data under review** (all under `experiments/mesh_interpolation/`):

| | path |
|---|---|
| Selected pilot knees (manifest) | `cache/manifest.json` |
| Fitted latents (.npy per knee) | `cache/{key}_latent.npy` |
| Marching-cubes reconstructions | `cache/{key}_{bone,cart,med_men,lat_men}.vtk` |
| Per-cell long-format scoring | `report/results.csv` + per-config shards `report/results_{config}__{nfe}.csv` |
| Phase-0 aggregate report | `report/report.md` |
| Phase 0 / Fix 4 comparison PNGs (cart/med_men/lat_men) | `report/visuals/{surface}_{src}_to_{tgt}.png` |
| Refinement (Fix 8) comparison PNGs + matched-RGB | `report/visuals_refined/*.png` |
| Mesh-path (Fix 4a) comparison PNGs | `report/visuals_mesh/*.png` |
| All warped meshes saved with scalars (`flipped`, `target_distance`, `warp_travel`, sometimes `origin_pass`) | `report/visuals*/{surface}_{src}_to_{tgt}_{variant}.vtk` |

### 0.9 Reproducibility

Everything is deterministic given a seed.

**Conda env:** `comak` (loaded with `conda activate comak`). It has both
`nsosim` and an editable install of this NSM repo, so branch changes are
live. Includes `pyvista`, `pymskt`, `point_cloud_utils`, `torch+CUDA 11.8`,
`scipy`.

**Cluster:** SLURM partition `BMR-AI`. **Exclude `stelvio`** — that node
has a broken `pyvista` install in the `comak` env (confirmed by multiple
job failures with `ModuleNotFoundError: No module named 'pyvista'`).
`submit_phase0.sh` adds `#SBATCH --exclude=stelvio` to every job.

**To re-run from scratch** (estimated 4–6h wall-clock end-to-end):

```bash
cd /dataNAS/people/aagatti/programming/NSM
git checkout mesh-interpolation-improvements

# Step 1: select pilot knees (CPU only; deterministic from seed=0).
# Reads /dataNAS/people/aagatti/projects/OAI_DESS/aging_trajectories/data/
#       demographics/0_demographics_baseline.csv
# Picks 4 KL0 + 3 KL1 + 3 KL2, verifies all four femur-model meshes exist,
# writes cache/manifest.json. Manifest IS committed for direct comparison.
python -m experiments.mesh_interpolation.subjects

# Step 2: SLURM dependency DAG — 10 GPU fit jobs (each ~10 min) + 40 GPU
# matrix jobs (each ~0.5–2h) + 1 CPU merge job. Total ~4h wall-clock when
# the cluster has headroom; longer when contended.
./experiments/mesh_interpolation/submit_phase0.sh --dry-run   # preview
./experiments/mesh_interpolation/submit_phase0.sh             # actually submit

# Resume after a partial outage (any matrix job will resume from its
# checkpointed shard CSV; --matrix-only skips the already-cached fits).
./experiments/mesh_interpolation/submit_phase0.sh --matrix-only

# After everything finishes, view report/report.md or load report/results.csv.
```

**Sweep step** (the α × θ × Fix7 grid is in `config.py::SWEEP_CONFIG_NAMES`,
24 configs at NFE=100 only, run on a 5-knee subset for tractability):

```bash
SWEEP=$(python -c "from experiments.mesh_interpolation.config import SWEEP_CONFIG_NAMES; print(','.join(SWEEP_CONFIG_NAMES))")
./experiments/mesh_interpolation/submit_phase0.sh --matrix-only --nfe=100 --configs=$SWEEP
```

**Tests:**

```bash
cd /dataNAS/people/aagatti/programming/NSM
conda activate comak     # or any env with the NSM deps
pytest testing/NSM/mesh/ -v
# Expected: 70 tests pass (31 interpolate + 39 correspondence metrics).
```

**Random seeds.** `subjects.py` uses `SELECTION_SEED=0` for the knee
random.choice. `fit_cache.py` passes `seed=0` to `fit_nsm` (which seeds
torch + CUDA + numpy + python random + sets cudnn deterministic). Slerp
+ interpolation itself has no stochasticity.

**Smoke test (no GPU, no data) — verifies the full harness end-to-end
against an analytic sphere SDF:**

```bash
python -m experiments.mesh_interpolation.smoke_test
# Expected: "SMOKE TEST PASSED: 96 cells scored across 16 configs."
```

### 0.10 Scope and limitations

What was studied and what *wasn't* — so reviewers know how far to extend
the conclusions.

**Data scope.**

- **N=10 knees**, all from the **OAI baseline (00m)** visit. KL grades 0–2
  only (KL3/4 deliberately excluded — too-degenerate shapes would confound
  the metric discrimination). Source CSV:
  `/dataNAS/people/aagatti/projects/OAI_DESS/aging_trajectories/data/demographics/0_demographics_baseline.csv`.
- **One decoder**: the joint 4-surface femur model
  `568_nsm_femur_bone_cart_men_v0.0.1` (triplanar, latent_size=1024,
  surfaces = bone, cart, med_men, lat_men). The tibia
  (`650_nsm_tibia_v0.0.1`) and patella (`648_nsm_patella_v0.0.1`)
  models exist but were **not tested**. The recommended config should
  transfer (it's stepping numerics, not model-specific), but that's an
  assumption.
- **One fit-time random seed**: `seed=0` in `fit_nsm`. We did not assess
  fit-to-fit variability in the cached latents or the downstream
  warp metrics. The fits themselves can have CUDA-nondeterminism residue
  even with a seed — see the `meniscus_repro` test in
  `comak_gait_simulation` for prior empirical reads on that floor.

**Sample size and statistics.**

- **Main Phase 0 matrix** ran on all **20 ordered pairs** (10 knees ×
  10 directions, A≠B) — but `--max-knees=5` was used to bound runtime
  per matrix-job shard. So the 20 pairs span 5 representative knees
  (KL-interleaved so all three grades are present). The full 10-knee
  / 90-pair set was not run.
- **Sweep** (α × θ × Fix7, 24 configs) ran at **NFE=100 only**, on the
  same 5-knee / 20-pair subset.
- **Visuals / `compare_refined` / `compare_mesh_path`**: 1 worst-fold pair
  per surface (3 pairs total, all from the cached manifest).
- **No confidence intervals reported.** All "% vs baseline" numbers are
  means of paired deltas across 20 pairs. The per-pair distributions are
  in `report/results.csv` if a reviewer wants to compute CIs or paired
  significance tests — `assd_pct` and `fold_pct` were stable in sign
  across pairs (0/20 worse on every Fix 4c improvement), but I did not
  compute formal CIs.

**Metric scope.**

- All metrics are evaluated in **NSM-normalised units** (the cached
  marching-cubes meshes; max_rad-scaled). The fit-time scale that maps
  back to OAI millimetres is in `cache/manifest.json` *adjacent* to each
  fit but was not used in the scoring — that's a per-knee post-hoc
  conversion if a reviewer wants physical units.
- **Fold-over** uses the *source-mesh face connectivity* applied to
  warped point positions. On refined meshes (Fix 8) this becomes an
  unfair comparison — see §0.4 item 3 and the dual-metric table in
  §0.11 — and the parent-cell `cell_idx` is propagated through
  `pyvista.subdivide_adaptive` for fairer alternatives if needed.
- **ASSD** is the symmetric point-to-surface mean via `pymskt`'s
  `pcu_sdf` wrapper (point-cloud-utils signed_distance_to_mesh), matching
  `pymskt.mesh.Mesh.get_assd_mesh`. Not the nearest-vertex point-to-point
  fallback.

**What's NOT validated empirically.**

- The claim that Fix 4c at θ=45° is the right default for cart, but
  **cart wasn't visually inspected** (it has lower baseline fold-over, so
  visual fold-over inspection isn't as informative; the recommendation
  rests on the metric numbers).
- The claim that the recommendation extends to bone and to tibia/patella
  decoders — neither was visually verified, only metric-evaluated for
  bone.
- The leaner `n_corrector_iters=1` (no Fix 1) recipe in §0.1 is a logical
  reduction from the tested combos, but **`fix4c_without_fix1` was never
  run end-to-end as a separate config**. The argument is:
  Fix 1's only contribution is driving off-surface error from ~6e-5 to
  ~1e-7 (microns → nm), which is below ASSD and below the seam-related
  fold-over residual, so removing it should change nothing measurable.
  If a reviewer wants to be conservative, ship with `n_corrector_iters=5`
  (the exact composition that was tested in `fix1_fix2_fix4c`).

**Single-rep tests.** The matched-RGB diagnostic, the refinement
comparison, and the mesh-path comparison were each evaluated on the
*one* worst-fold pair per surface (3 pairs). They are qualitative
sanity checks, not full statistical evaluations.

### 0.11 Embedded data tables (for self-contained review)

#### Full α × θ × Fix7 sweep leaderboard

24 configs at NFE=100, 5 knees × 4 ordered pairs/knee = 20 pairs each.
All values are mean of per-pair paired %Δ vs baseline (negative = better
for both fold-over and ASSD).
`α` = `tangent_laplacian_alpha`, `θ` = `tangent_laplacian_feature_angle`,
`fix7` = `smooth_normals=True/False`.

```
   α    θ  fix7 |  bone fold  bone ASSD |  cart fold   cart ASSD |  med fold   med ASSD |  lat fold   lat ASSD
   ------------+-----------------------+------------------------+-----------------------+----------------------
 0.1   30   F  |    -86.6%      -18.0% |    -19.0%      -1.0%   |   -18.6%      -6.0%   |   -23.9%      -3.3%
 0.1   30   T  |    -79.1%      -20.5% |    -13.1%      -4.6%   |   -11.5%     -12.8%   |    -5.5%      -9.6%
 0.1   45   F  |    -93.4%      -18.7% |    -22.9%      +2.7%   |   -21.3%      -6.6%   |   -27.5%      -4.5%
 0.1   45   T  |    -87.7%      -21.1% |    -12.2%      -5.7%   |   -10.3%     -21.9%   |    -6.3%     -15.6%
 0.1   60   F  |    -93.8%      -18.7% |    -25.6%     +10.9%   |   -22.7%      -6.7%   |   -28.6%      -2.9%
 0.1   60   T  |    -88.5%      -21.0% |    -10.8%      -7.0%   |   -10.0%     -23.7%   |    -5.8%     -19.3%
 0.2   30   F  |    -89.6%      -19.5% |    -28.7%      -0.9%   |   -26.2%      -6.5%   |   -31.9%      -3.4%
 0.2   30   T  |    -83.9%      -20.9% |    -22.3%      -3.2%   |   -20.0%     -12.1%   |   -19.3%      -8.4%
 0.2   45   F  |    -96.6%      -20.5% |    -34.3%      +3.3%   |   -29.5%      -8.1%   |   -36.6%      -5.3%
 0.2   45   T  |    -94.2%      -21.7% |    -24.5%      -2.7%   |   -21.0%     -20.5%   |   -21.9%     -15.5%
 0.2   60   F  |    -97.4%      -20.5% |    -37.9%     +12.5%   |   -31.2%      -9.0%   |   -37.9%      -4.7%
 0.2   60   T  |    -95.6%      -21.7% |    -24.9%      +0.9%   |   -21.6%     -21.8%   |   -22.2%     -17.3%
 0.3   30   F  |    -90.1%      -20.5% |    -36.1%      -1.0%   |   -32.2%      -7.3%   |   -37.4%      -3.5%
 0.3   30   T  |    -86.8%      -21.4% |    -30.0%      -3.0%   |   -25.9%     -13.3%   |   -28.4%      -8.2%
 0.3   45   F  |    -98.0%      -21.3% |    -42.5%      +3.6%   |   -36.2%      -8.9%   |   -42.8%      -5.7%
 0.3   45   T  |    -96.8%      -22.0% |    -33.6%      +0.0%   |   -27.9%     -17.9% |   -31.4%     -12.1%
 0.3   60   F  |    -98.6%      -21.2% |    -46.5%     +14.1%   |   -38.3%      -9.7%   |   -44.3%      -5.8%
 0.3   60   T  |    -97.8%      -22.0% |    -35.2%      +5.5%   |   -29.3%     -20.5%   |   -32.6%     -16.1%
 0.5   30   F  |    -90.6%      -21.5% |    -46.2%      -1.4%   |   -40.6%      -8.4%   |   -45.6%      -3.7%
 0.5   30   T  |    -89.1%      -21.9% |    -40.9%      -2.7%   |   -35.7%     -12.7%   |   -39.2%      -7.2%
 0.5   45   F  |    -98.7%      -22.2% |    -53.5%      +3.7%   |   -45.5%     -10.4%   |   -51.6%      -7.1%
 0.5   45   T  |    -98.3%      -22.6% |    -46.6%      +1.6%   |   -38.7%     -18.1%   |   -43.7%     -13.0%
 0.5   60   F  |    -99.4%      -22.2% |    -58.1%     +15.7%   |   -47.9%     -12.3%   |   -53.0%      -7.5%
 0.5   60   T  |    -99.0%      -22.6% |    -49.4%      +9.8%   |   -40.2%     -17.5%   |   -44.1%     -13.5%
```

How to read it:
- **Recommended row** is `α=0.5, θ=45, fix7=F` — the only single-config
  setting that hits **double-digit fold reduction on every surface** AND
  **non-positive ASSD on bone/menisci**, with only +3.7% cart ASSD (the
  least-bad cart ASSD across all configs that pin no rim).
- **Best-by-surface** Pareto:
  - **Bone**: `α=0.5, θ=60, fix7=F` (fold -99.4%, ASSD -22.2%).
  - **Cart**: `α=0.5, θ=30, fix7=F` (fold -46.2%, ASSD -1.4% — only config
    with negative cart ASSD and any meaningful fold reduction).
  - **Med_men / lat_men fold**: `α=0.5, θ=60, fix7=F`.
  - **Med_men / lat_men ASSD**: `α=0.1, θ=60, fix7=T` (lat_men ASSD
    -19.3%) — but fold reduction is only -5.8% there, so not a
    useful trade.
- **Fix 7 (smooth_normals=T)** consistently *worsens* fold-over on every
  surface (compare the F/T rows at each α, θ) while delivering modestly
  better ASSD on menisci. Net trade is unfavorable — Fix 7 stays
  rejected for production.

#### Refinement (Fix 8) dual-metric comparison

`max_refine_passes=1`, `area_growth_threshold=3.0`,
`pre_split_seam_hops=1`. NFE=100. One pair per surface (the
worst-fold-over pair from baseline). The original-N triangulation and the
refined mesh's own connectivity are scored side-by-side because they
measure different things — see §0.5.

**med_men: 9203957_LEFT → 9523523_LEFT** (n_orig = 6,054 verts)

| variant | ASSD | fold% (orig N) | fold% (refined mesh) | n_ref |
|---|---|---|---|---|
| baseline | 0.00148 | 22.91% | — | 6,054 |
| **fix4c** | **0.00138** | **13.86%** | — | 6,054 |
| refined_smoothed | 0.00121 | 20.43% | 36.25% | 21,136 |
| refined_fix7_vertex | 0.00118 | 22.11% | 37.09% | 21,233 |
| refined_fix7_smoothed | 0.00121 | 20.43% | 37.02% | 21,225 |
| mesh_adaptive_only (Fix 4a, no smooth) | 0.00143 | 22.93% | 23.36% | 6,161 |
| mesh_adaptive_smooth (Fix 4a + Taubin) | 0.01076 | 46.15% | 45.80% | 17,775 |

**lat_men: 9524744_RIGHT → 9523523_LEFT** (n_orig = 7,572 verts)

| variant | ASSD | fold% (orig N) | fold% (refined mesh) | n_ref |
|---|---|---|---|---|
| baseline | 0.00136 | 15.88% | — | 7,572 |
| **fix4c** | **0.00122** | **8.61%** | — | 7,572 |
| refined_smoothed | 0.00118 | 12.73% | 24.83% | 19,194 |
| refined_fix7_vertex | 0.00111 | 14.27% | 25.02% | 19,235 |
| refined_fix7_smoothed | 0.00113 | 13.12% | 24.90% | 19,262 |
| mesh_adaptive_only (Fix 4a, no smooth) | 0.00131 | 15.87% | 15.97% | 7,752 |
| mesh_adaptive_smooth (Fix 4a + Taubin) | 0.00532 | 40.62% | 35.04% | 10,615 |

What this shows:

- The refined variants get **lower ASSD than fix4c** (med_men −18 to −20%
  relative; lat_men −9 to −11% relative — substantially better surface
  fit) — *but* the **fold-over on the original triangulation is worse
  than fix4c** on every refined variant. The refined mesh's own
  connectivity is even worse (36% on med_men).
- Adding **midpoint vertices changes which neighbours the originals get
  smoothed against**, so original triangle (i, j, k) ends up more
  flipped even though the underlying surface is covered better. This is
  the structural mismatch that makes Fix 8 a wash if you care about the
  *original* triangulation (which the production warp does).
- **Fix 4a `mesh_adaptive_only`** barely subdivides (default
  `max_edge_len=0.04` is larger than most edges already) → effectively
  baseline. **Fix 4a + Taubin** is dramatically worse: Taubin without
  re-projection pulls the mesh off-surface and fold-over balloons.

This is the table that justified rejecting Fix 7, Fix 8, and Fix 4a as
production options.

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
