# Plan: A learned rectified-flow correspondence operator for NSM

## State

**Updated:** 2026-08-17 · **Status:** blocked

- **Next:** nothing until the interpolation API is stable. Re-read this after
  `NSM_CODE_HEALTH_REFACTOR.md` Phase 4 §8.1 (the decoder registry) lands.
- **Blocked on:** a stable interpolation API, and `NSM_CODE_HEALTH_REFACTOR.md` Phase 2.
- **Done:** nothing. The prerequisite plan
  (`completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md`) delivered the
  `correspondence_metrics.py` module this plan depends on.
- **Surprises:** none yet — no work has been done against this plan.

---

**Status:** Proposed — not started. Prerequisite
`NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` **completed 2026-05-22** (archived to
`completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md`). Track-A
delivered ~50% menisci fold-over reduction and shipped the
`correspondence_metrics.py` module this plan depends on; the remaining seam
fold-over is what this plan would address.
**Created:** 2026-05-18 (split out of the original combined plan).
**Repo:** `/dataNAS/people/aagatti/programming/NSM/` (NSM). Sole production
consumer: `nsosim` → the comak gait pipeline.
**Goal:** Replace the emergent, per-vertex SDF-stepping correspondence with a
single **learned global velocity network**, conditioned on both endpoint
latents, that transports surface-A points onto surface-B. This fixes the one
thing the numerical fixes cannot: the **emergent, non-bijective coupling**
(defect 5 of the sibling plan) — collapsing, crossing, off-surface drift with
no guarantee of a clean map — and adds amortized fast inference plus a Jacobian
topology/osteophyte signal.

> **Why this is a separate plan.** This is a weeks-scale GPU training + research
> effort. The cheap numerical fixes in `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md`
> are likely ~80% of the value for the meniscus warp and are days of work. Do
> that plan + its Phase 0 metrics first, re-measure, and only fund this if
> (a) the numerical fixes are insufficient, or (b) the amortization / Jacobian
> osteophyte detector is independently wanted. This work subsumes the numerical
> plan's improved stepping as its data-generation seed, so neither effort is
> wasted.

---

## 1. What the numerical fixes do *not* solve

The numerical plan integrates a sequence of closest-point projections. Every
step moves points purely along the surface normal; the **tangential** degree of
freedom — *which* B-point an A-vertex lands on — is left at zero. Consequences:

- The coupling is **emergent, not chosen.** Nothing constrains trajectories to
  be non-crossing or the endpoint map to be bijective. Two A-vertices can
  collapse onto one B-point.
- At an **osteophyte** (A has one, B does not) no continuous bijection exists;
  the method collapses many A-vertices onto one B-point — pathological
  (duplicate vertices, inverted triangles, NaN biomarkers).

A learned velocity field gets to **choose the full step** — including the
tangential component — so it can produce a deterministic, non-crossing map and
expose where the map is degenerate. cf. Helbling, *A Visual Introduction to
Rectified Flows* (Jan 2026), https://alechelbling.com/blog/rectified-flow/ —
the reflow procedure adapted here. The key adaptation: we *observe both
endpoints*, so we condition on `(z_A, z_B)` rather than solving the harder
one-sided generative problem.

| | Numerical fixes (sibling plan) | This plan — rectified-flow operator |
|---|---|---|
| Effort | ~days, no training | ~weeks, GPU training + research |
| Touches | edits to `interpolate.py` | new `NSM/NSM/flow/` submodule |
| Fixes | drift, magnitude, convergence, mild regularization | + amortized fast inference, guaranteed non-crossing, Jacobian topology/osteophyte signal |
| Risk | low | medium (research) |

---

## 2. Prerequisite — Phase 0 metrics

This plan depends on the correspondence-quality metrics built in Phase 0 of
`NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` (`correspondence_metrics.py`:
off-surface error, bijectivity / round-trip, fold-over, forward–backward
disagreement, NFE sensitivity, plus surface-fit and triangle-health). The same
held-out latent pairs and the same menisci-first scoping apply. Do not start
this plan until those metrics and the numerical-fix baseline exist.

---

## 3. B1 — Data generation (initial coupling π₀)

- Generate correspondence pairs `(x_0, x_1)` for many `(z_A, z_B)` pairs, seeded
  by the **numerically-improved** stepping from the sibling plan (a better π₀
  than the raw current method).
- **Initial coupling matters more than reflow count.** Optionally improve π₀
  further with a Sinkhorn / entropic-OT matching between the two point clouds
  with a geometric cost — the repo already has `NSM/dependencies/sinkhorn.py`.
  Reflow straightens; it cannot fix a wrong seed (osteophyte-vertex → wrong
  condyle stays wrong).
- Sample latent pairs widely across the latent distribution (oversample large
  `‖z_A−z_B‖` and topology-changing pairs). Cache pairs to disk.
- **Pilot scope: menisci first.** Generate for `surface_idx` 2/3 (femur model)
  before bone — menisci are the active need and have no topology change, a
  cleaner first validation of the whole pipeline. Then extend to bone.
- The improved stepping's Fix 3 (`∂SDF/∂z` predictor) lets each generated pair
  use far fewer steps; data-generation volume is exactly where that fix pays
  off, so enable it here.

## 4. B2 — Velocity network + conditional flow matching

- `v_θ(x, t, z_A, z_B, surface_idx)` → 3D velocity. New submodule
  `NSM/NSM/flow/` (sibling of `mesh/`); mirror `NSM/NSM/train/` conventions
  (config dict, `{experiment_directory}/model/{epoch}.pth` checkpoints).
- Loss: conditional flow matching on the linear path,
  `L = E_{t,x_0,x_1} ‖(x_1−x_0) − v_θ(x_t, t, z_A, z_B)‖²`,
  `x_t = (1−t)x_0 + t·x_1`. Conditioning on **both** latents is the point —
  it sidesteps the velocity-averaging that curves vanilla flow-matching paths.
- **On-surface regularizer** (important here, unlike generative FM): add
  `λ·|SDF(x_t, z(t))|²` (or a projection-consistency term) so trajectories do
  not drift off the evolving level set mid-path. `z(t) = slerp(z_A,z_B,t)`.
- Train globally over all cached pairs; one model serves any pair.

## 5. B3 — Reflow

- Integrate `v_θ` forward from surface-A points to produce a new, deterministic
  (hence **non-crossing**) coupling π₁; retrain `v_θ` on it; iterate K times
  (K small — 1–3; diminishing returns). Same global model throughout.

## 6. B4 — Bidirectional consistency + Jacobian (topology-mismatch detector)

- Run the flow A→B and B→A. Where the two disagree, or where the forward map's
  Jacobian determinant blows up / collapses, is where the surfaces are not
  homeomorphic — i.e. **osteophytes / erosions**.
- Expose the log-Jacobian as a per-vertex scalar field: an unsupervised,
  localized osteophyte/erosion map. Validate against MOAKS osteophyte scores on
  a labeled subset.

## 7. B5 — Integration

- Add a flow-based path to `interp_ref_to_subject_to_osim`
  (`nsosim/nsosim/nsm_fitting.py:545` is the sole production caller of
  `interpolate_points`). Gate behind a config flag; default off until validated;
  A/B against the SDF-stepping path with the Phase 0 metrics.

## 8. Validation

Phase 0 metrics, flow vs the numerical-fix baseline, on the held-out latent
pairs, menisci then bone separately. Re-run the comak meniscus warp. Success =
materially better bijectivity / off-surface error / fold-over with
equal-or-fewer NFEs.

---

## 9. Honest caveats (calibrate expectations)

- **Rectified flow does not preserve detail through a topology change.** It
  makes the osteophyte collapse *smooth and bijective* instead of degenerate,
  but the osteophyte still compresses onto a small patch of B (and expands
  back). That is the mathematically correct answer to an impossible question —
  and the Jacobian magnitude *is* the useful signal (B4). It is not "perfect
  correspondence."
- **The flow has no notion of anatomy.** It does geometric transport;
  geometric-closest ≠ anatomical-homolog when shapes differ pathologically
  (trochlear groove deep vs flat, remodeled bone, etc.). If anatomical
  correspondence is the goal, domain knowledge must be injected — see §10.
- **Intermediate `t∈(0,1)` states are not anatomically real.** Fine for
  visualization and population interpolation; do not treat them as physically
  meaningful intermediate-severity samples.
- **Initial coupling > reflow count.** A bad π₀ reflows into a *straight path to
  the wrong place*. Spend effort on B1's seed.
- **For whole-shape statistics, latents already suffice** — correspondences are
  needed for *localized* analyses (per-vertex statistics, region biomarkers,
  longitudinal local change, displacement-field SSMs, biomech meshing). Scope
  the effort to those use cases.

## 10. Open decisions / optional enhancements

1. **One global model vs per-surface.** Recommend one model conditioned on
   `surface_idx`; pilot menisci-only first (§3).
2. **Anatomical-correspondence injection** (if geometric transport proves
   insufficient): (a) **landmark loss** — pull a handful of known anatomical
   landmarks toward each other during training; (b) **spectral/feature
   conditioning** — condition `v_θ` on local geometric/spectral features of the
   source point, not just position (closer in spirit to FOCUSR-style SSM
   registration). Decide after the menisci pilot.
3. **Soft / one-to-many correspondence** — if a hard bijection is the wrong
   model for osteophytes, use entropic-OT soft correspondences; the spread mass
   itself measures "no clean match." Larger scope; defer.

## Code touchpoints

- `NSM/NSM/mesh/interpolate.py` — the numerically-improved stepping; the seed
  generator for π₀ (B1).
- `NSM/NSM/models/triplanar.py:166,330` — `TriplanarDecoder`; multi-surface SDF
  query for the on-surface regularizer (B2) and the Jacobian (B4).
- `NSM/NSM/train/train_deep_sdf.py`, `NSM/utils.py` — training + checkpoint
  conventions to mirror for the new `NSM/NSM/flow/` submodule.
- `NSM/dependencies/sinkhorn.py` — OT seeding for π₀ (B1).
- `NSM/NSM/mesh/correspondence_metrics.py` — Phase 0 metrics (built by the
  sibling plan); reused for validation here.
- `nsosim/nsosim/nsm_fitting.py:545` — sole production caller; B5 integration.
- Data: `COMAK_SIMULATION_REQUIREMENTS/nsm_models/568_nsm_femur_bone_cart_men_v0.0.1/`
  (`model/2000.pth`, `latent_codes/2000.pth` → 6239×1024 femur latents,
  `model_params_config.json` with `mesh_names=[bone,cart,med_men,lat_men]`).

## Related

- `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` — the numerical fixes + Phase 0
  metrics; **prerequisite** to this plan.
- `NSM_TRAINING_IDEAS.md` — upstream decoder-training levers.
- `comak_gait_simulation/.claude/plans/MENISCUS_LIGAMENT_ATTACHMENT_FIX.md` —
  the warp this work ultimately improves.
- Helbling, *A Visual Introduction to Rectified Flows* (Jan 2026),
  https://alechelbling.com/blog/rectified-flow/.
- ShapeMed-Knee paper: Gatti et al. 2024.
