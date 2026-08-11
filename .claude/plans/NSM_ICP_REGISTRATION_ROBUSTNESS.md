# Plan: ICP registration robustness (inference-only)

**Status:** Open — Phase 0 (diagnostics) and the Phase 2 tradeoff study (§7) complete;
Phases 1, 3 and the sampler decision not started. §7.4 was retracted after adversarial review.
**Created:** 2026-08-11.
**Repo:** `/home/gattia/programming/kneepipeline/DEPENDENCIES/nsm` (NSM).
**Branch:** `icp-registration-robustness` off `main` @ `bb2c6a3`.

> **Scope.** Inference-only. No retraining. Every change here alters the fitted latents and
> therefore invalidates any B-score normative constants derived under the current recipe — see
> §6. Deliberately **out of scope**: decoder training changes (`NSM_TRAINING_IDEAS.md`) and the
> Sim(3) joint pose+shape optimiser, which is deferred to §5.

---

## 1. Why — the diagnostic that motivated this

Full investigation and raw data:
`/home/gattia/programming/kneepipeline/analysis/bscore_variance_source_2026-08-11/`.

The femur B-score turned out to be almost insensitive to bone *geometry* and extremely sensitive
to the **vertex ordering** of the mesh handed to the NSM. On one fixed dense mesh, seed held at 42:

| perturbation | B-score SD |
|---|---|
| identical input twice | 0.00003 |
| ±1 µm jitter on **every** vertex | 0.00017 |
| 0.5° rigid rotation of the whole mesh | 0.00004 |
| NSM seed changed | 0.0078 |
| **vertex order permuted, geometry bit-identical** | **0.1210** (range 0.274) |

Freezing the ICP transform and repeating the permutation test removes **97–99.9%** of that
variance (SD 0.0668 → 0.0111 on one knee, 0.1166 → 0.0033 on another). The residual is exactly the
optimiser-init noise.

**Root cause.** `vtkIterativeClosestPointTransform` reduces the source to
`MaximumNumberOfLandmarks` by striding the point list, so the landmark set is a function of the
**array layout**, not of the surface. NSM passes `n_landmarks=1000` explicitly at both call sites
(`NSM/datasets/sdf_dataset.py:266` and `:542`) against meshes of ~100k vertices, and hardcodes
`max_n_iter=100`.

Transform spread across 5 vertex orderings of one unchanged mesh, registered to the real NSM mean:

| n_landmarks | scale spread | max rotation spread | max translation spread | s / registration |
|---|---|---|---|---|
| **1000 (current)** | CV 1.71e-3 | **0.72°** | **0.41 mm** | 4.0 |
| 5000 | CV 3.10e-4 | 0.46° | 0.35 mm | ~20 |
| 20000 | CV 1.41e-4 | 0.18° | 0.12 mm | 79.4 |
| ALL (113k) † | max−min 4.6e-13 | **0.0000°** | **0.0000 mm** | 392.2 |

† The first three rows are CVs over **5** vertex orderings (`icp_landmarks.py`); the ALL row is an
absolute max−min over a separate **4**-ordering run (`reg_compare2.py`) and is therefore not a CV.
It is included because it pins the mechanism — with every vertex used, the order-dependence is
float noise — not because it is commensurable with the rows above.

Two facts that shape the whole plan:

1. **ICP-1000 is essentially unbiased** — its mean answer sits within **0.021%** of the all-points
   answer. The defect is pure *variance* over which 1000 points get picked, not a systematic error.
   So the fix is a better/averaged landmark set, not a bigger one.
2. **Pose error is aimed at the readout.** The fraction of the induced latent perturbation landing
   on the B-score direction is **0.0741**, versus 0.0353 for isotropic 512-d noise and 0.0309–0.0332
   measured for seed / frozen-ICP noise. Pose error is ~2.1× more aligned with the B-score axis than
   chance *and* 4–6× larger, so ~8.5× worse on the readout. Likely mechanism: the similarity
   transform's **scale** DOF aliases onto bone size, and size is a large component of the
   OA-vs-control mean difference. This is why §4 (shared scale across timepoints) matters.

---

## 2. Phase 1 — expose the ICP parameters

Today none of the registration knobs are reachable from `reconstruct_mesh`. Thread them through:

```
reconstruct_mesh(...)
  -> read_mesh_get_sampled_pts / read_meshes_get_sampled_pts   (sdf_dataset.py:266, :542)
     -> Mesh.rigidly_register(...)                             (pymskt)
        -> vtkIterativeClosestPointTransform
```

Proposed parameters (names to be bikeshedded, semantics not):

| parameter | default | note |
|---|---|---|
| `icp_n_landmarks` | `1000` | unchanged default |
| `icp_max_n_iter` | `100` | currently hardcoded. **Measured (§7.3): 30 is as good or better at 1/3 the cost.** Expose now, change the default only as part of a recipe re-baseline |
| `icp_reg_mode` | `"similarity"` | unchanged |
| `icp_sampling` | `"vtk_stride"` | `vtk_stride` \| `canonical_stride` \| `bluenoise` |
| `icp_n_repeats` | `1` | ensemble size (§3) |
| `icp_seed` | `0` | seeds the landmark sampler, **never** passed as 0 to pcu (see §3) |
| `icp_return_spread` | `False` | return the ensemble spread as a QC metric |
| `icp_transform` | `None` | already exists in `read_meshes_get_sampled_pts`; expose on `reconstruct_mesh` so a precomputed transform can be supplied (needed for §4) |

**Hard requirement: the defaults must reproduce current behaviour bit-for-bit.** Changing a default
silently would invalidate every latent ever fitted with this library. Regression test in §6.

Deliverable: parameters plumbed, defaults unchanged, regression test green.

---

## 3. Phase 2 — better landmark sampling and ensembling

### 3.1 Sample the surface, not the vertex array

`vtk_stride` picks *vertices by index*. Two scans of the same knee produce two different vertex
arrays, so they get different landmark sets even after canonicalising the order. Sampling the
**surface** (Poisson-disk / blue noise) instead should make the landmark set a property of the
geometry. **This is an argument, not yet a measured result** — the grid in §7 was unable to test it
fairly (§7.4), so it remains open.

> **Bug to avoid — verified.** `pcu.sample_mesh_poisson_disk` documents `random_seed=0` as *"use
> the current time"*. `pymskt.mesh.meshTools.rand_sample_pts_mesh` (line 1440) does **not** pass a
> seed, so **pymskt's blue-noise sampling is non-deterministic as shipped**. Three calls on one
> unchanged mesh returned three different point sets *and* three different counts — 2858 / 2876 /
> 2846 for a request of 2000. Any use of it here must pass an explicit **non-zero** seed and must
> not assume the returned count equals the request. Otherwise this reintroduces the exact bug in
> §1 in a new place. Either fix upstream in pymskt or call `pcu` directly.

### 3.2 Ensemble K independent registrations

```python
def register_similarity_robust(source, target, n_landmarks, n_repeats, sampling, seed, max_n_iter):
    Ts = [icp(sample(source, n_landmarks, seed + k, sampling), target, max_n_iter)
          for k in range(n_repeats)]
    return average_similarity(Ts), spread(Ts)
```

- **Rotation averaging must be chordal**, not element-wise: mean the rotation matrices, project back
  to SO(3) via SVD, guard `det < 0`. Element-wise averaging does not produce a rotation.
- Average **log-scale**, not scale.
- **Check the spread before averaging.** If members converged to genuinely different minima the mean
  can be worse than any of them — that is a case to flag, not to average away.

Measured on the A/B re-processing pair (two pipeline jobs over the **same** MRI volume, 2 voxels
different — see the box in §7, and §7.4 for why this metric flatters the stride sampler):

| method | A-vs-B rotation | A-vs-B scale | s / mesh |
|---|---|---|---|
| single ICP @1000, canonical order | 0.0729° | 0.0218% | 3.6 |
| single ICP @20000, canonical order | 0.0078° | 0.0058% | 67.5 |
| **bagged 10 × 1000, canonical order** | **0.0024°** | **0.0000%** | **34.1** |

Bagging is 30× more stable than a single 1000-landmark ICP and 3× more stable than a single
20000-landmark ICP, at half the latter's runtime. The *ensembling* conclusion survives §7.4's
retraction — it is a within-sampler comparison, so the pairing asymmetry does not affect it.

### 3.3 The tradeoff study (complete — results in §7)

Free parameters trade against each other: **landmark count × ensemble size × max_n_iter × sampling
rule**, all against wall-clock. Iso-budget design — for each landmark count, run enough independent
ICPs to reach a 20,000-point total, then evaluate ensembles of size `k` from **disjoint** groups so
every cell gets an error bar.

- `n_pts` ∈ {250, 500, 1000, 2000, 4000, 8000}, max repeats {80, 40, 20, 10, 5, 2}
- `sampling` ∈ {`canonical_stride`, `bluenoise` (seeded)}
- `max_n_iter` ∈ {30, 100, 300} at `n_pts=1000`
- metrics: rescan disagreement (A vs B), order disagreement (A vs permuted A), deviation from the
  all-points reference, measured wall-clock

Harness: `scripts/icp_grid_run.py` + `scripts/icp_grid_analyze.py` in the kneepipeline analysis
folder. **Results: see §7.** Note the analyzer reports both *paired* and *unpaired* A-vs-B
disagreement — §7.4 explains why reporting only the paired number produced a wrong conclusion.

---

## 4. Phase 3 — group / longitudinal joint registration

**The idea.** Today every mesh is independently similarity-registered to the NSM mean, so **every
timepoint gets its own scale estimate**. Bone size does not change in an adult, so that per-timepoint
scale variation is pure error — and per §1.2 it lands preferentially on the B-score axis. A
longitudinal B-score change is therefore contaminated by independent per-timepoint scale noise.

**Proposal.** For a set of meshes that should share a scale (repeat scans of the same knee):

1. **Rigid** (6-DOF, *no* scale) register the group to a common reference. Registering a knee to
   *itself at another timepoint* is a far better-conditioned problem than registering it to a
   population mean — near-identical shape, so the correspondences are meaningful.
2. **Similarity** register the group reference to the NSM mean **once**, using the robust ensemble
   from §3.
3. **Compose** — apply that single similarity transform to every member.

**Consequences.**

- Scale is **identical across timepoints by construction**, which is both a variance reduction and
  biologically correct.
- Only the rigid mesh-to-mesh residual varies within subject, and that is small.
- The one similarity step can pool landmark ensembles across the whole group, making it more stable
  still.

**Design cautions.**

- **Use an unbiased group reference — but not textbook generalized Procrustes.** GPA is defined on
  *corresponded* landmark sets. Our group members are independently segmented marching-cubes
  surfaces with different vertex counts and **no correspondence**, so "the running group mean" is
  not a well-defined object — you cannot average vertex positions, and ICP needs a concrete target
  surface. Two workable substitutes: (a) register every member to every other member pairwise, then
  solve for the set of rigid transforms that best satisfies all pairwise constraints
  (rotation averaging / pose-graph optimisation) — unbiased and correspondence-free; or (b) pick a
  member as reference but *symmetrise*, and report sensitivity to that choice. Note that for the
  dominant case of **exactly 2 timepoints** any unbiased scheme degenerates to splitting the single
  pairwise transform in half, so the "group" machinery only earns its keep at T ≥ 3.
- **Real shape change breaks rigid correspondence** (osteophyte growth, post-surgical change). This
  wants a trimmed / robust rigid ICP — **which does not exist in the current stack**:
  `pymskt.mesh.meshRegistration.get_icp_transform` wraps `vtkIterativeClosestPointTransform` with
  no outlier rejection at all. Implementing trimmed ICP is therefore a real work item here, not a
  configuration change. Report the residual either way so bad cases surface.
- **Bilateral is a judgement call, not an obvious extension.** Left and right femurs of one subject
  are similar but genuinely not the same size; do not force shared scale across sides without
  evidence.
- Requires `icp_transform` to be settable on `reconstruct_mesh` (§2).

**Deliverable.** `register_group_to_mean(meshes, ...) -> list[transform]`, plus a kneepipeline
entry point that accepts multiple timepoints for one knee.

---

## 5. Deferred — Sim(3) joint pose + shape

`NSM/reconstruct/reconstruct_latent_S3.py` already implements DeepSDF × Sim(3)
(arXiv:2004.09048). It was tried previously and judged not to work. **It has a gradient-flow bug
that alone explains that**, so the method has not actually been given a fair test:

- `get_w()` (line 17–18) builds the skew matrix as `torch.Tensor([[0, -w3, w2], ...])` from tensor
  elements. That constructs a **new** tensor from Python floats and severs the graph. Verified:
  `get_w(...).requires_grad = False`; after backward, `polar_angle.grad is None` while
  `theta.grad = -0.0816`. **The rotation axis receives no gradient at all.**
- The axis is therefore frozen at its init. With `polar, azimuthal ~ N(0, 0.01)` that is
  ≈ **(0, 0, 1)** (measured: −0.0065, 0.0001, 0.99998). So the optimiser could only rotate about z,
  by an angle soft-capped at `π/36` = **±5°**. One of three rotational DOF.
- `scale` is initialised to a hardcoded `N(100, 0.01)` while the soft constraint bands it to ±5% of
  `init_scale` (~65 mm here), so the penalty term hits ~1e6 the moment constraints activate at step
  500 and dominates the SDF loss.
- `init_center` is computed but never used to initialise `translation` (which starts at ~0 while
  `xyz` are raw mm coordinates); it is used only inside the translation constraint.
- Uses the **legacy concatenated-`x` decoder interface** (`TriplanarDecoder.forward(x=...)`,
  `NSM/models/triplanar.py:330`). Checked against the real bone-only checkpoint: it still works and
  agrees with the fast path to float32 epsilon (max abs diff 1.2e-7). It is only *slower* — it goes
  through `unique_consecutive` on an N×512 expanded latent instead of `FastUnique`. Worth switching
  to `decoder(latent=..., xyz=...)` during the rewrite; **not** a blocker, and not a reason the
  original attempt failed.
- Returns `latent_` from the best step but `R`, `s`, `t` from the last step — mismatched iterates.

**When resumed** (after Phases 1–3): rewrite the parameterisation — continuous 6D rotation
representation, or so(3) `torch.linalg.matrix_exp` built with torch ops only; `exp(log_s)` for
scale; warm start from the robust ICP; separate (much lower) LR for pose than latent; keep it a
similarity, never affine. **Acceptance gate: a unit test asserting every pose parameter receives a
non-`None` gradient**, before any conclusion is drawn about whether the method works.

---

## 6. Validation

**Unit.**
- Determinism: same mesh twice → bit-identical transform.
- Order-invariance: permute vertices → transform unchanged to float tolerance. This is the test that
  would have caught the original bug.
- Seeded blue noise: same seed → identical points; seed 0 → **assert it is rejected**.
- Rotation averaging: chordal mean of known rotations recovers the truth; `det > 0` guaranteed.

**Regression (blocking).** Defaults must reproduce pre-change transforms bit-for-bit on a stored
fixture. Without this, every historical latent silently becomes incomparable.

**Integration.** Re-run the kneepipeline shuffle harness
(`analysis/bscore_variance_source_2026-08-11/scripts/followup.py`): B-score spread across vertex
permutations should collapse from SD 0.0668 / 0.1166 to the ICP-frozen level (0.0111 / 0.0033).

**Cohort.** Re-measure scan–rescan mdc95. That is the number that decides whether any of this
worked. Observed today: 0.4.

**Coordination.** Changing registration changes latents, which invalidates `mean_healthy` /
`std_healthy`. Per the kneepipeline report (Q13/Q14) the *direction* `bscore_vector` is safe
(0.73° tilt at n≈4500/group ⇒ ~0.005 B-score), but recalibration costs a re-fit and a full 9k-knee
re-fit is ~125 GPU-hours on a T4. **Batch all recipe changes into one re-baseline** rather than
paying that per change.

---

## 7. Results — Phase 2 tradeoff study (2026-08-11)

1,046 registrations across 14 jobs. Raw data `/mnt/data/knee_pipeline_data/icp_grid/`.

> **What A and B actually are — read this before trusting any number below.** They are **two
> pipeline jobs over the same MRI volume** (`f370ce7d`, Apr 3 and `c4bb8d73`, Apr 6): identical
> array size, spacing, origin and direction; the label maps differ at **exactly 2 voxels**; the job
> configs differ only in `perform_bone_and_cart_nsm`. There is **no repositioning, no FOV
> difference, no independent re-tessellation** — none of the things that make a rescan a rescan.
> After canonical sorting, 98.8% of vertices agree to <1e-4 mm. This is a *re-processing* pair, not
> a scan–rescan pair, and §7.4–§7.5 show that distinction is load-bearing.

### 7.0 Two regimes — do not conflate them

- **MATCHED** — both meshes use the *same* deterministic landmark rule (same stride offsets, same
  blue-noise seeds). **This is what production does**, so it is the primary metric.
- **UNMATCHED** — the two meshes draw independent landmark sets. Bootstrapped. This upper-bounds
  the damage when the two vertex arrays are effectively unrelated, which is closer to a *genuine*
  rescan than our test pair is (see §7.5).

### 7.1 Order-invariance is solved — by either sampler

Mesh A vs a vertex-permutation of A, matched rule. The transforms are **bit-for-bit identical** —
max element difference **exactly 0.0** across all 157 matched cells, for *both* samplers. (The
"5e-7°" that the analyzer prints is `arccos` round-off on identical matrices, not a measurement.)

Against **0.72°** for the current `vtk_stride` on the raw array. The Finding-1 bug is closed by
canonicalising the landmark rule; ensembling is not needed for this part.

**Corollary — this arm carries no discriminative signal.** Both candidate samplers are
order-invariant *by construction*, so order-stability cannot rank them. Everything that
distinguishes the two options rests on the A-vs-B number, which §7.4–§7.5 show is compromised.

### 7.2 More repeats of fewer points wins — clearly

Matched rescan rotation disagreement at roughly equal cost (canonical stride):

| budget | many small | ← vs → | one big | ratio |
|---|---|---|---|---|
| ~12 s | 250 × 10 → **0.0132°** | | 2000 × 1 → 0.0382° | **2.9×** |
| ~25 s | 250 × 20 → **0.0071°** | | 4000 × 1 → 0.0360° | **5.1×** |
| ~50 s | 1000 × 10 → **0.0042°** | | 8000 × 1 → 0.0198° | **4.7×** |

Ensembling beats a single larger ICP at *every* budget tested. The optimal landmark count drifts
up slowly with budget — 250 at ~12–25 s, 500–1000 at ~50 s, 1000–2000 at ~110 s — so the
**sweet spot is ~500–1000 points × 10–20 repeats**. Scaling follows the expected ~`1/sqrt(k)`.

### 7.3 `max_n_iter=100` is wasted compute — and slightly harmful

n_pts=1000, canonical stride, matched rescan:

| max_n_iter | k=1 | k=5 | k=10 | cost at k=10 |
|---|---|---|---|---|
| **30** | **0.0492°** | **0.0041°** | **0.0020°** | **36.3 s** |
| 100 (current) | 0.0524° | 0.0041° | 0.0024° | 51.3 s |
| 300 | 0.0521° | 0.0042° | 0.0027° | 97.8 s |

30 iterations is as good as or better than 100 and 300, at 1/2 to 1/3 the cost. More iterations is
*slightly worse* — plausibly because ICP commits harder to its particular landmark set, which
increases the variance across landmark draws. **The hardcoded 100 should be exposed and lowered.**

### 7.4 Blue noise vs canonical stride — RETRACTED, the two arms do not measure the same thing

An earlier draft of this section claimed canonical stride beats blue noise 6–15×. **That claim is
withdrawn.** Adversarial review established that the comparison is an artifact of the pairing, not
a difference in sampling quality:

| n_pts × k | stride paired | stride **unpaired** | stride within-A | bluenoise paired | bluenoise **unpaired** | bluenoise within-A |
|---|---|---|---|---|---|---|
| 250 × 1 | 0.0731° | 0.877° (12.0×) | 0.883° | 0.601° | 0.585° (0.97×) | 0.575° |
| 1000 × 1 | 0.0508° | 0.433° (8.5×) | 0.427° | 0.233° | 0.258° (1.11×) | 0.249° |
| 1000 × 5 | 0.0080° | 0.167° (20.9×) | 0.164° | 0.121° | 0.084° | 0.102° |
| 4000 × 1 | 0.0360° | 0.191° (5.3×) | 0.209° | 0.108° | 0.105° (0.97×) | 0.079° |

Read the *within-A* columns. For **stride**, unpaired A-vs-B (0.877°) ≈ within-A spread (0.883°):
pairing by rep index removes essentially **all** of the landmark-sampling variance, because the rep
index *is* the canonical-sort offset and A and B are 98.8% rank-identical — the same offset selects
literally the same physical points on both meshes. Measured landmark displacement between A and B
at the same offset: n=250 mean 0.208 mm with **median exactly 0.0 mm**. For **blue noise**, paired
≈ unpaired ≈ within-A (ratios 0.97–1.11×): its Poisson-disk accept/reject sequence diverges between
the two meshes even at a fixed seed (345 vs 364 points; mean nearest-neighbour 2.76 mm = 44% of the
inter-landmark spacing), so nothing cancels and its number carries the full sampling variance.

**So stride was scored on a metric with its dominant error term cancelled, and blue noise was not.**
The two arms ran different experiments. Nothing here ranks the samplers.

### 7.5 What would settle it

The honest comparison needs a pair whose vertex arrays have no index correspondence — i.e. real
scan–rescan meshes: repositioned limb, different FOV, independent segmentation and tessellation.
In that regime both samplers land nearer their *unpaired* columns, where they are within ~1.3× of
each other and blue noise is marginally ahead at larger k.

**Resolved in §7.5b** by adding an accuracy anchor rather than new data: blue noise wins on
accuracy at every ensemble size. Replication on genuine scan–rescan pairs is still wanted, but the
sampler question is no longer blocked on it.

### 7.5b RESOLVED — blue noise is the better sampler, once an accuracy anchor is added

§7.4 said the sampler question was open. It is now answered, by adding the accuracy anchor that
§7.7 flagged as missing: ICP using **every** vertex (113,142) on both meshes, 494 s each.

**The anchor changes the whole reading of this study:**

```
ALL-POINTS reference, A vs B:  rotation 0.00004 deg,  scale 0.00003 %
```

The two surfaces are, for registration purposes, **identical**. So the true A-vs-B disagreement is
zero, and *every* nonzero A-vs-B number in §7.2–§7.4 is pure landmark-sampling artifact — not a
rescan signal. On this pair the only meaningful quality metric is **distance from the all-points
answer**.

Measured at ~1000 landmarks per draw, ensembles of k (accuracy = rotation from the all-points
transform):

| k | bn_trim accuracy | bn_notrim accuracy | **stride accuracy** | stride A-vs-B |
|---|---|---|---|---|
| 1 | 0.0801° | 0.1944° | 0.4659° | 0.0730° |
| 5 | 0.1303° | 0.1334° | 0.1876° | 0.0034° |
| 10 | 0.0922° | 0.0959° | 0.1410° | 0.0024° |
| 20 | **0.0656°** | 0.0860° | 0.1039° | 0.0025° |

**Blue noise is closer to the truth than canonical stride at every ensemble size** — 1.6× closer at
k=20, at equal coverage (20 × 1000 = 20k samples either way).

**Stride is reproducibly wrong, which is exactly the failure mode §7.7 warned about.** Its A-vs-B
disagreement collapses to 0.0025° while it sits 0.104° from the correct transform. The reason is
that the two meshes share 98.8% of their vertices, so a vertex-index rule picks the *same* points on
both and therefore incurs the *same* bias, which cancels in the difference and is invisible in any
reproducibility-only metric.

**Why stride is biased at all:** it samples *vertices*, and marching-cubes vertex density is not
uniform per unit area — it concentrates where the surface is curved. A vertex-strided subset
therefore over-weights high-curvature regions and pulls the ICP fit. Blue noise samples the
**surface** uniformly by area, so it is unbiased by construction. This is precisely the argument in
§3.1, and it is now measured rather than asserted.

**The trim was not the problem.** `bn_trim` (linspace-trim to exactly 1000) is as good as or better
than `bn_notrim` (use pcu's natural ~988 output) at every k, so the harness's trim hack can stay.

**Consequence for §7.4:** the retraction stands, but the resolution is the opposite of the original
draft. Ensembled blue noise is the recommended sampler; canonical stride's apparent advantage was an
artifact of a shared bias on a near-duplicate pair.

### 7.6 Recommended configurations (clean single-threaded timings, no worker contention)

| config | order-invariant | A-vs-B rot (paired) | cost |
|---|---|---|---|
| **CURRENT** — `vtk_stride`, 1000, `max_iter=100` | ✗ (0.72° across orderings) | — | 4.55 s |
| canonical stride, 1000 × 1, `max_iter=30` | ✓ bit-identical | 0.0492° | **2.33 s** |
| canonical stride, 500 × 10, `max_iter=30` | ✓ | ~0.0083° (but see §7.5b — reproducibly biased) | 11.7 s |
| canonical stride, 1000 × 10, `max_iter=30` | ✓ | 0.0020° | 23.0 s |
| canonical stride, 2000 × 10, `max_iter=30` | ✓ | 0.0011° † | 45.7 s |

† **Winner's curse — do not treat this as the best cell.** It is a single realisation (`ngroups=1`).
Jackknife on its own 10 runs gives k=9 → **0.0043 ± 0.0015**, and the two disjoint k=5 halves are
0.0093 and 0.0090. The reported 0.0011 sits below the 5th percentile of its own leave-one-out
family — a fortuitous cancellation. Honest expected level ~0.004–0.008, which puts
**1000 × 20 (0.0025) ahead of it**. The same caution applies to every `ngroups=1` cell in §7.2–§7.4,
and the analyzer's `+/- 0.00000` for those cells means *undefined*, not *measured zero*.

**The minimum-viable fix is cheaper than what runs today**: canonical ordering with `max_iter=30`
and no ensembling costs **2.33 s vs 4.55 s** and removes the order-dependence entirely.

**Recommended default, revised after §7.5b: seeded blue noise, ~1000 points × 10–20 repeats,
`max_iter=30`.** Blue noise is 1.6× closer to the all-points transform than stride at k=20 and is
unbiased by construction; stride's lower A-vs-B number is a shared bias that will not survive a real
rescan. Cost is comparable — blue-noise sampling adds ~0.15 s per draw on top of the same ICP — so
expect ~12–25 s for k=10–20 at 1000 points. Ship the ensemble-spread QC metric with it.

### 7.7 Caveats

- **The A/B pair is a re-processing pair, not a rescan pair** (see the box at the top of §7). This
  is the single biggest limitation and it is what invalidated §7.4.
- **Winner's curse on the Pareto front.** The running minimum is taken over 54 cells whose median
  relative SD is ~0.5, so the lowest values are biased low. Read the front as a *shape* (ensembling
  beats one big ICP) rather than as a ranking of individual cells.
- Grid timings used 3 concurrent workers on 4 cores, so absolute costs there are inflated; §7.6
  re-times the recommended configurations single-threaded. Relative comparisons within the grid
  hold because all wave-1 jobs shared the same contention and every cost is taken from the A side.
- One knee, one pair. Needs replication across knees.
- **Accuracy anchor added in §7.5b** — and it mattered: canonical stride turned out to be exactly
  the "reproducibly wrong" case this caveat anticipated. Note the anchor was measured at
  `max_iter=100`; the `max_iter=30` recommendation (§7.3) is still stability-only and should be
  re-checked against the anchor before adoption.
- Across all 1,022 recorded transforms every `det(R) = 1.000000` and the max rotation deviation from
  the global mean is 1.996°, with no run beyond 5° — so no local-minimum events occurred anywhere in
  this grid. Rotation averaging was never asked to average across distinct basins here; that
  safeguard remains untested and still belongs in the implementation (§3.2).

---

## 8. Evidence locations

| what | where |
|---|---|
| full diagnostic report | `kneepipeline/analysis/bscore_variance_source_2026-08-11/REPORT.md` |
| perturbation arms, shuffle harness | `.../scripts/sampling_experiment.py`, `.../scripts/followup.py` |
| ICP landmark sweep | `.../scripts/icp_landmarks.py` |
| ICP vs CPD comparison | `.../scripts/reg_compare2.py` |
| bagged ICP | `.../scripts/bagged_icp.py` |
| tradeoff grid | `.../scripts/icp_grid_run.py`, `.../scripts/icp_grid_analyze.py` |
| raw outputs | `/mnt/data/knee_pipeline_data/sampling_sensitivity/`, `/mnt/data/knee_pipeline_data/icp_grid/` |
| cached NSM mean mesh | `/mnt/data/knee_pipeline_data/sampling_sensitivity/nsm_mean_mesh.vtk` |
