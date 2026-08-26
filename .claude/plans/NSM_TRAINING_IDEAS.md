# Ideas to improve NSM decoder training

## State

**Updated:** 2026-08-25 · **Status:** open

> **This is the ideas file.** Per `CLAUDE.md` § Documents and work, new training ideas are
> appended here rather than given their own plan. An idea graduates to its own plan only
> when someone commits to executing it.

- **Next:** the maintainer's coupled Idea 4 decision — what training does about the
  bound (4(b): remove / raise / soften) and what reconstruction does about landing
  off the shell (Idea 7 is the recon-side lever) — now that 4(a)'s diagnostic is in
  (2026-08-25, results in the Idea 4 entry). The re-sweep (Idea 11) still waits on
  that decision. One input only the maintainer can supply: the shipped 647/551
  checkpoints carry no training latent codes, so their saturation is confirmed by
  analogy (231's codes) plus observation — a histogram over the training runs'
  `latent_codes/*.pth` on the training machine settles it directly. Idea 6 keeps
  its measurement question with downstream stakes (2026-08-23): is scale leaking
  into the shipped distal-femur latents, and how does the scaling choice move the
  severity task? Ideas 10 (surface-residual training metric)
  and 11 (hyperparameter re-sweep) added 2026-08-23. Idea 12 (recon-hyperparameter
  sweep entry point; inference-side, no retrain) added 2026-08-24, parked by
  §8.0.E's deletion of `tune_reconstruction`. Idea 3 (test the Eikonal loss)
  keeps its live dependency — `NSM_CODE_HEALTH_REFACTOR.md` §8.2 found three
  independent failures and gated the loss behind `NotImplementedError`.
- **Blocked on:** the Next is a maintainer call; every entry stays independently
  executable in the meantime.
- **Done:** Idea 4(a), 2026-08-25 — training-shell saturation confirmed on the one
  shipped latent-code file (median code norm 9.987 of a 10 bound), fitted
  production norms median ~7.3, and the error–norm correlation splits by level
  (within-subject spearman −0.81; across-subject +0.12 to +0.56). Numbers, method
  and provenance in the Idea 4 entry.
- **Surprises:** the error–norm correlation has opposite signs at the two levels —
  with anatomy fixed, fits landing nearer the training shell are almost universally
  better, yet across subjects the higher-norm fits are the *worse* ones (difficulty
  confound) — so "pull every fit to norm 10" is not the free win the premise alone
  suggested. And the shipped production checkpoints turn out to carry no training
  latent codes at all, so the saturation claim is directly measurable only on the
  231 cartilage model. Earlier: the Eikonal loss turned out to be unrunnable rather
  than untested — it
  crashed on the first backward pass, cannot work at all for triplanar models, and opposes
  the clamped-training regime NSM actually uses. See `NSM_CODE_HEALTH_REFACTOR.md` §8.2.

---

**Status:** Open master list — a running register of decoder-training changes
to try. Not a single sequenced plan; each entry is an independent experiment
that can be picked up on its own. Append new ideas as they come up.
**Created:** 2026-05-18.
**Repo:** `/dataNAS/people/aagatti/programming/NSM/` (NSM).

> **Common theme.** Unless an entry says otherwise, ideas here are **upstream,
> retraining-required** changes to how the SDF decoder is trained. They are deliberately out of scope
> of `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` (which is inference-only numerical
> fixes), but several of them would make that plan's stepping *exact* rather
> than approximate. Where an idea supports a downstream plan, it is noted.

---

## Idea 1 — Eikonal training along interpolation paths

**What.** Enforce the Eikonal property `‖∇_x SDF‖ = 1` not only at the trained
latents but at **interpolated** latents `z(t) = slerp(z_A, z_B, t)`. During
training, sample random latent pairs, random `t`, random points, and add the
Eikonal loss `(‖∇_x SDF(x, z(t))‖ − 1)²` at those off-manifold latents.

**Why.** A true SDF has `‖∇_x SDF‖ = 1` everywhere. The NSM decoder is not
Eikonal-trained, so its gradient magnitude drifts — worst off the training
manifold, which is exactly where the correspondence flow lives. This is the
principled fix for `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` Fix 2: if the
decoder were near-Eikonal along paths, the unit-normal projection step would be
the exact Newton step and the magnitude correction would be unnecessary. The
Eikonal property is intrinsically per-fixed-latent; "across latents" just means
enforcing it at *more* (interpolated) latents — there is no canonical
`‖∇_z SDF‖` constraint because latent space has no physical metric.

**How.** `NSM/NSM/losses.py` already has an Eikonal loss — but `CLAUDE.md` notes
it is **untested** (see Idea 3). Extend the sampling in the training loop
(`NSM/NSM/train/train_deep_sdf.py`) to draw interpolated latents and points,
add the extra `autograd` gradient term, weight it via the existing weight
scheduler (`NSM/NSM/train/utils.py`).

**Cost / retrain.** Requires a full retrain. Adds an extra gradient evaluation
per step at the interpolated samples — standard SIREN/IGR-style overhead.

**Caveat.** Will *not* delete the meniscus medial ridge — that crease is
intrinsic to any thin-shell SDF and is still Eikonal away from the
measure-zero ridge. It will, however, discourage the network from
over-flattening the ridge (a flattened crease has `‖∇‖ < 1`), mildly sharpening
it. The which-wall ambiguity remains.

**Supports.** `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` (Fix 2 / Fix 6).

**Status.** Idea — not started.

---

## Idea 2 — Normal-smoothness-vs-`z` regularizer

**What.** Penalize *abrupt* change of the surface normal `∇_x SDF` as the latent
moves. Concretely: penalize the **second-order** term — `∂²(∇_x SDF)/∂z²`, or
high-frequency content of `∇_x SDF` measured *along* `slerp` interpolation
paths parameterized by `t`. Sample latent pairs, sample `t`, penalize the
roughness of `∇_x SDF(x, z(t))` along `t`.

**Why.** The correspondence flow integrates the normal field as `z` changes. If
the normal field varies smoothly and slowly with `z`, the flow is a smooth,
low-curvature ODE → fewer steps, less tangential slip, more stable
interpolation. If normals flip abruptly at some `z`, the trajectory kinks and
integration breaks down.

**Two sharpenings (important).**
- *Penalize roughness, not change.* Some normal rotation with `z` is **correct**
  — the shape genuinely deforms. Penalizing the first derivative `∂(∇_x SDF)/∂z`
  directly would fight the model's ability to represent shape variation. Target
  the second derivative / high-frequency content: normals free to rotate, just
  not to jerk. Measuring along `slerp`-`t` (not raw latent units) sidesteps the
  "latent space has no metric" problem.
- *At genuine topology changes the normal **should** jump.* When an osteophyte
  appears there is no smooth normal interpolation; the field is legitimately
  discontinuous. The penalty will fight reality there and smear the topology
  change. So this lever helps the **smooth-morph** case (menisci — the active
  need) and is questionable at osteophytes.

**Scope.** This is a *flow-niceness* lever — it makes interpolation
well-behaved. It does **not** change the correspondence coupling (still
least-norm / closest-point). It is not a substitute for the rectified-flow
operator.

**Cost / retrain.** Requires a full retrain. Adds a second-derivative-style
penalty term along sampled paths.

**Supports.** `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` (integration stability,
fewer steps).

**Status.** Idea — not started.

---

## Idea 3 — Test the existing Eikonal loss

**What.** `NSM/NSM/losses.py` contains an Eikonal loss; `CLAUDE.md` explicitly
flags it as never tested ("EIKONAL LOSS HAS NOT BEEN TESTED. WE SHOULD TEST
THIS TO MAKE SURE IT WORKS, DOESN'T ERROR, AND TO SEE HOW IT CHANGES THINGS.
NAMELY — DOES IT CHANGE INTERPOLATION?"). Run a controlled experiment: train
with the Eikonal loss on, measure that it does not error, and measure its
effect on reconstruction accuracy and on interpolation quality (using the
`correspondence_metrics.py` from the mesh-interpolation plan's Phase 0).

**Why.** Idea 1 builds on this loss; it should be known-good first. And the
open question — does Eikonal training change interpolation? — is directly
answerable once the Phase 0 metrics exist.

**Cost / retrain.** One or more training runs. Cheap relative to Ideas 1–2;
it is a prerequisite diagnostic for Idea 1.

**Status.** Idea — not started. Prerequisite for Idea 1.

---

## Idea 4 — Remove (or soften) the latent `max_norm` bound; close the train/recon norm gap

**What.** Training embeds latents in `nn.Embedding(..., max_norm=latent_bound)` with
`latent_bound: 10` in both shipped configs, and the maintainer observes the trained
latents all sit **at** norm 10 — the clip is actively binding, so the training codes
live on the radius-10 shell. Reconstruction fits the latent with no norm constraint
(`l2reg_recon: false`) and lands at norm ~6–7, **even for subjects that are in the
training set**. Experiment: (a) diagnostic first, no retrain — histogram the norms of
the shipped checkpoints' latent codes to confirm saturation, and check whether recon
error correlates with the fitted norm's distance from 10; (b) retrain with the bound
removed or raised, and/or replaced by a soft penalty (the L2 code regularization
already exists), and compare recon accuracy on training subjects, generalisation, and
the maintainer's reconstruction artefacts.

**Why.** If every training code has norm exactly 10 and every fitted code has norm
6–7, reconstruction is decoding from a region of latent space the decoder never saw
during training — a plausible cause of the observed recon artefacts. A hard
`max_norm` renormalisation is also invisible to the optimizer (PyTorch applies it
in-place on forward), unlike a soft penalty the gradients can feel.

**Cost / retrain.** (a) is an afternoon with existing checkpoints. (b) is one
training run per bound setting.

**(a) executed 2026-08-25 — saturation confirmed; the error correlation splits by
level.** Measured on the kneepipeline machine's artifacts. Models:
`NSM_MODELS/{647_nsm_femur,551_nsm_femur_bone,231_nsm_femur_cartilage}_v0.0.1`, all
`latent_size` 512 / `latent_bound` 10 / `latent_init_std` 0.01 (init norm ≈ 0.23).
Fits: `/mnt/data/knee_pipeline_data/archive` (website production jobs, one fit per
subject) and `nsm_seed_analysis` (74 subjects × 10 seeds, bone-only).

- **Training-side saturation is real — on the one shipped latent-code file.** Only
  the 231 cartilage model ships `latent_codes/2000.pth`; the `aagatti/ShapeMedKnee`
  HF listing carries **no latent codes for 647/551** (decoder + optimizer state
  only), so the production models' training codes exist only on the training
  machine. 231's 6,325 codes: median norm 9.987, min 8.99, 95% within 0.1 of 10 —
  the codes live on the radius-10 shell, the clip actively binding.
- **The fitted-norm gap is real.** 161 archived bone+cart fits: median ‖z‖ 7.26
  (p5–p95 6.30–8.70), 4 fits *above* 10 (max 10.70 — reconstruction is unbounded).
  123 bone-only fits: median 7.27, max 9.45. So production reconstruction decodes
  from radii below every measurable training code (min 8.99).
- **Within subject the gap predicts error; across subjects the sign reverses.**
  74 × 10 seeds, bone-only: spearman(assd_bone, ‖z‖) median **−0.81**, negative for
  98.6% of subjects — anatomy fixed, the seed that lands nearer the shell fits
  better, though the seed-level stakes are tiny (assd range across a subject's
  seeds: median 0.005 mm vs ~0.27 mm between-subject median). Across subjects it
  reverses: spearman(assd, ‖z‖) **+0.12** (bone-only), +0.11 (bone+cart, bone), and
  **+0.56** for cartilage assd (excluding the four >10 fits) — atypical anatomy
  lands at higher norm *and* fits worse, a difficulty confound dominating between
  subjects.
- **BScore is nearly uncorrelated with fitted norm** (spearman |r| ≤ 0.17, both
  models) — no sign today's severity output rides the norm.

Read for the decision (hypothesis, not measurement): the within-subject result
supports the premise — decoding off-shell costs accuracy, so both levers stay live
(recon-side: pull the fit toward the shell, Idea 7; training-side: free the codes
off the shell, 4(b)). The across-subject reversal says "force every fit to norm 10"
is not a free win: the worst-fit subjects are already the highest-norm ones.

**Status.** (a) executed 2026-08-25, results above; (b) not started — it now has the
saturation and gap numbers to design against. Raised by the maintainer 2026-08-22
while reviewing the audit disposition (register since retired; the filed issues are
#40–#61). Related trap: latent *gradients*
scale with query-point count (ARCHITECTURE §6), which affects any soft-penalty
balance chosen in (b). **Re-raised and prioritized 2026-08-23: this goes first.** The
recon-side norm-gap fix and the training-side bound are one coupled decision — what
reconstruction should do about landing at norm 6–7 only makes sense against a decision
about why training pins codes at 10 — so (a)'s diagnostic precedes both, and it also
precedes the re-sweep (Idea 11), since the bound decision changes the latent geometry
every other knob is tuned against. Idea 7 (the barrier penalty) is the existing
recon-side lever if the answer is "constrain the fit toward the training shell".
One residue only the maintainer can close: 647/551's saturation is confirmed by
analogy (231) plus the maintainer's observation — one histogram over the training
runs' `latent_codes/*.pth` on the training machine settles it directly.

---

## Idea 5 — Re-test concatenated vs summed triplanar features, after the draft-6 fix

**What.** `sum_conv_output_features: false` (concatenate per-plane features instead of
summing) silently trained on **one plane of three** — the yz and xy plane slices were
zero-width (issue #45). The maintainer's past experiments
with this option concluded concatenation never improved results and sometimes hurt;
those runs were degraded by the bug, so the comparison has never actually been run.
After the fix lands: train matched pairs (sum vs concat, same data/seeds) and compare
recon accuracy and training stability.

**Why.** The original hypothesis — concatenation preserves per-plane information that
summing destroys — was never tested, only a broken implementation of it. The prior
negative result should be withdrawn rather than trusted.

**Cost / retrain.** Two training runs. The prerequisite fix has landed.

**Status.** Idea — not started, and **unblocked as of 2026-08-26**: §8.0.H shipped the fix
(all three planes get `sdf_latent_size // 3` channels; `KNOWN_ISSUES.md` § History 15).
The VAE's output width is unchanged, so a pre-fix `sum_conv_output_features: false`
checkpoint still loads — and computes something different, which is why the pairs have to
be retrained rather than re-evaluated.

---

## Idea 6 — A size-preserving mode: rigid-at-true-size registration under `scale_jointly`

**What.** An *additional* dataset mode in which reference-mesh registration still uses
similarity (rigid + uniform scale) for anatomical correspondence, but the transform's
uniform-scale component is then undone before sampling — each subject sits rigidly
aligned at its own true size — and `scale_jointly`'s shared max-radius +
`joint_scale_buffer` frame maps everyone into the unit domain while preserving
between-subject size ratios.

**Why.** Today between-subject size never reaches the model on any registered path:
per-subject normalization erases it by construction (`max_rad` to the unit sphere), and
reference-mesh registration erases it because both samplers hardcode
`reg_mode="similarity"` — measured 2026-08-22 with the exact `rigidly_register` call
`read_meshes_get_sampled_pts` makes: a radius-1.3 sphere registered to a radius-1.0
reference lands at radius 1.0000 (ICP scale factor 0.7692 = 1/1.3). Size survives only
with no `reference_mesh` at all under `scale_jointly=True`. Bone size plausibly carries
disease signal (maintainer, 2026-08-22), so a mode that keeps it may be a beneficial
addition. Size-invariant training stays valid and remains the default — this is an
alternative worth having, not a defect in the current behaviour, which is why it lives
here and not in `KNOWN_ISSUES.md`.

**How.** Extract the uniform scale from the ICP transform (cube root of the 3×3 block's
determinant) and divide it out of the transform before applying, behind a new opt-in
config key; the reconstruction path (`reconstruct/main.py`, `register_similarity`) must
offer the matching mode so fitted latents live in the frame the model trained in. The
coordinate frame changes for runs that opt in, so the key needs the loud-config
treatment of `NSM_CODE_HEALTH_REFACTOR.md` §4.

**Cost / retrain.** Full retrain for any model that wants size sensitivity; existing
models and defaults are untouched (opt-in).

**Downstream stakes — measure before designing (maintainer, 2026-08-23).** For the
distal femur models the suspicion now runs in *both* directions: size may carry signal
this pipeline erases (above), but scale may also be **leaking into the latent
uncontrolled** ("it seems like we're sneaking scale in"), and either failure — erased
when it carries signal, or leaking when it should be controlled — moves the
disease-severity predictions (BScore) built on these latents. So the first step is a
measurement, not the mode: for the shipped femur models, determine what actually
reaches the latent today (correlate fitted latents against subject size on data with
known sizes, across the registration/scaling paths kneepipeline really uses), and
quantify how the scaling choice moves the downstream severity task. This raises the
idea's priority: it is not only a capability gap but a potential confounder in the
pipeline's primary output.

**Status.** Idea — not started.

---

## Idea 7 — Make the barrier norm penalty usable from an infeasible start

**What.** Two candidate designs for `norm_penalty_type='barrier'` (the soft latent-norm
constraint in `reconstruct_latent`) so it works when the latent starts outside the
`(min, max)` range — which is the initialization state for any plausible range
(std 0.01 puts a 256-dim latent at norm ~0.16). Since the #48 fix it raises by name
there; these designs would make it *work* instead.

- **(a) Relaxed log barrier** (interior-point literature). Splice at a threshold δ
  *inside* the feasible region: `−log(t)` for `t > δ`, and for `t ≤ δ` a quadratic
  extension matching value and slope at δ. C¹ and monotone from far outside, through
  the boundary, into the interior. The naive alternative — true log inside plus a
  linear tail outside — has a cliff at the boundary: approaching from outside the tail
  decreases, but the step that crosses lands on the log's +∞ wall, so the optimizer
  parks just outside forever (maintainer's observation, 2026-08-23). Price of the
  relaxation: the wall is finite near the edge, so "guarantee" degrades to "stiff
  spring" exactly where the barrier's selling point was the guarantee.
- **(b) Range annealing / continuation** (maintainer's suggestion: curriculum). Start
  with bounds wide enough to contain the init norm and shrink them toward the target
  range over the first N steps; the true log barrier stays defined at every step and
  keeps its infinite wall. This is what interior-point solvers do with the barrier
  coefficient μ → 0. Failure mode to design around: the moving bound must never cross
  the current norm — that is instant NaN — so if the data term pins the latent against
  the shrinking wall, a fixed schedule breaks; it wants an adaptive schedule ("never
  tighten past the current norm"), which is more machinery to get right.

**Why.** The barrier is the only *interior* penalty in the option set — quadratic and
huber are exterior penalties, zero inside the range and acting only after a violation.
The property worth wanting is "the fit can never leave the range", which the exterior
penalties do not give. But a log barrier assumes a feasible start, and reconstruction
cannot provide one.

**Evaluation bar.** Whether any of the three penalty types beats the others has never
been measured. The bar for adopting either design is a comparison against `quadratic`
on real fits — recon error and achieved norm — not merely "no NaN". Without that
measurement this stays an idea.

**Cost / retrain.** None — reconstruction-time only (the exception to this file's
common theme), ~10–30 lines either way.

**Status.** Idea — not started. Raised 2026-08-23 while fixing the #48 NaN
(`docs/KNOWN_ISSUES.md` § History §8). Production never sets `latent_norm`, so there
is no current user to migrate.

---

## Idea 8 — Supervised / contrastive signal on the latent during training

**What.** Bring subject-level factors (age, OA grade, sex, …) into *training* as an
auxiliary objective on the latent codes, instead of only probing for them at
validation time. Today the factor signal touches nothing: `Regress`
(`reconstruct/predictive_validation_class.py`) is a validation-time linear probe —
fit latents to held-out meshes, regress factors parsed from filenames, report
`val_prediction_<factor>` R² — with no gradient to the model. Candidate shapes:

- **Supervised auxiliary loss** — a small head on the latent predicting the factor,
  weighted into the training loss; organizes latent space along the factor axes.
- **Contrastive** — SupCon-style (pull latents sharing a label together, push others
  apart) or CLIP-style alignment against an embedding of non-imaging data, the
  "language signal helps vision" analogy.

**Why.** Maintainer (2026-08-23): the long-term want is "something akin to `Regress`
but maybe very different structure", for contrastive/supervised learning on the
latent. Reported evidence, not reproduced here: Katie tried contrastive objectives in
her fork — <https://github.com/3D-fossils-Haag/nsm> — and it improved **both**
reconstruction and downstream predictions — which is the interesting part, since an
auxiliary factor loss could plausibly have traded recon quality away instead.

**How.** First step is reading Katie's fork (`3D-fossils-Haag/nsm`, link above) before
designing anything — it is a concrete, reportedly-working implementation, and the
plan's fork-coordination note
(`NSM_CODE_HEALTH_REFACTOR.md` §10) already flags that active forks carry modules
upstream does not have. Mechanically the hook is `train_deep_sdf`'s loss composition;
latents are an `nn.Embedding`, so a latent-side auxiliary loss is cheap. Two known
interactions to design around: the `max_norm` clamp saturating training latents onto
the radius-10 shell (Idea 4 — a contrastive geometry fights a fixed-norm shell), and
latent gradients scaling with query-point count (ARCHITECTURE §6), which affects the
balance of any new latent-side term.

**Evaluation.** The repaired `Regress` probe is the natural metric: `val_prediction_*`
R² with and without the auxiliary signal, alongside recon error — the colleague's
result predicts both should improve. Implication for current code: `Regress` stays a
thin evaluator; the training-time mechanism is new code, not an extension of it.

**Cost / retrain.** Full retrain per objective/weight tried; needs factor labels
available at training time (today they are parsed from validation filenames only).

**Status.** Idea — not started. Raised by the maintainer 2026-08-23 while reviewing
the #48 `Regress` seam fix (PR #73).

---

## Idea 9 — Specifiable sampling composition for latent reconstruction

**What.** Make the composition of the reconstruction fitting pool an explicit,
per-surface specification — e.g. fractions or counts for *surface vertices*,
*near-surface Gaussian draws at sigma*, and *uniform far-field draws* — instead of
the accident it is today. Currently, with `get_rand_pts=True`, each surface's pool is
`n_pts_random` random draws plus **all** of that surface's vertices
(`include_surf_in_pts` is hardwired to `get_rand_pts`), and the per-step batch
inherits that ratio in expectation: the on/off-surface mix is set by `n_pts_random`
relative to whatever vertex count the mesh happens to have, and nothing controls it
directly. Sigma only sets how far off the off-surface points sit.

**Why.** Maintainer (2026-08-23, §8.0.C review discussion): the composition seems
like it matters, and it has never been an experimental variable. Every historical
"does random sampling help" run was confounded anyway — `n_pts_random` was dead
(#16, History §9) and the multi-object appended points were another surface's
vertices in the wrong frame (#17, History §7) — so the question is open, and
composition is the natural axis for the rerun. A structural detail that shapes the
design: every sampled point supervises **every** surface's SDF channel (each
surface's distances are precomputed at all points), so "on-surface for bone" is
simultaneously an off-surface constraint for cartilage; composition controls where
points *concentrate*, not which channels they feed.

**How.** Reader-level, reconstruction-only — no retrain. The pieces that exist:
per-surface sigma is already a list and a `None` entry already draws uniformly from
the buffered cube, so "near-surface vs far-field" needs only letting one surface
contribute *two* draws. Missing pieces: a vertex-count control (subsample the
appended vertices rather than always taking all of them; make `include_surf_in_pts`
a count/fraction, not a hardwired bool), and — if per-step ratios should be exact
rather than in-expectation — a `pts_type` label alongside `pts_surface` so
`reconstruct_latent`'s balanced draw can stratify by type as well as surface. The
femur clip is **not** a blocker for the far-field draws: pcu's pseudonormal sign
gives an open clipped mesh the same coherent field as its capped counterpart, and
training ran the same clip with the same `fix_mesh: False` — measured and pinned by
`testing/NSM/datasets/test_open_mesh_sdf.py`.

**Evaluation.** The deferred resampling experiment, now unconfounded (post-#15/#16/
#17): on/off composition sweep × sigma sweep (informative band is |SDF| <
`clamp_dist` = 0.1 in normalized units; the shipped 0.001–0.01 defaults hug the
surface) against `get_rand_pts=False` vertices-only fits. Metrics: recon
chamfer/ASSD plus `val_prediction_*` R². Watch #75 (per-step sample counts are
memory-bound) if the sweep pushes `n_samples_latent_recon` up.

**Cost / retrain.** None — reconstruction-side only. Interacts with #3 (sigma's
coordinate space) and should land after or alongside its migration guard rather than
adding new sigma semantics on top of ambiguous ones.

**Status.** Idea — not started. Raised by the maintainer 2026-08-23 while reviewing
the §8.0.C fixes (PR #74).

---

## Idea 10 — A surface-residual metric during training: |f(z, x)| on true surface points

**What.** A cheap, frequent training-time proxy for reconstruction quality: evaluate
the decoder at points *on* each training subject's surface — where the true SDF is 0
by definition — using the subject's current training latent, and log the mean
`|f(z_i, x_surf)|` into the epoch's `log_dict`. Forward-only under `no_grad`, no
latent optimization, no marching cubes; a subsample of each subject's mesh vertices
suffices, and the dataset holds every mesh (in the training frame — vertices there
*are* the zero level set) at build time.

**Why.** Today the only reconstruction signal during training is the full validation
run every 50–100 epochs (`get_mean_errors`: latent optimization + marching cubes +
chamfer/ASSD per subject) — the right measurement, but so expensive it is sparse. The
training loss itself is dominated by off-surface points, so it can improve while the
zero level set — the only thing mesh extraction uses — drifts. The surface residual is
exactly the quantity the training loss under-weights, and it is dense: a per-epoch
curve instead of a point every 50 epochs. It is deliberately one-directional
(prediction error *at* the true surface; it cannot see surface the decoder
hallucinates elsewhere, so it is not ASSD and does not replace the validation runs —
it complements them between their samples).

**How / notes.** Compute on the raw prediction, not the clamped one —
`enforce_minmax` clamps predictions during loss computation (KNOWN_ISSUES § Open),
and a clamp would only mask large residuals, though near the surface most values sit
inside the band anyway. With #28 landed, anything added to `log_dict` reaches both
wandb and the returned history with no new plumbing. The same evaluation on *held-out*
subjects still requires a fitted latent, so the dense curve is training-subjects-only;
that is the point — it tracks whether the decoder is representing what it has already
been shown.

**Cost / retrain.** None — a diagnostic, not a loss term. Negligible per-epoch cost
(one forward pass per subject over a few thousand points).

**Status.** Idea — not started. Raised by the maintainer 2026-08-23.

---

## Idea 11 — Re-sweep the training hyperparameters on the repaired trainer

**What.** A structured hyperparameter sweep over the training knobs — rerun, because
the values in the shipped configs were tuned on a trainer whose semantics have since
changed under them, plus axes for knobs that have never been swept and the new ones
the ideas above introduce.

**Why — the tuned values are optimal for bugs that are now fixed.**

- Every Adam/AdamW config was tuned under the swapped LR mapping (KNOWN_ISSUES §1):
  the chosen schedule values were optimal for latents-under-the-model-schedule and
  vice versa. The §9 assessment deferred retuning as "a separate exercise" — this is
  that exercise.
- The `schedule_free_*` configs were tuned against the Adam path (§1) and, until #42,
  died at their first checkpoint epoch — as configured they have effectively never
  been evaluated end to end.
- `weight_decay` under `Adam` was silently dropped (History §4); it is now honoured,
  which changes the landscape for any Adam axis.
- `clamp_dist` is a training-dynamics knob, not the target transform its name
  suggests — KNOWN_ISSUES § Open measured 44.6% of a fresh decoder's predictions
  outside ±0.1, each contributing zero gradient — so it deserves an axis, not an
  inherited constant (the shipped default is 0.1; both production configs use 1.0).
  The axis is the *form* of the clamp as well as δ (maintainer, 2026-08-24): double
  clamp (status quo, inherited from DeepSDF), **target-only clamp**
  (`|pred − clamp(gt, δ)|` — gradient everywhere, far field forced to ±δ plateaus),
  and tanh-plus-loose-clamp (the de-facto production regime: the tanh-bounded output
  makes a ±1.0 prediction clamp inert). Judge with the harness and Idea 10's
  surface-residual metric; the full regime breakdown is in the KNOWN_ISSUES § Open
  entry.
- Never swept in this repo's memory: the curriculum knobs (`surface_accuracy_e`,
  `sample_difficulty_weight` and their schedules), `latent_bound` (Idea 4), and the
  code-regularization family beyond its current values.

**Sequencing.** After Idea 4(a) — the norm-bound decision changes the latent geometry
every other knob is tuned against (maintainer, 2026-08-23: the norm test goes first).
Idea 10's surface-residual metric would make sweep evaluation denser than per-run
`get_mean_errors` alone; #28's returned history means a sweep harness can read every
run programmatically instead of scraping wandb.

**Cost / retrain.** One training run per configuration — the expensive idea on this
list by construction. Sweep infrastructure (wandb sweeps or a driver script) is a
prerequisite decision.

**Status.** Idea — not started. Raised by the maintainer 2026-08-23.

---

## Idea 12 — A reconstruction-hyperparameter sweep entry point, on the shared validation mapping

**What.** A supported way to tune reconstruction hyperparameters (latent-fit lr,
`num_iterations`, `latent_reg_weight`, convergence settings, sampling composition —
Idea 9's knobs included): a thin driver that maps a config to `get_mean_errors`
kwargs, reduces the returned metric dict to a scalar objective (e.g. mean
chamfer/ASSD over held-out subjects), and reports it to an external search tool
(wandb sweep, optuna, plain grid) — one run per config.

**Why.** The fitted reconstruction is what the consumer sees (kneepipeline →
BScore), and its hyperparameters have never been swept independently of training's.
The previous vehicle, `tune_reconstruction`, was deleted by §8.0.E's pass over
`reconstruct/main.py` (SCOPE §2 dead ruling; CHANGELOG Unreleased § Breaking) —
zero callers ever, 22 of its 27 config keys absent from the shipped default, a
second drifting copy of the config→`get_mean_errors` mapping
(`register_similarity` hardcoded), the metric dict discarded (returned `None`),
and wandb plumbing that inits one run per *subject*, the wrong shape for a sweep.
The capability is worth having; that adapter was not it.

**How / notes.** Three pieces, none built until someone commits: (1) extract
`_run_validation`'s config→`get_mean_errors` kwarg block into a shared helper so
the trainer and the tuner drive one mapping — at need, not before; (2) the
objective is `get_mean_errors`' return value reduced to a scalar — since #28/#29
the library returns honest values (NaN on degenerate decoders), so no side-channel
logging is needed; (3) the search loop lives outside NSM, in the driver script.
Post-#5, wandb stays optional: only the driver asks for it.

**Cost / retrain.** None — inference-side tooling, no retraining. Roughly a 20-line
driver plus the mapping extraction. (Unlike most entries here, this is *not* an
upstream retraining-required change.)

**Status.** Idea — not started. Parked 2026-08-24 when §8.0.E deleted
`tune_reconstruction` (PR #78), so the deletion does not silently drop the intent.

---

## Idea 13 — Give the triplanar VAE a pointwise activation, opt-in, and find out if it helps

**What.** `VAEDecoder`'s conv stack has **no pointwise activation**. `__init__` built one
and never appended it, from the first triplanar commit (`71df387`, Aug 2023) onwards, so no
triplanar model NSM has ever produced has had one. `conv_activation` now exists and defaults
to off (§8.0.H); what this idea is, is the matched pair that says whether turning it on is
worth a retrain.

**Why it is not obvious either way.** The stack is not degenerate — LayerNorm supplies a
nonlinearity — but a narrow one: a radial projection that preserves direction and rescales
magnitude. It cannot zero a feature out or form a decision boundary. What an activation adds
is *selectivity*, which for an SDF field should matter most at sharp features and creases.
Against that, the gain LayerNorm supplies is already weak and gets weaker with depth
(`ARCHITECTURE.md` §7.1: σ spreads of 1.71×, 1.30×, 1.15×, 1.02×, 1.00× across 647's five
layers), so there is real headroom — and the shipped models were fitted without any of it.

**What is already settled, so nobody repeats it.**

- **The fix is available and costs nothing to old checkpoints.** Verified: a
  `conv_activation` defaulting to `None` builds an identical module list, strict-loads a
  shipped checkpoint and is bitwise-identical. Only an *unconditional* insert breaks them,
  by shifting every index inside `nn.Sequential`.
- **The regression harness cannot answer this.** Measured over 3 seeds × 2 norm types on the
  synthetic fixture: reconstruction ASSD flips sign with the seed (5.80× better, 1.09×,
  0.69× on `layer`), because the *control* alone varies 11× across seeds — a bad latent fit
  dominates the effect. A single-seed run looked like a 5.8× win and was noise.
- **A naive drop-in trains worse.** Training loss was worse in **5 of 6** of those runs.
  Consistent with LayerNorm's scale invariance currently normalizing the gradients: adding
  activations changes that balance, so **both learning rates need retuning** before the
  comparison means anything. `Conv → LN → SiLU`, or leaky ReLU at 0.2, is the shape to try.

**Cost / retrain.** Two production-scale training runs plus LR retuning. Not answerable at
harness scale — that is the finding above, not an assumption.

**Priority.** Below triplane resolution and feature dimension. Expect a modest gain rather
than a step change; the honest position is that nobody knows, and the shipped models work.

**Status.** Idea — not started, and **the code no longer blocks it** (§8.0.H, PR #90):
`conv_activation` exists, defaults to `None` (the historical stack, byte-identical and
loading every existing checkpoint), and `load_model` requires the config to state which
architecture it means. What is left is purely the experiment — two production-scale runs
with the learning rates retuned. No tracker issue remains: the defect's code half shipped
in that PR, and this entry is the research half (per the docs rules, an experiment is not
a fixable defect and does not meet the tracker bar).

---

## Idea 14 — Weight norm *and* per-layer normalization together in the MLP decoder

**What.** `deep_sdf.Decoder` applies weight norm to every linear layer, or LayerNorm to
selected layers, but never both — the branch is an `elif`. Commit `01d774a` (Jun 2023) set
that structure out with the message *"separate wieght norm and batch norm **so can use
both**"*, which is precisely what the `elif` prevents, so the stated goal was never
delivered and no run has ever had both. Make it possible, and measure whether it helps.

**Why.** The two normalizations do different jobs — weight norm reparameterizes the weights,
LayerNorm normalizes activations — and combining them is standard elsewhere. The reason to
suspect it is worth trying is that a maintainer deliberately built toward it and the code
silently did not follow. The reason not to assume it helps is that nobody has run it.

**What is already settled.** The `elif` is verified: with `weight_norm=True` nothing is ever
appended to the norm list, so every shipped model has weight norm and no LayerNorm. The
`norm_layers` argument itself is **deleted** (`SCOPE.md` §1, unsupported by design), so this
is not "un-delete it" — it is a fresh option whose shape can be chosen with the experiment
rather than inherited. Note the design trap that deletion avoided: the old key indexed the
norm list by absolute layer index, so any set not starting at layer 0 raised `IndexError`.

**Cost / retrain.** Two training runs plus the option. Like Idea 13, adding the layers
renumbers state-dict keys, so it must be opt-in with the historical layout as the default.

**Priority — the lowest in this file, and it should stay there.** Every other entry is here
because a measurement, a defect or a maintainer observation pointed at it. This one is here
because a 2023 commit message stated a goal the code did not reach. That is a reason to
*record* it, not a reason to do it: no shipped model wants it, nothing measured suggests
weight norm alone is the limiting factor, and nobody has asked for it. It also costs more
than its two runs — it means re-adding an option §8.0.H just deleted, so the bar is a
result, not a hunch. **Do not pick this up ahead of anything else here**; if the ideas file
is ever pruned, this is the first entry to go.

**Status.** Idea — not started, very low priority. Surfaced 2026-08-26 while §8.0.H deleted
the half-working option; recorded here so the intent does not die with the code that failed
to implement it.

---

## Related

- `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` — inference-only numerical fixes;
  these training ideas would make several of those exact.
- `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` — the learned correspondence operator.
- `NSM/NSM/losses.py` — Eikonal loss (untested).
- `NSM/NSM/train/train_deep_sdf.py`, `NSM/NSM/train/utils.py` — training loop
  and weight schedulers.
