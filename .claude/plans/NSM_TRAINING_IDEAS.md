# Ideas to improve NSM decoder training

## State

**Updated:** 2026-08-22 · **Status:** open

> **This is the ideas file.** Per `CLAUDE.md` § Documents and work, new training ideas are
> appended here rather than given their own plan. An idea graduates to its own plan only
> when someone commits to executing it.

- **Next:** nothing scheduled. Idea 3 (test the Eikonal loss) is the one with a live
  dependency — it is `NSM_CODE_HEALTH_REFACTOR.md` §8.2, which found three independent
  failures and gated the loss behind `NotImplementedError`.
- **Blocked on:** nothing. Each entry is independent and can be picked up on its own.
- **Done:** nothing from this list has been executed.
- **Surprises:** the Eikonal loss turned out to be unrunnable rather than untested — it
  crashed on the first backward pass, cannot work at all for triplanar models, and opposes
  the clamped-training regime NSM actually uses. See `NSM_CODE_HEALTH_REFACTOR.md` §8.2.

---

**Status:** Open master list — a running register of decoder-training changes
to try. Not a single sequenced plan; each entry is an independent experiment
that can be picked up on its own. Append new ideas as they come up.
**Created:** 2026-05-18.
**Repo:** `/dataNAS/people/aagatti/programming/NSM/` (NSM).

> **Common theme.** Every idea here is an **upstream, retraining-required**
> change to how the SDF decoder is trained. They are deliberately out of scope
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

**Status.** Idea — not started. Raised by the maintainer 2026-08-22 while reviewing
the audit disposition (register since retired; the filed issues are #40–#61). Related
trap: latent *gradients*
scale with query-point count (ARCHITECTURE §6), which affects any soft-penalty
balance chosen in (b).

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

**Cost / retrain.** Two training runs plus the draft-6 fix as a prerequisite.

**Status.** Idea — not started. Blocked on the draft-6 fix.

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

**Status.** Idea — not started.

---

## Related

- `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` — inference-only numerical fixes;
  these training ideas would make several of those exact.
- `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` — the learned correspondence operator.
- `NSM/NSM/losses.py` — Eikonal loss (untested).
- `NSM/NSM/train/train_deep_sdf.py`, `NSM/NSM/train/utils.py` — training loop
  and weight schedulers.
