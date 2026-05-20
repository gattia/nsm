# Ideas to improve NSM decoder training

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

## Related

- `NSM_MESH_INTERPOLATION_IMPROVEMENTS.md` — inference-only numerical fixes;
  these training ideas would make several of those exact.
- `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` — the learned correspondence operator.
- `NSM/NSM/losses.py` — Eikonal loss (untested).
- `NSM/NSM/train/train_deep_sdf.py`, `NSM/NSM/train/utils.py` — training loop
  and weight schedulers.
