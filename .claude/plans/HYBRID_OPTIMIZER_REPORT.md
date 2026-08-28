# Adam + L-BFGS Hybrid Latent Optimizer — Findings Report

## State

**Updated:** 2026-08-28 · **Status:** open

- **Next:** run the six-config comparison in §7 on the 139-case femur validation set. The
  synthetic version is measured and in `docs/KNOWN_ISSUES.md` § Open; what it cannot tell
  us is whether a Wolfe line search still dominates on a trained decoder over real meshes.
  That single run also answers the questions the Aug-2025 sweep never wrote down: fastest,
  best reconstruction, and — with a second fit per case at a different seed — best latent
  for reproducibility.
- **Blocked on:** nothing technical. Needs the validation meshes and the
  `568_nsm_femur_bone_cart_men_v0.0.1` model, both on the Torino box.
- **Done:** the Aug-2025 sweep produced the recipe in §3. Its parameters are audited
  against the code in §4 and **four of them do not mean what the sweep thought**. The
  optimizer's behaviour is measured in §7.
- **Surprises:**
  1. The sweep's `n_samples_latent_recon: 1,000,000` almost certainly meant *no
     subsampling*, which is the one thing the Aug-2026 measurement independently found to
     be best. The sweep and the refactor agree, and neither noticed (§4.4).
  2. `norm_penalty_weight: 100` is not a soft constraint in effect. It outweighs the
     reconstruction term by ~10⁶ at initialization (§4.3).
  3. Three of the recipe's tuned parameters have never existed in NSM (§4.1).
  4. There is no line search. `lbfgs_lr=1.0` is an unguarded full quasi-Newton step, and
     `convergence="recon_loss"` is the only reason the sweep's numbers looked sane (§7).

---

## Purpose of this document

Where the Adam→L-BFGS hybrid came from, what its parameters actually do when the current
code reads them, and what is measured about whether any of it is a good idea.

The Aug-2025 sweep is the origin. The §8.0.K refactor (`latent-fit-internals`, PR #94)
rewrote the function it ran against. Everything below was re-derived from the code on that
branch; nothing is inherited from the 2025 notes without being run.

## 1. Where the code lives

The hybrid path arrived in `01c3fda Implement hybrid optimizer for latent reconstruction`.
It moved out of `main.py` in the §8.0.C split.

**File: `NSM/reconstruct/latent_fit.py`.** `reconstruct_latent(...)` is the whole of it:

| What | Symbol |
|---|---|
| hard hypersphere rescaling | `project_latent` |
| soft norm penalty (quadratic / huber / barrier) | `latent_norm_penalty` |
| the per-evaluation draw | `_select_samples` |
| the Adam→LBFGS switch, and both optimizer constructions | inside `reconstruct_latent`'s loop |
| the LBFGS closure | `step_closure`, nested |

`reconstruct_mesh` (`NSM/reconstruct/main.py`) threads all ten hybrid/norm parameters
through. **`get_mean_errors` does not thread any of them**, so the hybrid is unreachable
from training-time validation — verified against both signatures.

## 2. Where the experiments live

`/dataNAS/people/aagatti/projects/nsm_recon_optimization_testing/` on the Torino box — not
part of this repo. Configs, `test_optimization/test_optimization.py`, and a `wandb/`
directory (project `bone-modeling/nsm-optimization-testing`). The markdown notes are
mirrored at `stuff/Notes_Recon_Testing_Torino/` in this repo.

Validation set: **139 femur cases** (bone + cart + med_men + lat_men) from OAI DESS, with a
14-case subset for iteration. Model: `568_nsm_femur_bone_cart_men_v0.0.1`, `latent_size=1024`,
`objects_per_decoder=4`.

**Numeric rankings were never written down.** The README designates
`test_hybrid_norm_10_3_full_dataset.json` as "the advanced hybrid config" and that is the
entire basis for calling it best. Per-tissue ASSD for the 10_3 and baseline full-dataset
runs is in wandb and nowhere else.

## 3. The recipe as written, and whether it still runs

`test_hybrid_norm_10_3_full_dataset.json`:

| Param | Value | Status against the current code |
|---|---|---|
| `hybrid_optimizer` | `true` | read |
| `latent_optimizer_name` | `"lbfgs"` | **raises `ValueError`** — see §4.2 |
| `adam_iterations` | `10` | read |
| `lbfgs_iterations` | `50` | read |
| `lbfgs_max_iter` | `10` | read (hybrid only) |
| `lbfgs_history_size` | `50` | read (hybrid only) |
| `lbfgs_lr` | `1.0` | read (hybrid only); **no line search guards it** — §7 |
| `lr` (Adam warm-up) | `0.01` | read |
| `n_lr_updates` / `lr_update_factor` | `1` / `1.1` | read; horizon fixed in §8.0.K (§ History 22) |
| `num_iterations` | `2000` | **read for nothing** in hybrid mode |
| `convergence` | `"recon_loss"` | read — and load-bearing, §7 |
| `convergence_patience` | `5` | read; interacts badly with `adam_iterations=10`, §4.5 |
| `min_rel_improve` | `1e-3` | **not an NSM parameter** — §4.1 |
| `grad_tol` | `1e-5` | **not an NSM parameter** — §4.1 |
| `param_change_tol` | `1e-3` | **not an NSM parameter** — §4.1 |
| `latent_norm` | `10.0` | read |
| `norm_penalty_weight` | `100` | read; **dominates the objective** — §4.3 |
| `n_samples_latent_recon` | `1,000,000` | read; **means "full cloud"** — §4.4 |
| `batch_size` | `1,000,000` | **not a latent-fit knob** — §4.1 |
| `loss_type` / `clamp_dist` | `"l1"` / `0.1` | read |
| `latent_init_std` / `latent_init_mean` | `0.01` / `0.0` | read |
| `objects_per_decoder` | `4` | read (model-specific) |

Baseline it was compared against — `baseline_config_full_dataset.json`, Adam, 139 cases:
`lr: 0.005`, `num_iterations: 2000`, `n_samples_latent_recon: 20000`, `batch_size: 300000`,
`n_lr_updates: 100`, `lr_update_factor: 1.1`, `convergence_patience: 50`, `loss_type: "l1"`,
`clamp_dist: 0.1`, `l2reg: false`, no norm constraint. These are the model config's
`*_recon` defaults with the optimizer name corrected from training's `"AdamW"`.

## 4. What four of those parameters actually do

Each of the following was run, not reasoned about. The probes are five-line scripts against
`reconstruct_latent` on CPU.

### 4.1 Four keys that never reached the optimizer

`min_rel_improve`, `grad_tol` and `param_change_tol` **do not exist anywhere in NSM** — zero
hits across `NSM/` and `testing/`. Whatever tolerances the sweep believed it was setting,
it was not setting them here; either the external harness implemented them itself, or they
were swallowed. Before §8.0.J/K's keyword refusal they would have gone silently into
`**kwargs` (§ History 20's exact defect). They now raise `TypeError`.

`batch_size` **is a `reconstruct_mesh` parameter, but not a latent-fit one**: it is the
marching-cubes grid batch (default `32**3`), used when extracting meshes. The knob the
model config means by "batch" is `batch_size_latent_recon`, which is deprecated and does
nothing. So neither the recipe's `1,000,000` nor the baseline's `300,000` touched the fit.

### 4.2 `latent_optimizer_name: "lbfgs"` now raises

Hybrid mode derives its optimizer from the step number and never reads `optimizer_name`.
§8.0.K found that (an accepted-and-ignored parameter) and closed it by refusing the pair:

```
ValueError: hybrid_optimizer=True runs Adam and then LBFGS, so optimizer_name is not
consulted; it was 'lbfgs'. Drop one of the two.
```

**The recipe on disk sets both, so it will not run as written.** Deleting
`latent_optimizer_name` from the config restores exactly the behaviour the sweep had — the
key was inert then too. This is a correct change that invalidates a config file; it is
worth a line in whatever re-runs the sweep.

### 4.3 The norm penalty is soft in code and hard in effect

The 2025 report's headline caveat — "the norm is *soft*, not a hypersphere", because
`use_soft_norm_constraint` defaults to `True` — is right about the code path and backwards
about the consequence. At `latent_norm=10.0`, `norm_penalty_weight=100`, `latent_size=1024`,
`latent_init_std=0.01`:

| ‖z‖ | quadratic penalty at w=100 |
|---|---|
| 0.329 (initialization) | **9,353** |
| 9.0 | 100 |
| 9.9 | 1.0 |
| 9.99 | 0.010 |

An L1 reconstruction loss on SDFs clamped at 0.1 is O(10⁻²). So until ‖z‖ is within ~0.01
of the target, the objective is *entirely* "inflate the latent norm" and reconstruction is
numerically invisible. That is a hypersphere constraint enforced by penalty scale rather
than by projection.

It also reframes the Adam phase. Traced over a real fit at the recipe's settings, ‖z‖ goes
0.33 → 3.39 in the ten Adam steps and needs about forty to reach 10. **`adam_iterations=10`
hands LBFGS a latent at ‖z‖ = 3.4 against a target of 10**, so the LBFGS phase spends its
own first steps on norm inflation too. Whatever the hybrid bought, it was not "Adam warms
up the reconstruction and LBFGS refines it".

The open question the 2025 report asked — "test the hard projection path,
`use_soft_norm_constraint: false`" — is worth less than it looks for the same reason: at
w=100 the two paths are trying to do the same thing. What is worth testing is
`norm_penalty_weight` across orders of magnitude, and whether `latent_norm=10` is anywhere
near the norm of the model's own training latents. Nothing in the sweep checked that.

### 4.4 `n_samples_latent_recon: 1,000,000` means "stop subsampling"

`_select_samples` draws `n_samples // n_surfaces` per surface, capped at what each surface
has, and **returns the cloud untouched and in order when the total equals the cloud size**.
Verified:

| cloud | `n_samples` | drawn | full and ordered |
|---|---|---|---|
| 200,000 (4 × 50k) | 1,000,000 | 200,000 | **yes** |
| 200,000 | 20,000 | 20,000 | no |
| 390,000 (300k + 3 small) | 1,000,000 | 340,000 | **no** — the 300k surface is capped at its 250k share |
| 390,000 | 4,000,000 | 390,000 | yes |

With `get_rand_pts_recon: false` the cloud is the mesh vertices, which for a four-surface
femur is well under a million. **So the recipe was almost certainly running the full cloud
— a deterministic objective — and the Adam baseline at 20,000 was not.** That is the single
most consequential difference between the two configs, and the 2025 report describes it as
"dense surface sampling".

It is also the same conclusion the Aug-2026 measurement reached from the other direction:
the full cloud beats every subsampled regime for LBFGS (`KNOWN_ISSUES` § Open). The sweep
found it by accident and the refactor found it by experiment.

**Caveat for anyone re-running:** the per-surface split means a generous-looking global
`n_samples` still subsamples a surface that is larger than its share, and the runtime
warning does not fire in that case. Confirm the cloud size rather than assuming.

### 4.5 `convergence_patience: 5` can cancel the LBFGS phase entirely

Patience is not reset at the Adam→LBFGS switch. With `adam_iterations=10` and
`convergence_patience=5`, six non-improving Adam steps end the whole fit before LBFGS
runs once — verified: *"switched to LBFGS: False; Converged (recon_loss) after 6 steps"*.
And because `recon_loss` is recomputed from a fresh draw each step whenever the objective is
subsampled, non-improvement happens by chance. Any re-run should log which optimizer the
fit actually finished under.

## 5. How the sweep got to 10_3

All on 1 test case unless noted.

| Config | adam | lbfgs | max_iter | patience | n_samples | batch | loss | Note |
|---|---|---|---|---|---|---|---|---|
| `test_hybrid_norm_10_1` | 50 | 100 | 10 | 10 | 1M | 300k | l1 | first attempt, too long |
| `test_hybrid_norm_10_2` | 200 | 20 | 10 | 5 | 1M | 1M | l1 | Adam-heavy |
| **`test_hybrid_norm_10_3`** | **10** | **50** | **10** | **5** | **1M** | **1M** | **l1** | **chosen** |
| `..._10_3_full_dataset` | same | | | | | | | promoted to 139 cases |
| `..._10_3_14_examples` | same | | | | | | | 14-case subset |

Pre-hybrid LBFGS-only exploration is in `test_lbfgs_optimizer_*.json` (about a dozen, ending
at `_v5`), superseded.

Three L2 variants were created after 10_3 was promoted and **never evaluated**:
`_l2` (adam 50 / lbfgs 100 / max_iter 5), `_l2_1_maxiter` (adam 10 / lbfgs 100 / max_iter 1),
`_l2_2_matix_small_batch` (adam 10 / lbfgs 100 / max_iter 5, n_samples 20k). Plus
`baseline_config_l2*.json`. No memo calls any of them best.

Note for the `max_iter: 1` variant: the loop runs one extra no-gradient forward per LBFGS
step to record the loss, so `max_iter=1` doubles the per-step decoder cost rather than
tenth-ing it. Measured: 11 forward passes per LBFGS step at `max_iter=10`, against 1 for an
Adam step.

## 6. Metrics

wandb project `bone-modeling/nsm-optimization-testing`, per-tissue ASSD:
`mean_assd_{bone,cartilage,med_meniscus,lat_meniscus}`, `mean_assd`,
`mean_reconstruction_time`, plus `std_`/`median_`/`n_samples_` variants.

**Caution on the loss curves in wandb.** For an LBFGS step, `optimizer.step(closure)`
returns torch's `orig_loss` — the loss at the step's *initial* parameters. The logged
`total_loss`/`l1_loss` therefore lags one full outer step, while `recon_loss` beside it is
recomputed after the step. Verified on a full-cloud fit, where the draw is deterministic so the two are
directly comparable: every log line's `Loss=` equals the previous line's `Recon=`. Adam is
unaffected — it evaluates the loss once per step.

## 7. What is measured about the optimizers

20 synthetic fitted-latent problems: a frozen random MLP decoder, 16-d latent, 1500-point
full cloud, L1, held-out median |pred − truth| against noise-free truth. Not a trained NSM
decoder — the direction is credible, the magnitudes are not transferable.

| config | last latent | best latent (`recon_loss`) | decoder evals |
|---|---|---|---|
| Adam `lr=0.01`, 1000 steps | 0.0073 | 0.0048 | 1001 |
| Adam `lr=0.005`, 1000 steps (kneepipeline shape) | 0.529 | 0.529 | 1001 |
| LBFGS `lr=1.0`, 90 steps | 5.62 (worst 1.0e6) | 0.184 | 554 |
| LBFGS `lr=0.005` (what plain `optimizer_name="lbfgs"` gives you) | 0.403 | 0.371 | 991 |
| hybrid Adam 10 + LBFGS 50 (10_3 shape) | 3.24 (worst 2.1e9) | 0.200 | 373 |
| hybrid Adam 200 + LBFGS 20 (10_2 shape) | 3.10 (worst 1.1e18) | 0.104 | 414 |
| **LBFGS + `line_search_fn="strong_wolfe"`** | **0.0000** | **0.0000** | **738** |

Three readings.

**There is no line search.** `line_search_fn` is `None`, torch's default, and the code never
sets it, so each inner iteration takes a fixed step of size `lr` with nothing checking that
it decreased anything. `lbfgs_lr=1.0` is standard *because* it is normally paired with a
Wolfe search; unguarded it diverges.

**`convergence="recon_loss"` is what made the sweep's numbers look reasonable.** It snapshots
the best latent seen and so discards the blow-ups: 5.62 → 0.18 for LBFGS, 3.24 → 0.20 for the
hybrid. The recipe sets it. Nothing in the 2025 notes says the LBFGS phase depends on it, and
a reader copying the optimizer settings without it gets a very different result.

**Adding the line search removes the problem instead of surviving it** — best result under
both convergence modes, and the cheapest of the accurate configurations. That is the change
this plan's **Next** is testing.

## 8. Open questions, in the order they are worth answering

1. **Does `line_search_fn="strong_wolfe"` hold up on the real model?** One run of the six
   configs above on the 139-case set. If it does, it is a one-parameter change that makes
   LBFGS strictly better than Adam on both accuracy and cost, and most of the rest of this
   document stops mattering.
2. **Is the hybrid buying anything at all** once the LBFGS phase is guarded, or is the Adam
   warm-up only there to inflate the latent norm (§4.3)?
3. **`norm_penalty_weight` across orders of magnitude**, and whether `latent_norm=10`
   matches the model's own training latents. Fixed at 100 throughout the sweep and never
   justified.
4. **L1 vs L2**, from §5's three unevaluated variants.
5. **Latent reproducibility and downstream prediction**, which the sweep never measured —
   two fits per case at different seeds, compared on latent distance and on BScore.

The hard-projection path (`use_soft_norm_constraint: false`) is *not* on this list; §4.3
explains why it is close to a duplicate of the soft path at w=100. If it is tried anyway,
note that `step_closure` applies the projection **after** computing the step's loss and
gradients, mutating the parameter under the optimizer between its own inner iterations.

## 9. How to run

```bash
conda activate comak
cd /dataNAS/people/aagatti/projects/nsm_recon_optimization_testing

python test_optimization/test_optimization.py baseline_config_full_dataset.json
python test_optimization/test_optimization.py test_hybrid_norm_10_3_full_dataset.json
```

Before the second one runs against current NSM, delete `latent_optimizer_name` (§4.2) and
the three non-parameters in §4.1 from the config, or it raises.
