# Known issues

Two sections, and the difference matters:

- **[Open](#open)** — reproduced, user-visible, **not fixed**. What you need in order to
  interpret results from the code as it stands today.
- **[History](#history)** — was wrong, silently changed results, **now fixed**. What you
  need in order to interpret a run you already have on disk.

**Why this file exists.** For research code, "which of my results are affected by this
bug?" has to be answerable years later, by someone who was not there. A fix commit and a
code comment cannot answer it. Every entry should let a reader determine, for a run they
have, whether it is affected and what to do about it.

**When an entry moves Open → History.** When the fix lands *and* it silently changed
numerical output for inputs that previously ran without error. A bug that always crashed
closes by deletion instead — nobody has results from it. See `CLAUDE.md`
§ Numerical-behaviour changes.

**Open is not the work queue** — GitHub issues are. An entry here says "this is true of
the library today"; an issue says "we intend to fix it". Most Open entries have both, and
neither replaces the other: issues live on GitHub, and this file is what survives in the
repo.

---

# Open

Every entry below was **executed**, not inferred, and each names the test that pins it.
Ordered by file and function rather than by severity, so they surface as you open the
relevant code.

Each entry says what is wrong and how to tell whether it affects you. **How to fix it is in
the issue** — that is the split: this file is what survives in the repo, the issue is the
queue.

**Every row links to its entry, and a test asserts the two are the same set**
(`testing/NSM/test_docs_references.py`). They were not, until plan §8.0.N: nine rows
against twelve entries, with three rows naming no entry and six entries appearing in no
row. An index maintained by hand goes stale the same way a transcribed number does, and
this file's whole promise — "which of my runs are affected, answerable in 2031" — is worth
more than that.

| Defect | Severity | Issue |
|---|---|---|
| [Sigma coordinate space depends on `scale_jointly`](#sigma-sampling-coordinate-space-depends-on-scale_jointly) | **High** — ~100× over/under-sampling | [#3](https://github.com/gattia/nsm/issues/3) |
| [A `None` surface cannot build](#a-none-surface-cannot-build) | Medium — advertised feature, unusable | [#67](https://github.com/gattia/nsm/issues/67) |
| [`center_pts` and `norm_pts` do not select which normalization happens](#center_pts-and-norm_pts-do-not-select-which-normalization-happens) | Medium — silent, and the shipped config asks for the half it does not get | [#20](https://github.com/gattia/nsm/issues/20) |
| [Shipped model configs predate the `Target` requirement](#shipped-model-configs-predate-the-target-requirement-and-cannot-be-trained-from) | Medium — refused at train, inference unaffected | *none — the migration message is the fix, see below* |
| [Shipped model configs omit two required architecture keys](#shipped-model-configs-omit-two-keys-load_model-requires-so-it-refuses-them) | Medium — refused at load, not silent | [#26](https://github.com/gattia/nsm/issues/26), [#45](https://github.com/gattia/nsm/issues/45) |
| [`sample_difficulty_lx` is shipped and read by nothing supported](#sample_difficulty_lx-is-shipped-and-read-by-nothing-supported) | Medium — four config keys that do nothing | [#18](https://github.com/gattia/nsm/issues/18) |
| [Hybrid / LBFGS reconstruction is unvalidated](#hybrid--lbfgs-reconstruction-is-unvalidated-on-current-nsm) | Medium — runs, unmeasured; production uses Adam | *none — see below* |
| [`F401` is project-ignored, so unused imports never appear](#f401-is-project-ignored-so-unused-imports-do-not-appear-in-make-lint) | Low — tooling, not behaviour | *none — a judgement call, see below* |
| [Latent gradients are summed over query points](#latent-gradients-are-summed-over-query-points-so-the-reg-balance-depends-on-n) | Medium — the reg balance moves with N | *none — a convention change, see below* |
| [`enforce_minmax` clamps predictions](#enforce_minmax-clamps-the-prediction-not-just-the-target) | Medium — config semantics | *none — a docs/design call, see below* |
| [`grad_clip` clips the model only, never the latent codes](#grad_clip-clips-the-model-only-never-the-latent-codes) | Medium — a global-sounding knob that is not | *none — an experiment first, see below* |
| [The bare `compare_cart_thickness` scores femoral regions whatever the model is](#the-bare-compare_cart_thickness-scores-femoral-regions-whatever-the-model-is) | Low — NaN, not a wrong number | *none — documented at the constant, see below* |
| [`Pool` deadlocks after an in-process build](#pool-deadlocks-after-an-in-process-build) | Low — hangs, does not corrupt | [#25](https://github.com/gattia/nsm/issues/25) |

**Two rows left this table on 2026-08-30** rather than gaining an entry: "Parameters
accepted and never read" and "`xyz_in_all` accepted and never read", both citing
[#20](https://github.com/gattia/nsm/issues/20). #20 is **closed** — §8.0.H swept `models/` — and `xyz_in_all` is no longer a
silent no-op: `Decoder(xyz_in_all=True)` raises `TypeError` naming it, measured. A falsy
value is still ignored, deliberately and in writing, because that is what every NSM-owned
config ships and it asked for nothing. The defect *class* has not gone anywhere and is
tracked where a class belongs, in `ARCHITECTURE.md` §7.

---

## `datasets/sdf_dataset.py`

### Sigma sampling coordinate space depends on `scale_jointly`

`SDFSamples` interprets `sigma_near` / `sigma_far` in two different coordinate spaces:

- `scale_jointly=False` — sigma is applied **after** per-mesh normalization, so it is in
  normalized `[-1, 1]` cube units (typical values 0.01–0.1).
- `scale_jointly=True` — sigma is applied **before** normalization, so it is in original
  mesh units, e.g. mm (typical values 0.5–5.0).

The same number means two very different things, and using values tuned for one mode in the
other changes effective sampling density by roughly **100×**, silently.

**How to tell whether a run is affected:** check `scale_jointly` and the sigma values
together in `model_params_config.json`. Small sigmas (< 0.2) with `scale_jointly=True`, or
large sigmas (> 0.5) with `scale_jointly=False`, indicate a probable mismatch.

*Planned fix:* an explicit `sigma_coordinate_space` parameter, standardizing on original
coordinate space, with a migration guard of the same shape as History §1's. Written up in
`.claude/plans/BREAKING_CHANGE_PROPOSAL.md` and
`.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md`, scheduled into
`.claude/plans/NSM_CODE_HEALTH_REFACTOR.md` §8. Tracked as
[#3](https://github.com/gattia/nsm/issues/3), open since Sept 2025.

### A `None` surface cannot build

`MultiSurfaceSDFSamples` accepts `None` in a subject's path list — a missing structure,
the fdfe902 feature — but `get_sample_data_dict` preallocates `data["xyz"]` with
`sum(n_pts_)` rows per combo while `read_meshes_get_sampled_pts` returns points only for
the non-None surfaces, so the first buffer write raises `RuntimeError` (expanded-size
mismatch). The feature has never worked through the dataset class; the downstream
NaN-column handling (`remove_overlapping_points`, `sdf_pos_neg_idx`) is reachable only by
direct call.

*Fix:* [#67](https://github.com/gattia/nsm/issues/67). *Pinned by:*
`test_dataset_cache.TestEmptySignedSamples::test_a_none_surface_subject_must_build`.

### `center_pts` and `norm_pts` do not select which normalization happens

Together they decide *whether* points are normalized: the block is skipped only if both are
false. They do **not** decide *which* operation runs — if either is set, points are both
centered and scaled to unit max-radius.

So `center_pts: true, norm_pts: false` — the shipped configuration — asks for centering
without scaling and gets both. No run has ever been centered-but-unscaled.

This is what is left of #20 in this file after the function-level half was fixed: the
arguments `get_pts_center_and_scale` accepted and ignored are gone (removed rather than
honoured, because honouring them would have switched scaling off on every default run), but
the two config keys still read as independent switches and are not.

*Fix:* [#20](https://github.com/gattia/nsm/issues/20) — **read its traps first.** Any fix
that makes `norm_pts` authoritative changes the coordinate frame of every dataset and
checkpoint ever produced, and needs a migration, not a patch. *Pinned by:*
`test_dataset_cache.TestPointCenteringAndScaling::test_centering_and_scaling_still_happen_unconditionally`.

### `Pool` deadlocks after an in-process build

A `multiprocessing=True` build hangs indefinitely with idle workers (`SDFSamples.__init__`)
**if a
dataset was already built in the same process with `multiprocessing=False`.** Fork-after-VTK.

The trigger is narrower than "a second dataset", and the distinction matters because
`multiprocessing=True` is the constructor default:

| Sequence in one process | Result |
|---|---|
| `multiprocessing=True` → `multiprocessing=True` | **Fine.** Measured 2.6 s then 0.4 s |
| `multiprocessing=False` → `multiprocessing=True` | **Hangs.** First build 5.7 s, second never returns |

So a script that builds a train split in-process and then a val split on the default path
deadlocks, with no message, on the second one. Long-standing rather than new.

**Ruled not-fixed, 2026-08-30 (maintainer, plan §8.0.N).** A `spawn` context would change
behaviour nobody has asked for, and the constraint is cheaper than that: build both splits
the same way, or build each in its own process. The constraint is now stated on the
`multiprocessing=` parameter itself (`SDFSamples.__init__`'s docstring), which is where a
user meets it — this entry is what they find afterwards. [#25](https://github.com/gattia/nsm/issues/25)
stays open as the queue entry.

*Worked around in:* `test_dataset_cache.TestSeedDerivation`, which builds its two datasets
in separate subprocesses.

## Packaging and configuration

### Shipped model configs predate the `Target` requirement and cannot be trained from

Both production model configs — `647_nsm_femur_v0.0.1` and `551_nsm_femur_bone_v0.0.1` —
carry two `LearningRateSchedule` entries with no `Target`. `get_learning_rate_schedules`
raises `ValueError` on either.

**Inference is unaffected**, which is why nobody has hit it: `steps/run_nsm.py` reads those
files only for constructor kwargs and never builds a schedule. What does not work is
resuming or re-training from a shipped model's own config without annotating it first.

The consequence for the codebase is that `NSM/_lr_migration.py`'s delete-when condition —
no configs left that omit `Target` — is objectively unmet, and was not recorded anywhere.
Do not delete that module on the assumption the migration is finished.

**How to tell whether you are affected:** if you load a `model_params_config.json` written
before Aug 2026 and call anything in the training path, you get the `ValueError`, and its
message prints the paste-ready annotation for that run's optimizer. Take it from there
rather than hand-writing `Target`, since Adam and `schedule_free_*` migrate to opposite
values.

### Shipped model configs omit two keys `load_model` requires, so it refuses them

Both production model configs — `647_nsm_femur_v0.0.1` and `551_nsm_femur_bone_v0.0.1` —
carry no `padding` and no `conv_activation`. Since Aug 2026 (#26, #45) `load_model`
requires both for a triplanar config, because neither can be recovered from the checkpoint:
`padding` scales query coordinates before they index the feature planes and is not a
learned parameter, and `conv_activation` decides the module layout. So `load_model` on a
shipped checkpoint raises rather than loading.

**This is the refusal working, not a regression.** What it costs is one edit. From v0.3.0
the message names both keys at once and ends with a JSON block that repairs the config:

```json
{
    "padding": 0.1,
    "conv_activation": null
}
```

Both are the values every model trained before Aug 2026 ran at, and both shipped configs
already state `conv_norm_type: "layer"`, so nothing else is needed.

**Inference through the downstream consumer is unaffected**, which is why nobody has hit
it: `kneepipeline`'s `steps/run_nsm.py` builds `TriplanarDecoder(**params)` by hand from
fifteen config keys and never calls `load_model`. What does not work unrepaired is
`examples/load_trained_model.py`, which is the documented way to load a trained model.

*Pinned by:* `testing/NSM/regression/test_shipped_checkpoints.py`, which is skipped unless
`NSM_SHIPPED_MODELS` points at the model directories — the two checkpoints are 275 MB and
260 MB and do not belong in CI. Run against both, it asserts the message names every
missing key, that the repaired config loads strictly, and that `load_model`'s model is
bitwise-identical to the consumer's own construction.

### `sample_difficulty_lx` is shipped and read by nothing supported

`NSM/configs/default_config.json` carries four keys — `sample_difficulty_lx`,
`sample_difficulty_lx_schedule`, `sample_difficulty_lx_cooldown`,
`sample_difficulty_lx_epsilon` — and the only code that reads them is
`train/train_deep_sdf_multi_head.py`, which `SCOPE.md` §2.1 rules unsupported and which
[#51](https://github.com/gattia/nsm/issues/51) says trains only its last decoder, and
`train/deprecated/train_deep_sdf_orig.py`, which §2.2 quarantines.

`train_deep_sdf` — the trainer `SCOPE` supports — reads `sample_difficulty_weight` and
stops. So setting any of the four in a config for a supported run changes nothing and
reports nothing: the inverse-Lx loss weighting they configure is in a file that is not the
trainer.

**How to tell whether it affects you:** it does not affect a *result* — nothing silently
changed, the feature simply never ran. It affects you if you set one of these keys and
believed it did something. Grep your config for `sample_difficulty_lx`; if it is there and
non-null, the run you got is the run you would have got without it.

*Fix:* [#18](https://github.com/gattia/nsm/issues/18) — port the ~12-line inverse-Lx branch out of the quarantined trainer,
under the two conditions `SCOPE.md` §2.2 sets (impossible to enable by accident, documented
at the config key). Scheduled at plan §8.0.P. *Pinned by:*
`test_default_config_sync.test_the_sample_difficulty_lx_keys_are_read_by_nothing_supported`,
which goes red the day the port lands.

### `F401` is project-ignored, so unused imports do not appear in `make lint`

`.flake8`'s `extend-ignore` has carried `F401` since before the Aug 2026 lint work. `make
lint` reports zero violations, and separately there are **44** unused imports it will never
show (`flake8 --extend-ignore="" --select=F401 NSM/ testing/`, re-run 2026-08-27; 54 before
§8.0.J opened `reconstruct/main.py`). "flake8 is at zero" is true and does not mean the
imports are gone.

The command matters and the one recorded here until §8.0.K did not work:
`--extend-select=F401` does not override an `extend-ignore` that already names `F401`, so
it reports **0** and reads like the problem went away. `--extend-ignore=""` clears the
ignore list; `--select` then narrows to the one code.

*Fix:* not filed. Removing the ignore is a judgement call — several are deliberate
re-exports, which is the usual reason `F401` gets ignored wholesale. `reconstruct/main.py`
is the worked case: ten hits there were five dead imports, deleted, and five re-exports the
import-compat test freezes, now carrying `# noqa: F401` and a comment. The two kinds are
tellable apart one file at a time, which is what lifting the ignore would take.

### Hybrid / LBFGS reconstruction is unvalidated on current NSM

`optimizer_name="lbfgs"` and `hybrid_optimizer=True` run, and nothing in production uses
them: both shipped configs and kneepipeline fit with Adam. Three things a caller should
know before relying on either.

**torch's LBFGS here has no line search.** `line_search_fn` is never set, so `lbfgs_lr` is
the raw step length rather than a trust region, and a full quasi-Newton step at
`lbfgs_lr=1.0` diverges routinely — latent norms of 172, 560 and 3444 have been observed.
Setting `latent_norm` is the workaround that has been used; it caps the symptom.

**A subsampled objective is redrawn on every loss evaluation**, and LBFGS evaluates several
times per step, so it optimizes something that moves. Removing the redraw is not the fix —
it is how the fit covers the point cloud, and the full cloud beats every subsampled regime
measured. `reconstruct_latent` warns when it sees the combination and names the sample
count to raise to. Note that `n_samples` is split per surface and capped at each surface's
size, so a budget equal to the cloud size still subsamples when the surfaces are unequal.

**The one multi-case result on record was produced on a code path that no longer exists**
(a triplanar feature cache added and removed the same afternoon, Aug 2025), so it cannot be
reproduced as measured.

*Fix:* not filed. The measurements, the validated configuration and what it would take to
resurrect are in `.claude/plans/HYBRID_OPTIMIZER_REPORT.md`.

*Pinned by:* `test_reconstruct_latent_internals.TestTheDrawIsPerEvaluation` and
`TestTheLbfgsParametersAreReadOnBothPaths`.

## `reconstruct/cartilage_func.py`

### The bare `compare_cart_thickness` scores femoral regions whatever the model is

`cart_regions` defaults to `CART_REGIONS`, which is `CART_REGIONS_DICT["femur"]` — the
five femoral subregions, 11–15. `DICT_VALIDATION_FUNCS` in `train_deep_sdf.py` exposes the bare
`compare_cart_thickness` under its own name, so a config whose `recon_val_func_name` is
`"compare_cart_thickness"` for a tibia or patella model takes that default and scores
**NaN for every region**, with pymskt's `UserWarning` per region as the only signal.
`get_mean_errors` then averages NaN into the logged metric.

**How to tell whether a run is affected:** if `recon_val_func_name` is
`"compare_cart_thickness"` and the model is not a femur model, every `cart_thick_*` metric
in the run is NaN. The three joint-named wrappers pick the right set and are unaffected.

*Not fixed, deliberately (plan §8.0.N′):* refusing a region set the original's labels do
not contain would be a new design decision about what a validation function may assume,
and the wrappers already exist for the case. The default is documented at the constant and
in `compare_cart_thickness`'s docstring, which is where someone choosing a
`recon_val_func_name` is reading.

---

## `models/triplanar.py`

### Latent gradients are summed over query points, so the reg balance depends on N

When a latent is optimized (reconstruction fitting, and the training embedding), the
gradient it receives from the data term is **summed** over the N query points — 10× the
points, 10× the pull — while the latent-regularization term does not scale with N.
Measured: 10.00× at N=10, 1000.00× at N=1000, **identically** on both decoder
interfaces (`triplanar.UniqueConsecutive` and `triplanar.FastUnique`), so it is a
long-standing convention, not a regression. Details: ARCHITECTURE §6.

No shipped run is affected — `l2reg_recon: false` in both production configs, so the
imbalance multiplies a term that is zero.

**Why this entry exists (maintainer, 2026-08-22): to be revisited, deliberately.** The
maintainer reports latent regularization was historically a pain to tune and was
abandoned — consistent with the effective weight being silently divided by N
(thousands), so nominal weights would have felt inert. That is a hypothesis, not a
finding. The experiment when someone picks this up: enable `l2reg` with the weight
scaled by ~N versus nominal, compare fit quality and fitted-latent norms (see also
`NSM_TRAINING_IDEAS.md` Idea 4, the norm-saturation gap). Any change to the convention
rescales every training and reconstruction run and needs a § History entry plus a
Phase-A-style migration.

## `train/train_deep_sdf.py`

### `enforce_minmax` clamps the prediction, not just the target

`train_epoch` clamps `pred_sdf` as well as the target, and `torch.clamp` passes no
gradient outside its bounds. Every sample predicted outside ±`clamp_dist` therefore
contributes **exactly zero gradient**, however wrong it is. Inherited from the original
DeepSDF loss, which clamps both sides too — standard practice, not something NSM added.

**The three regimes** (the first is why the clamp exists; the third is the hazard):

- *Prediction and truth both beyond ±δ, same side* — loss 0, gradient 0. Intended:
  this is how DeepSDF concentrates capacity near the surface instead of fitting exact
  far-field distances.
- *Prediction inside the band* — normal gradient toward the clamped target.
- *Prediction outside the band while the truth is inside it* — the model is badly
  wrong about a near-surface point: **nonzero loss, zero gradient**. The sample shows
  up in the logged loss and contributes nothing to learning (a loss curve can sit high
  while nothing moves the samples that keep it there); it recovers only via
  generalization from live samples. The subject's **latent code** gets its gradient
  through the same clamped path, so a subject whose samples are mostly dead also has a
  mostly-frozen latent that epoch.

**Measured.** On a freshly built triplanar decoder, **44.6%** of predictions already fall
outside ±0.1 before the first step. The shipped `default_config.json` uses `clamp_dist: 0.1`;
both ShapeMedKnee configs use `1.0`.

**Why the shipped models are largely immune.** The triplanar output is `tanh`-bounded to
(−1, 1) (measured in § History 16, the `padding` entry), so at the production
`clamp_dist: 1.0`
the prediction-side clamp **never binds** — the tanh acts as a soft clamp whose gradients
shrink smoothly instead of cutting to zero, and the target-side clamp still provides the
intended don't-care-beyond-the-band behaviour. The trap is training from the shipped
default config, whose DeepSDF-inherited `0.1` puts a fresh decoder's samples 44.6% dead
at initialization.

Whether that stalls a given run is configuration-dependent — an earlier claim that it always
does was **false**, and was withdrawn after being run. The defect is that the name and the
docs describe a target transform while the behaviour is a training-dynamics knob. This is a
documentation-or-decision call, not a bug fix, so it has **no issue** — it belongs with the
config work in `SCOPE.md` §2.2. **Deliberately kept open** (maintainer, 2026-08-24): this
has real effects and should stay in view until the decision is made.

**The untested alternative:** clamp only the *target* — `|pred − clamp(gt, δ)|` — which
keeps gradient everywhere at the cost of forcing far-field predictions to sit exactly at
±δ (a different learned function: plateaus at the band edge). The experiment — double
clamp vs. target-only clamp vs. tanh-plus-loose-clamp at matched δ — is a named axis of
`NSM_TRAINING_IDEAS.md` Idea 11, judged with the regression harness and Idea 10's
surface-residual metric.
*Pinned by:* `test_training_regression.TestClampedPredictionGradients`.

### `grad_clip` clips the model only, never the latent codes

`train_epoch` hands `torch.nn.utils.clip_grad_norm_` the model's parameter tensors and
nothing else; the latent `nn.Embedding` is a first-class optimizer param group and is
never clipped. Verified by wrapping the clip call on a real epoch: called on the 21 model
tensors only. A user setting a knob named `grad_clip` will reasonably assume it is
global. Clipping the latents now would silently change the numerics of every run that
sets `grad_clip`, so this is documented rather than fixed.

**Revisit (maintainer, 2026-08-22):** worth an experiment rather than a permanent
shrug — train with the clip applied to both groups (or one global clip) and compare
stability and latent-norm trajectories against the current behaviour. If adopted, it
changes numerics for every run that sets `grad_clip` → § History entry.

## Upstream

Dependency bugs that reach an NSM user.

- **[pymskt#56](https://github.com/gattia/pymskt/issues/56)** — `rand_pts_around_surface`
  raises a broadcast error under `surface_method="bluenoise"`, which is its **default**.
  NSM is not currently hit because every call site passes `surface_method="random"`
  explicitly (`sdf_dataset.py`), so this is a trap for anyone who changes that, not a live
  defect. Open upstream.
- **`mskt>=0.1.21` is required**, not optional — see History §3. An older install raises
  `TypeError` on the first sampling call rather than silently reverting to unseeded draws.

---

# History

## 1. Learning-rate schedules applied swapped (model ↔ latent codes)

| | |
|---|---|
| **Affects** | May 2023 → Aug 2026 |
| **Affected optimizers** | `Adam`, `AdamW` |
| **Unaffected optimizers** | `schedule_free_AdamW`, `schedule_free_SGD` |
| **Severity** | Silent — wrong numerics, no error, no warning |
| **Fixed in** | Two PRs, Aug 2026 — dating a checkout needs the **second**: PR #9 (`fix-lr-schedule-mapping`) mapped schedules by param-group name; PR #10 (`lr-schedule-target-key`) landed the `Target` contract this entry documents (the migration guard, the `{target: schedule}` dict, the swapped shipped defaults) |
| **Reported by** | Dr. Katherine Wolcott, Florida Museum of Natural History / BioVision Lab, 2026-07-10 |

### What was wrong

`get_optimizer()` built optimizer param groups in the order `[latent, model...]`, correctly
assigning `lr_schedules[1]` to the latent codes and `lr_schedules[0]` to the model. But
`adjust_learning_rate()`, called at the top of every epoch, reassigned them **by position**:

```python
for i, param_group in enumerate(optimizer.param_groups):
    param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
```

Group 0 is the latents, so the latents received `lr_schedules[0]` — the entry intended for
the model — and the model received `lr_schedules[1]`.

**Net effect: 100% of every affected run used the swapped mapping.** Not "most of" it.
`get_optimizer()` does set the intended learning rates at construction, but the epoch loop
is `range(resume_epoch + 1, n_epochs + 1)` — it starts at 1, never 0 — and
`adjust_learning_rate()` is called at the *top* of `train_epoch`. So the intended values
are overwritten before the first `optimizer.step()` and never influence a single weight
update.

The mismatch dates to a 2023 refactor that made the optimizer able to loop over multiple
models, which moved the latent codes to the front of the param-group list without updating
the positional assignment in `adjust_learning_rate`.

### schedule-free runs: not hit by the bug, but arguably hurt worse

The training loop skips the LR adjustment entirely for those optimizers:

```python
if not ("schedule_free" in config["optimizer"]):
    adjust_learning_rate(config["lr_schedules"], optimizer, epoch)
```

So they kept `get_optimizer()`'s assignment — entry 0 to the model, entry 1 to the latents
— for the whole run. In the narrow sense they were never mis-mapped.

**In practice this made them worse, not better.** Every config in this project was written
and tuned against the Adam/AdamW path, where entry 0 was in effect the latent LR. Running
that same file with a `schedule_free_*` optimizer applied the values the other way round.
The same config meant opposite things depending on which optimizer you chose, and only the
Adam reading matched how the numbers were picked.

Nothing decayed either, since `adjust_learning_rate` is where decay is applied, so whatever
the epoch-0 values were, they held for the entire run.

Worked through with the ShapeMedKnee_2024 values (entry 0 `0.005`, entry 1 `0.0001`):

| | latent LR | model LR |
|---|---|---|
| `AdamW` (as tuned) | `5e-3` → `1.5e-5` | `1e-4` → `1e-6` |
| `schedule_free_AdamW` | **`1e-4`, constant** | **`5e-3`, constant** |

The decoder trains at 50× the rate it was tuned for, flat, for the whole run. **If a
`schedule_free_*` run appeared not to work, this is a candidate explanation, and it is not
a property of schedule-free optimizers.** Re-run it with the values annotated the way they
were tuned before concluding anything about the method.

Consequence for migration: for a `schedule_free_*` config, the annotation that reproduces
the historical run is entry 0 → `model`, entry 1 → `latent` — the opposite of the Adam
case — but reproducing that run faithfully is often *not* what you want. The migration
error prints this caution when it sees a schedule-free optimizer.

### How to tell whether one of your runs is affected

1. Check `model_params_config.json` in the experiment directory.
2. If `"optimizer"` is `"Adam"` or `"AdamW"`, and the run predates the fix → **affected**.
3. If `"optimizer"` starts with `"schedule_free"` → not hit by the runtime bug, but read
   the schedule-free section above before assuming the run was fine.
4. If the `LearningRateSchedule` entries carry `"Target"`, the run was configured after
   the fix and its mapping is explicit.

For an affected run, the learning rates actually used were:

- latent codes ← `LearningRateSchedule[0]`
- model/decoder ← `LearningRateSchedule[1]`

### How to reproduce an affected run under fixed code

Annotate each entry with the group it historically drove. Change nothing else — no
reordering, no edits to any value:

```json
"LearningRateSchedule": [
    {"Target": "latent", ...entry 0 unchanged... },
    {"Target": "model",  ...entry 1 unchanged... }
]
```

That is the correct annotation for `Adam`/`AdamW`. **`schedule_free_*` is the opposite** —
`"model"` on entry 0, `"latent"` on entry 1 — because those runs skipped
`adjust_learning_rate()` and kept `get_optimizer()`'s own assignment. Getting this
backwards inverts the run.

You do not have to work it out. Run the config; the error prints the paste-ready block for
your optimizer.

Verified by `TestHistoricalEquivalence` in `testing/NSM/test_lr_schedules.py`, which
asserts equality against a reimplementation of the pre-fix mapping across a range of
epochs, using the real ShapeMedKnee_2024 schedules.

For new runs, set `Target` to whatever each entry is meant to drive. Order is ignored.

### Migration guard

A pre-fix config run on fixed code would otherwise train with a different mapping than it
did historically, with no error. So **any** config with an entry missing `Target` now
raises — including a half-annotated one, which is the case most likely to slip through a
glance. This applies to every optimizer; `schedule_free_*` was never affected by the
runtime bug, but its construction-time mapping was positional too, so it gets the same
rule rather than an exemption.

### Worked example: ShapeMedKnee_2024

The production knee model. Its two entries differ in every field, not just `Initial`:

| | `Initial` | `Interval` | `Factor` |
|---|---|---|---|
| Entry 0 | `0.005` | `16.67` | `0.952` |
| Entry 1 | `0.0001` | `1000` | `0.1` |

Optimizer is `AdamW`, so the run actually trained with **latents on entry 0** — starting at
`5e-3` and decaying smoothly to `1.5e-5` — and the **decoder on entry 1**, flat at `1e-4`
for the first 1000 epochs with `×0.1` steps at 1000 and 2000.

The two curves are 50× apart at epoch 0 and have completely different shapes. Annotating
this config the wrong way round does not perturb the run, it inverts it. This is the
canonical example of why the guard raises instead of warning.

### What this did to the shipped defaults

Before the fix, the shipped default resolved to latent `0.0005` / model `0.001` at
runtime, while the config read as the reverse.

This fix swaps the two entries in the shipped default configs, which **preserves that
historical effective behaviour** (model `0.001`, latent `0.0005`) rather than making the
config mean what it previously appeared to say. That is a deliberate choice for continuity
with the tuned production models — see the open action below. It is not an endorsement of
those values.

Note this is only about the shipped defaults, whose two entries were identical apart from
`Initial`. For a real training config the swap carries the whole schedule — `Type`,
`Interval` and `Factor` as well — so reasoning about `Initial` alone will mislead you. See
the ShapeMedKnee_2024 example above.

### Scientific consequence

Hyperparameter searches run before the fix were optimizing under the swapped mapping, so
the values chosen were optimal *for that mapping*. Two consequences:

- The chosen LR values are not necessarily optimal under the fixed mapping.
- Because both schedules were searched jointly, the resulting models are not invalid —
  they were trained with a self-consistent, tuned pair of learning rates. The labels on
  those two numbers were wrong, not the numbers themselves.

**Open action:** re-tune learning rates under the fixed mapping and compare against the
current production models before assuming either is better. Not yet done.

### Related

- `NSM/utils.py` — `get_optimizer`, `adjust_learning_rate`, `resolve_schedule_targets`
- `NSM/_lr_migration.py` — the migration error text. Not permanent API; delete the file
  once no config still in use predates the `Target` key.
- `testing/NSM/test_lr_schedules.py` — regression and equivalence tests
- `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md` §4 — this fix as the migration template

---

## 2. `train_deep_sdf_multi_head` optimizes only the last model

| | |
|---|---|
| **Affects** | All runs through this entry point |
| **Severity** | Silent — training appears to proceed normally |
| **Status** | Deprecated Aug 2026, not fixed |

`NSM/train/train_deep_sdf_multi_head.py` contains:

```python
for model in models:
    model = model.to(config["device"])   # rebinding is pointless, but .to() is in-place

optimizer = get_optimizer(model, ...)     # `model` is the LAST model only
```

The device move itself is fine — `nn.Module.to()` mutates in place and returns `self`, so
every model does reach the device despite the pointless rebinding. The defect is the line
below it: the optimizer is built from the leaked loop variable rather than from `models`,
so **only the last decoder in `models` ever receives gradient updates**. The others stay
at initialization.

It also cannot complete one epoch on the shipped `default_config.json`: a
non-short-circuit `&` between two membership tests raises `KeyError:
'surface_weighting'` before the epoch ends, so any run that would have hit the
silent-training defect crashes first on the default config.

The module now emits a `DeprecationWarning`. Use `NSM.train.train_deep_sdf` with
`objects_per_decoder > 1` instead. Whether to repair or delete this file is a Phase 0
decision in `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md`.

---

## 3. Sampling was never seeded — no run before Aug 2026 is reproducible

| | |
|---|---|
| **Affects** | Every training run up to Aug 2026, including the shipped ShapeMedKnee models |
| **Severity** | Silent — no error, and a warm cache makes it look like it works |
| **Fixed in** | Aug 2026, this branch. Requires `mskt>=0.1.21` |

### What was wrong

`SDFSamples(random_seed=...)` was documented as "Random seed". It seeded nothing. The value
was stored on the instance and appended to the cache key, and that was all it did — NSM
called no seeding function anywhere.

Two of the three sampling paths were unreachable even by a caller willing to seed `numpy`
globally, because `pymskt.Mesh.rand_pts_around_surface` had two independent draws that
bypassed the legacy global stream:

- the base surface points, via `pcu.sample_mesh_random(v, f, n, random_seed=0)` — and pcu
  documents `random_seed=0` as *"use the current time"*, not "seed 0";
- the perturbation offsets, via `np.random.default_rng()` with no argument, which seeds
  itself from OS entropy.

So `sigma_near`/`sigma_far` sampling — the path every production config uses — drew fresh
points on every call regardless of what the caller did. Only the uniform path
(`sigma` of `None`) responded to `np.random.seed`, and nothing in NSM called that either.

### Why nobody noticed

**A warm cache hides it perfectly.** The seed *did* change the cache key, so two runs with
the same `random_seed` resolved to the same `.npz` and the second reused the first's data.
That is a cache hit behaving correctly. Point the two runs at different cache directories —
or run the second on a machine that has never seen the first — and they diverge.

### A second consequence: subjects were correlated, not merely random

`multiprocessing=True, n_processes=2` is the constructor default, and `Pool` forks, so every
worker inherited one copy of the legacy global `numpy` state. On the uniform path this made
identical subjects come out **bit-identical**, with a correlation pattern that depended on
`n_processes`. Measured on the pre-fix tree: three subjects sampled under
`multiprocessing=True` versus `False` matched on none of them; after the fix, all three.

### How to tell whether one of your runs is affected

Every run trained before Aug 2026 is affected. There is no configuration that escapes it —
`random_seed` was inert whether you set it or not.

What that means in practice is narrower than it sounds:

- **Your model weights are fine.** The data was drawn from the right distribution; it just
  cannot be drawn again.
- **The `.npz` cache is the only record of what a run actually trained on.** If you still
  have the cache directory, the run is reproducible from it — the key is unchanged by this
  fix, so an existing cache still hits.
- **If the cache is gone, the exact training data cannot be regenerated.** Not from the
  config, not from the seed, not from anything. This is the part worth knowing before you
  delete a cache directory to save disk.

### What changed numerically

- **`random_seed=None` (the default): nothing.** Verified bit-for-bit, on both the sampled
  arrays and the cache keys. An unseeded call still draws from the legacy global stream
  precisely so this stays true — `np.random.default_rng(s)` and
  `np.random.seed(s); np.random.uniform(...)` produce different numbers, and switching the
  unseeded path to a `Generator` would have silently moved every existing result.
- **`random_seed` set, warm cache: nothing.** The key is unchanged, so the cached file
  still hits.
- **`random_seed` set, cold cache: the data is now deterministic** instead of freshly
  random. It will not match whatever that config produced before, because nothing did.

### Running fixed code against an older pymskt

It raises. `rand_pts_around_surface` gained `seed` in `mskt` 0.1.21 and takes no `**kwargs`,
so an older install fails with `TypeError` on the first sampling call rather than silently
reverting to unseeded draws. `requirements.txt` pins `mskt>=0.1.21`; the `TypeError` is the
backstop if something bypasses the pin.

### Related

The seed is derived per (subject, sampling pass, surface) rather than used directly — one
shared seed would hand the near- and far-surface passes the same base surface points, and
give bone and cartilage the same offset vectors. The subject component is keyed on **mesh
content**, not on the mesh path or its position in `list_mesh_paths`, so neither moving the
files nor reordering the list changes what a subject samples. See `derive_seed` in
`NSM/datasets/sdf_dataset.py` and `TestSeedDerivation` in
`testing/NSM/regression/test_dataset_cache.py`.

Upstream half: [gattia/pymskt#54](https://github.com/gattia/pymskt/issues/54), fixed in
[#55](https://github.com/gattia/pymskt/pull/55), released as 0.1.21.

---

## 4. `weight_decay` was silently ignored under `Adam`

| | |
|---|---|
| **Affects** | Every `optimizer: "Adam"` run that set `weight_decay`, from the initial commit → Aug 2026 |
| **Unaffected** | `AdamW` and `schedule_free_AdamW` runs — **both shipped ShapeMedKnee configs use `AdamW`** |
| **Severity** | Silent — the config said one thing, training did another |
| **Fixed in** | `wave-1`, Aug 2026 ([#47](https://github.com/gattia/nsm/issues/47)) |

### What was wrong

`get_optimizer` passed `weight_decay` to `AdamW` and `schedule_free_AdamW` but built
`torch.optim.Adam(list_params)` bare. `train_deep_sdf` forwards `config["weight_decay"]`
unconditionally, so an `Adam` config trained with **zero** weight decay at any configured
value. The regression harness's own config (`Adam`, `weight_decay: 1e-4`) ran through the
bug, so its committed training baselines described un-decayed training and were
regenerated with the fix.

### How to tell whether one of your runs is affected

`optimizer == "Adam"` and `weight_decay` set nonzero in its `model_params_config.json`
→ that run had no weight decay. The weights themselves don't record it; the config plus
the date is the test.

### Magnitude, and how to reproduce old behaviour

On the 8-epoch CPU harness at `weight_decay: 1e-4`, the loss trajectory diverges from the
un-decayed path by ~0.03% at epoch 1, growing to ~3% by epoch 6 — a slow drift, not a
jump. Reconstruction baselines did not move (`reconstruct_latent` builds its own
optimizer and never went through `get_optimizer`). To reproduce an affected run exactly
under fixed code, set `weight_decay: 0`.

---

## 5. Above two surfaces, overlap removal and multi-decoder reconstruction indexed wrong

| | |
|---|---|
| **Affects** | `MultiSurfaceSDFSamples` training runs with **3+ surfaces**; reconstructions with **2+ decoders** |
| **Unaffected** | Every two-surface, single-decoder run — **the shipped configuration** (both ShapeMedKnee models; `kneepipeline` passes one decoder) |
| **Severity** | Silent — no error, plausible outputs |
| **Fixed in** | `wave-1`, Aug 2026 ([#44](https://github.com/gattia/nsm/issues/44)) |

### What was wrong

Two instances of the same shape — positional bookkeeping that is only correct at the
shipped size:

- **`remove_overlapping_points`** tested a sign **sum** (`total == -2`), which equals
  "inside two or more surfaces" only at exactly two surfaces. Enumerated by execution
  over every sign pattern: at 3 or 5 surfaces it removed **nothing**; at 4 it removed
  only the inside-3-of-4 patterns, missing all-4-inside and 2-of-4-inside. It never
  wrongly removed a point, so affected datasets **retained** overlap points they should
  have dropped. Now a count: inside (strictly negative SDF) of two or more.
- **`reconstruct_latent`** indexed the flat `sdf_gt` list per decoder without a running
  offset: the single-output branch used the decoder index, and the multi-output branch
  restarted at 0 for every decoder — so with two 2-surface decoders, the second decoder
  was scored against the **first decoder's** surfaces (demonstrated: all-NaN ground
  truth for surfaces 2 and 3 left the loss bit-identical). Now a running surface offset.

### How to tell whether one of your runs is affected

Dataset side: `model_params_config.json` with 3+ surfaces (`objects_per_decoder` ≥ 3, or
a `mesh_names` list of length ≥ 3) trained before the fix kept overlap points.
Reconstruction side: any fit that passed 2+ decoders. Two-surface single-decoder runs
are bit-identical before and after — the committed regression baselines did not move,
and the sign-pattern enumeration at n=2 agrees with the old test on every pattern
(`testing/NSM/datasets/test_remove_overlapping_points.py`).

### Reproducing old behaviour

No compatibility switch — the old selection is a bookkeeping error with no meaning worth
preserving. Check out a pre-fix commit if an affected run must be reproduced exactly.

---

## 6. The uniform sampling cube was asymmetric, and the single-mesh sampler clipped its draws

| | |
|---|---|
| **Affects** | Uniform ("random") training samples: (a) any run with a nonzero `uniform_pts_buffer` — the parameter exists since Jan 2025 (`48c5f60`) and the shipped `default_config.json` sets it to `0.2` — spelled `dataset_uniform_pts_buffer` there until Aug 2026, so a config on disk from before then carries the old name; (b) any **single-surface** run with `norm_pts=True` |
| **Unaffected** | Multi-surface runs at buffer 0 (the arithmetic is exactly zero — the harness baselines did not move); every `norm_pts=False` run, which is what `scale_jointly` requires and **both shipped ShapeMedKnee configs use**; reconstruction unless `get_rand_pts=True` (`kneepipeline` leaves it off) |
| **Severity** | Silent — the samples were drawn from a slightly different region than configured |
| **Fixed in** | `sdf-dataset-fixes`, Aug 2026 ([#40](https://github.com/gattia/nsm/issues/40)) |

### What was wrong

The single- and multi-mesh samplers carried private copies of the buffered-cube
arithmetic, and both rebound `mins` before `maxs` read it:

```python
mins = mins - uniform_pts_buffer / 2 * (maxs - mins)
maxs = maxs + uniform_pts_buffer / 2 * (maxs - mins)   # (maxs - mins) has already grown
```

so a nonzero buffer grew the cube more above than below and moved its centre up. At the
shipped `0.2` on a normalized object the cube was `[-1.200, +1.220]` per axis instead of
`±1.200` — the top face 1% of the span too far out.

Separately, only the single-mesh copy clipped its draws — to `±1` originally, widened to
`±(1 + buffer/2)` by `48c5f60` — under `norm_pts=True`. The clip piled the truncated
samples onto the cube faces, and it caught the *near-surface* Gaussian draws too:
measured on a normalized synthetic bone, 0.6% of samples at `sigma=0.01` and 2.9% at
`sigma=0.03` had a coordinate beyond `±1` and were moved onto the faces. The multi-mesh
sampler never clipped, so with `uniform_pts_buffer=0.5, norm_pts=True` the two spanned
`±1.25` (clipped) versus `-1.50/+1.56` (asymmetric).

Both now share one helper, `get_buffered_cube_mins_maxs` — symmetric, centre preserved,
no clipping in either sampler.

### How to tell whether one of your runs is affected

Check the run's dataset settings: a nonzero `uniform_pts_buffer` (spelled
`dataset_uniform_pts_buffer` in configs written before Aug 2026), or a single-surface
dataset with
`norm_pts: true`, built before the fix → the uniform samples (and, under the clip, some
near-surface samples) came from the old region.

**The cache will not tell you, and it will not heal itself:** `uniform_pts_buffer` is not
in the cache key ([#19](https://github.com/gattia/nsm/issues/19)), so a post-fix run
pointed at a pre-fix cache silently reuses the old points. Delete the affected `.npz`
files to resample.

### Reproducing old behaviour

No compatibility switch — the asymmetry was a bookkeeping error, not a semantic option.
Check out a pre-fix commit, or reuse the pre-fix cache files (see above), if an affected
run must be reproduced exactly.

*Pinned by:* `test_dataset_cache.TestUniformSamplingCube`.

---

## 7. `include_surf_in_pts` on the multi reader appended another surface's points

| | |
|---|---|
| **Affects** | `read_meshes_get_sampled_pts` calls with `include_surf_in_pts=True` **and centering on** (`center_pts` or `norm_pts` True) and all-numeric sigmas — the only configuration of the flag that ran to completion. Via `reconstruct_mesh`, that is a multi-object reconstruction with `get_rand_pts=True` on a `scale_jointly=False` model. Aug 2023 (`5188417`) → Aug 2026 |
| **Unaffected** | The flag off (its default; the dataset classes never pass it, so **no training data is affected**); both shipped ShapeMedKnee configs (`get_rand_pts_recon: false`); `scale_jointly=True` models (centering off → `UnboundLocalError`, always crashed); any `None` sigma (`ValueError`, always crashed); the single reader, whose append block was and is correct |
| **Severity** | Silent — wrong points in the fitting set, with a plausible link to "enabling it never helped" |
| **Fixed in** | `sdf-reader-internals`, Aug 2026 ([#17](https://github.com/gattia/nsm/issues/17)) |

### What was wrong

The per-surface sampling loop appended a name it never bound:

```python
for new_pts_idx, new_mesh_ in enumerate(new_meshes):   # binds new_pts_idx, new_mesh_
    ...
    if include_surf_in_pts is True:
        rand_pts_ = np.concatenate([rand_pts_, new_pts_], axis=0)   # new_pts_ leaked
```

`new_pts_` held whatever an earlier section left in scope. On the only configuration
that ran — centering on, numeric sigmas — that was the centering loop's leftover
binding: the **last** surface's **pre-normalization** vertices, appended once per
surface. Wrong surface and wrong coordinate frame: an option whose purpose is "give
the fit this surface's own points to pull against" instead added copies of another
surface's points lying far outside the normalized domain. Measured on a two-sphere
subject (2026-08-23): 1,000 requested points came back as 1,580 — `1000 + 2×290`,
the last surface's vertex count twice — where correct is 1,820 (`1000 + 530 + 290`).

Every other configuration crashed before returning (`UnboundLocalError` with
centering off; `ValueError` with any `None` sigma, where the leaked name held a
list), so those paths affect no results. The fix appends `new_pts[new_pts_idx]` —
each surface's own, normalized vertices — and makes the previously-crashing
configurations work as documented.

### How to tell whether one of your runs is affected

Only reconstructions/fits, never training. Check the call: multi-surface, the flag on
(via `reconstruct_mesh`: `get_rand_pts=True`), and a `scale_jointly=False` model. In a
saved fitting set the signature is the point count: `n_random + n_surfaces × (last
surface's vertex count)` instead of `n_random + Σ(each surface's vertex count)`.

### Reproducing old behaviour

No compatibility switch — a leaked loop variable, not a semantic option. Check out a
pre-fix commit if an affected fit must be reproduced exactly.

*Pinned by:* `test_sampled_pts_readers.TestMultiMeshReader::test_include_surf_in_pts_appends_each_surfaces_own_vertices`
(and its uniform-cube sibling), strict-xfail pins of the correct behaviour until the
fix, plain assertions since.

---

## 8. `norm_penalty_type='barrier'` was NaN outside its range, with an inverted gradient

| | |
|---|---|
| **Affects** | `reconstruct_latent` / `reconstruct_mesh` calls with `latent_norm` set to a `(min, max)` range, `use_soft_norm_constraint=True` (the default) and `norm_penalty_type='barrier'`, at any step where the latent norm was outside the range — which is where every run starts unless `latent_init_std` is chosen to land inside it (the default 0.01 puts a 256-dim latent at norm ~0.16). Aug 2025 (`d421583`) → Aug 2026 |
| **Unaffected** | Production and both shipped ShapeMedKnee configs — `latent_norm` is never set, so the whole soft-constraint path is off; `norm_penalty_type` `'quadratic'`/`'huber'`; runs whose latent norm stayed strictly inside the range |
| **Severity** | Visible but easy to misread — every logged loss read `nan` from the first step, yet the run completed and returned a latent (verified by execution, 2026-08-23) |
| **Fixed in** | `recon-option-values`, Aug 2026 ([#48](https://github.com/gattia/nsm/issues/48)) |

### What was wrong

Below the range, `-log(current_norm - min_norm + eps)` takes a negative argument: its
value is NaN, but its gradient `-1/(negative)` is finite and **positive** — so the
barrier term pushed the norm further *below* the range it was meant to enforce. Above
the range, the `max` term mirrors this. The NaN poisoned every loss readout (`nan + x`
is `nan`) without poisoning the gradients, so optimization continued on the finite,
partly-inverted gradient field to completion.

Outside the open interval the penalty now raises by name at the first step; strictly
inside, nothing changed.

### How to tell whether one of your runs is affected

The option was never in a shipped config, so only hand-written calls qualify. The
signature is `nan` loss lines from the run's first step in a run that nevertheless
finished.

### Reproducing old behaviour

No compatibility switch. Check out a pre-fix commit if an affected fit must be
reproduced exactly.

*Pinned by:* `test_reconstruct_latent.TestBarrierNormPenalty`.

## 9. `reconstruct_mesh` ignored `n_pts_random` and always drew 200,000 points per surface

| | |
|---|---|
| **Affects** | The *fix* changes output, not the bug: any post-fix rerun of a `reconstruct_mesh` / `get_mean_errors` call with `get_rand_pts=True` draws the `n_pts_random` it asked for (default 100,000) instead of the 200,000 per surface every pre-fix run silently used — a different sample set, so a different fitted latent and reconstruction. Pre-fix runs are self-consistent. May 2023 (`09150fe`, the commit that introduced the sampled path — then `GenerativeAnatomy/sdf/reconstruct/main.py`; the readers already took `n_pts=`, so the parameter was born broken and never worked on any day) → Aug 2026 |
| **Unaffected** | Production and both shipped ShapeMedKnee configs — `get_rand_pts_recon: false`, so the sampled path never ran; single-object calls, which crashed on this path before the #15 fix; `get_rand_pts=False` calls, where `n_pts_random` reaches nothing by design |
| **Severity** | Silent — `reconstruct_mesh` forwarded `n_pts_random=` to readers whose parameter is `n_pts=`; their `**kwargs` swallowed it without a warning. Measured (2026-08-23): a request for 200 points over two surfaces yielded 400,688 |
| **Fixed in** | `recon-main-decomposition`, Aug 2026 ([#16](https://github.com/gattia/nsm/issues/16)) |

### What was wrong

The call site and the readers disagreed on the parameter's name, and the readers accept
`**kwargs`, which absorbs any misspelled keyword silently. The hardcoded deprecation
list is the only thing those kwargs are checked against.

### How to tell whether one of your runs is affected

Only multi-object `get_rand_pts=True` calls qualify, and no shipped config makes one.
Every such pre-fix run drew 200,000 points per surface regardless of its
`n_pts_random`; the runs are wrong only relative to what was asked, not to each other.

### Reproducing old behaviour

Pass `n_pts_random=200000` explicitly.

*Pinned by:* `test_reconstruct_mesh_options.TestNPtsRandomReachesTheReaders` (both
branches), and the end-to-end `TestSingleObjectSampledBranch`.

## 10. A decoder with no mean surface returned fake success instead of raising

| | |
|---|---|
| **Affects** | Every `reconstruct_mesh` call with `register_similarity=True` or `scale_jointly=True` against a decoder whose zero-latent SDF had no zero level set — the state of every model before it learns a sign change, so chiefly `get_mean_errors` validation early in training, and any new architecture being wired up. Aug 2023 (`5188417`) → Aug 2026 |
| **Unaffected** | Calls without registration/joint scaling (no mean mesh is built, so the state is never tested); any fit against a decoder with a learned surface |
| **Severity** | Silent — **the result looked successful**: `mesh` of Nones and `nan` metrics, but `latent` a correctly-shaped tensor of *zeros* (the untouched `mean_latent`, never fitted) and every other requested key (`center`, `scale`, `icp_transform`, timing, `orig_mesh`) dropped. `get_mean_errors` fed those zero latents to its predictive validation, so `val_prediction_*` r² was computed against fabrications; the downstream consumer read `result["center"]` unconditionally and died with `KeyError` |
| **Fixed in** | `recon-main-decomposition`, Aug 2026 ([#29](https://github.com/gattia/nsm/issues/29)) |

### What was wrong

An early return replaced the whole result with `{mesh, chamfer_*/assd_*, latent}`,
ignoring what the caller asked for — and the "latent" it returned was the zero vector it
was supposed to fit. `reconstruct_mesh` now raises `NoZeroLevelSetError` (named, with
the two causes in the message: model not trained far enough, or
`n_pts_per_axis_mean_mesh` too coarse). `get_mean_errors` catches it and scores the
subject NaN — per-surface metrics and `val_prediction_*` alike — so a training run still
survives its own early validation epochs.

### How to tell whether one of your runs is affected

A stored "fit" whose latent is exactly all-zero with `nan` reconstruction metrics was
never fitted. A `val_prediction_*` series that starts as a plausible number while the
same epoch's chamfer/ASSD are `nan` was regressed against zero vectors; post-fix those
epochs report `nan`.

### Reproducing old behaviour

No compatibility switch. The old result carried no information a caller could use —
catch `NoZeroLevelSetError` instead.

*Pinned by:* `test_reconstruction_regression.TestDecoderWithNoZeroLevelSet` (the raise)
and `test_reconstruct_mesh_options.TestGetMeanErrorsSurvivesADegenerateModel` (the NaN
seam).

## 11. `resume_epoch=1` trained a fresh model while claiming to resume

| | |
|---|---|
| **Affects** | Any `train_deep_sdf` run launched with `resume_epoch: 1`. The resume guard read `> 1` while the epoch loop starts at `resume_epoch + 1`, so the run loaded **nothing** — fresh model, fresh latents, fresh optimizer — and trained epochs 2..`n_epochs`: one epoch short, from random init, with the learning-rate schedules evaluated one epoch ahead of how many steps had actually run. Dec 2024 (`87c5e88`, the trainer overhaul that introduced `resume_epoch` — born with the `> 1` guard) → Aug 2026 |
| **Unaffected** | `resume_epoch: 0` (a fresh run, as asked) and `resume_epoch >= 2` (loads correctly); both shipped ShapeMedKnee configs, which do not set the key |
| **Severity** | Silent — the run printed nothing about resuming, trained, checkpointed and exited 0 |
| **Fixed in** | `trainer-decomposition`, Aug 2026 ([#49](https://github.com/gattia/nsm/issues/49)) |

### What was wrong

Two boundaries disagreed about what `resume_epoch` means. The epoch loop treats it as
the last *completed* epoch (it continues at `resume_epoch + 1`); the load guard treated
`1` as "not really resuming". Both now share the loop's convention: `resume_epoch >= 1`
loads that epoch's checkpoint and continues at the next.

### How to tell whether one of your runs is affected

Only runs launched with `resume_epoch: 1` qualify (`model_params_config.json` records
the config). Such a run's earliest saved checkpoint descends from a fresh
initialization, not from the epoch-1 checkpoint it named.

### Reproducing old behaviour

No switch. The old behaviour was a fresh run whose epoch numbering started at 2 —
`resume_epoch: 0` is the supported way to train from scratch, with epoch 1 included.

*Pinned by:* `test_training_regression.TestResumeContract` (all three boundaries:
0 runs every epoch, 1 and 2 load their checkpoints).

## 12. The logged latent-norm stats were the last batch's, scaled down by the batch count

| | |
|---|---|
| **Affects** | The `mean_vec_length` and `std_vec_length` metrics in every wandb run since the metric existed. `train_epoch` assigned (`=`) where every surrounding accumulator adds (`+=`), then divided by `len(data_loader)` — so the logged value was the last batch's stat over the batch count, wrong by roughly ×n_batches (issue verification on a real 2-batch run: true mean 0.0107, logged 0.0053). Nov 2024 (`0638e31`, the commit that added the metric — born with the `=`) → Aug 2026 |
| **Unaffected** | Weights, gradients, checkpoints, every other logged metric — the two stats were computed and discarded outside the loss path. Single-batch epochs (`objects_per_batch >= n_subjects`), where `=` and `+=` agree |
| **Severity** | Silent, wandb-only. Anyone who read the metric to judge latent-code scale saw a value shrunk by their batch count |
| **Fixed in** | `trainer-decomposition`, Aug 2026 ([#59](https://github.com/gattia/nsm/issues/59)) |

### How to tell whether one of your runs is affected

Every pre-fix run with more than one batch per epoch. The stored series is a scaled
proxy, not garbage: within a run with a fixed batch count, multiplying by
`ceil(n_subjects / objects_per_batch)` recovers the last-batch stat (still one batch's
value, not the epoch mean). Do not compare the metric across runs with different batch
counts.

*Pinned by:* `test_training_regression.TestLatentNormLogging` (latent LR 0 makes the
true epoch mean exact).

---

## 13. The dataset cache key omitted parameters and mesh identity that change cached data

One bundle, deliberately — every part invalidates the same cached `.npz` files, so the
fix costs one regeneration, not several.

| | |
|---|---|
| **Affected** | Any run with `load_cache=True` (the production setting) that reused a cache directory across differing configurations, or edited a mesh in place; any run whose `reference_mesh` was a loaded `Mesh` (cache never hit) |
| **Severity** | Silent — the second run trains on the first run's data, exit 0 |
| **Fixed in** | `cache-checkpoint-migration`, Aug 2026 ([#19](https://github.com/gattia/nsm/issues/19)) |

### What was wrong

The cache key was `md5` over a positional, stringified list of *some* of the parameters
that decide what `get_sample_data_dict` writes:

- `mesh_to_scale` (multi-surface) and `uniform_pts_buffer` (both classes) were absent.
  Two runs differing only in one of them shared a key, and the second silently trained
  on the first's data. `mesh_to_scale` is the worst: it decides which surface drives
  centering and normalization, so the two runs' cached points and SDFs are in different
  coordinate frames entirely.
- Mesh *content* was absent — the key held path strings alone. Editing or re-exporting
  a mesh in place left the key standing, so every later run reused the pre-edit samples.
- A `reference_mesh` passed as a loaded `Mesh` was stringified, and `Mesh.__str__`
  embeds the memory address: the key was per-object, so such a dataset could never hit
  its own cache and paid full regeneration every run.
- The multi-surface list also carried an unexplained `False` literal, inserted the
  subject's mesh paths in reverse order, and hashed an integer `reference_mesh` as the
  raw index — reordering `list_mesh_paths` re-aimed the reference while the key stood
  still. Position carried meaning: the same defect class as the LR-schedule bug (§1).
- `subsample` was also absent — but its only cached-content effect was that
  `sdf_pos_neg_idx` padded (repeated) the pos/neg index arrays for the build-time
  `subsample` and cached the result. Reloading with a larger one found too few
  entries: `__getitem__` took what there was and topped the batch up with uniform
  random points, so `equal_pos_neg=True` quietly stopped holding. Measured: 1.6×
  interior under-representation on the small surface (interior fraction 0.20 against
  a fresh 0.32) once the reloaded `subsample` exceeded the cached point count — and
  in a real dataset the small surface is the cartilage.

### What changed

The key is a named canonical mapping (`json.dumps(sort_keys=True)`, then `md5`):
`mesh_to_scale` and `uniform_pts_buffer` are in it; every mesh path contributes a
content-stable `(path, size, mtime)` identity, so an in-place edit moves the key
without any file being read; a `Mesh`-valued reference contributes a digest of its
geometry; an int or list reference resolves to the underlying path(s) first. A
`cache_format` entry versions the key, so the next content-affecting change is one
integer bump instead of a new hashing scheme.

`subsample` was decoupled instead of keyed — batch size is a serving parameter, and
forcing a full resample when it changes would be wrong in the other direction. The
cache stores the raw per-sign index sets; the padding happens at draw time
(`_draw_sign_share`), sized by the subsample in force, so cached bytes no longer
depend on it and a reused cache draws identically-balanced batches. For an unchanged
subsample the padded array the draw permutes is byte-identical to what the cache used
to store, so batches are bit-identical across the change.

### How to tell whether one of your runs is affected

You reused one cache directory across runs that differed in `mesh_to_scale` or
`uniform_pts_buffer` — the later run trained on the earlier one's data; you reloaded a
cache with a larger `subsample` than it was built at — that run's batches were
unbalanced; or you edited a mesh without renaming it — every later run reused the
pre-edit samples. One cache directory per configuration and unedited meshes, and you
are fine.

### Migration

None possible, and none needed: keys are opaque, so a legacy file is indistinguishable
from another configuration's, and serving wrong data is exactly what stops happening.
No pre-fix key can ever hit again — the first run per configuration regenerates its
cache (identical data when `random_seed` is set, §3), and old cache directories are
reclaimable disk.

*Pinned by:* `test_dataset_cache.TestFormerlyCollidingParameters`,
`test_dataset_cache.TestMeshContentInTheKey`, `test_dataset_cache.TestReferenceMeshHashing`,
`test_dataset_cache.TestHashedParametersChangeTheKey`.

## 14. `layer_split: false` meant "split at layer 0", and progressive depth's first phase-in epoch ran at full weight

Two `deep_sdf.Decoder` option defects, bundled because they are the same shape — a start
condition written more than once and inconsistently — and because no shipped model is
affected by either.

| | |
|---|---|
| **Affected** | `deepsdf` models built from a config carrying `"layer_split": false` (what `default_config.json` ships); training runs with `progressive_add_depth: true` **and** `layer_split` set, at exactly `epoch == start_epoch` of each phased-in block |
| **Unaffected** | Every triplanar model, including both shipped ShapeMedKnee models — `TriplanarDecoder` builds its inner `Decoder` with `layer_split=None` and never passes `progressive_add_depth`; every `progressive_add_depth: true` run without `layer_split`, which raised `TypeError` on the first forward below the last `start_epoch` and so produced no results |
| **Severity** | `layer_split`: silent, and it changes the architecture. Progressive depth: one epoch per block |
| **Fixed in** | `models-package-sweep`, Aug 2026 ([#46](https://github.com/gattia/nsm/issues/46)) |

### What was wrong

**`layer_split`.** `Decoder` decides whether to split with `self.layer_split is not None`,
and `False is not None`. So `"layer_split": false` — the value `default_config.json` ships
and the value every reader takes to mean *off* — selected a split at layer 0: every layer
became an `nn.ModuleList` of `n_objects` branches, moving every state-dict key from
`layers.N.weight` to `layers.N.0.weight`. With `objects_per_decoder > 1` it also changed
the output head, from one stack emitting `n_objects` channels to `n_objects` stacks each
emitting one. `False == 0` in Python, so no value comparison can separate the shipped
"off" from a deliberate split at layer 0 — only an identity check can.

**Progressive depth.** `forward_branch_` phases a block in at `epoch >= start_epoch`, while
`progressive_layer` blended only for `start < epoch < end`. `epoch == start` therefore fell
through to the `else` and applied the block at **full weight** — before the warmup, whose
own first weight is `(1 / warmup) ** 2`, near zero. So the block's contribution went
0 → 1 → ~0 → ramp. (`progressive_layer` also carried an `epoch < start` `RuntimeError` that
its one caller could not reach.)

### What changed

`layer_split=False` normalizes to `None` at construction; `0` still means split at layer 0.
`progressive_layer` blends for `epoch < end`, one condition rather than three, so
`epoch == start` weights the block at zero — an identity, which is exactly what skipping it
one epoch earlier does. A not-yet-started block now returns its input rather than `None`.

### How to tell whether one of your runs is affected

Check `model_params_config.json`. `"model_type": "triplanar"` — not affected, by either.
`"model_type": "deepsdf"` with `"layer_split": false` — that checkpoint was built with
every layer split, so it no longer loads into a model built from the same config: it fails
loudly with `Missing key(s)`/`Unexpected key(s)`, and **passing `layer_split=0` explicitly
reproduces the original architecture exactly**. `"progressive_add_depth": true` with
`layer_split` set — one epoch per phased-in block trained with that block at full weight;
everything else in the run is unchanged.

*Pinned by:* `test_model_options.test_layer_split_false_is_the_same_model_as_no_layer_split`,
`test_model_options.test_layer_split_zero_still_splits_at_layer_zero`,
`test_model_options.test_a_block_phases_in_continuously_across_its_start_epoch`.

## 15. `sum_conv_output_features: false` trained on one plane of three

| | |
|---|---|
| **Affected** | Any training or reconstruction run with `sum_conv_output_features: false` (the `TriplanarDecoder` argument `sum_sdf_features=False`) |
| **Unaffected** | Everything else, including **both shipped ShapeMedKnee models** — 647 and 551 set it `true`, and it defaults to `true` in the constructor, `loader` and `get_model_config_template` |
| **Severity** | Silent. The model builds, trains, converges and reconstructs; it is simply a third of the architecture it was asked for |
| **Fixed in** | `models-package-sweep`, Aug 2026 ([#45](https://github.com/gattia/nsm/issues/45)) |

### What was wrong

`TriplanarDecoder.__init__` sized the VAE output by `sdf_latent_size` when not summing —
correct, since the three planes are concatenated into the decoder's input width — while
`forward_with_plane_features` sliced `sdf_latent_size` **per plane**. For
`sdf_latent_size=12`, the xz plane received all 12 channels and yz and xy received
zero-channel slices. `grid_sample` on a zero-channel plane returns an `(N, 0)` tensor and
does not complain, so the concatenation produced a result `torch.equal` to sampling the xz
plane alone. Every VAE parameter still received gradient — through the xz geometry — so
training converged, to a model using one plane of three.

The `assert` guarding the branch said "if sum_sdf_features is True" while guarding the
`False` branch, which is one reason nobody read it as suspicious.

`conv_pred_sdf: true` combined with concatenation was broken past that: three
low-frequency SDF channels, one per plane, with no defined rule for combining them, and a
feature vector two channels wider than the SDF decoder was sized for. It always raised a
shape error on the first forward.

### What changed

Each plane's slice is `sdf_latent_size // 3` (plus one channel when `conv_pred_sdf`), so
the three concatenate to exactly the decoder's input width. The divisibility guard is a
`ValueError` rather than an `assert`, since `python -O` strips asserts and this one guards
a shape. Concatenation with `conv_pred_sdf` refuses at construction.

**The VAE's output width is unchanged** — it was `sdf_latent_size` before and is
`sdf_latent_size` after — so every parameter shape is the same and a pre-fix checkpoint
still loads under strict `load_state_dict`. It then computes something different, which is
the reason this entry exists.

### How to tell whether one of your runs is affected

`model_params_config.json`: `"sum_conv_output_features": false`. If the key is absent or
`true`, the run is unaffected. If it is `false`, that model used only its xz plane, and no
comparison drawn against a summed model is meaningful — retrain rather than re-evaluate,
since the checkpoint loads either way.

*Pinned by:* `test_model_options.test_concatenation_uses_all_three_planes`,
`test_model_options.test_the_concatenating_vae_keeps_the_width_it_always_had`,
`test_model_options.test_triplanar_feature_combination_works_or_refuses`.

## 16. A `padding` a config did not state was silently defaulted, at any trained value

| | |
|---|---|
| **Affected** | Any model trained at a `padding` other than 0.1 and loaded through `load_model` from a config that omits the key — every SDF it computed was wrong |
| **Unaffected** | Models trained at `padding=0.1`, which is the constructor default and what **both shipped ShapeMedKnee models** ran at; any caller that states `padding` in the config |
| **Severity** | Silent, and it scales the whole query domain |
| **Fixed in** | `models-package-sweep`, Aug 2026 ([#26](https://github.com/gattia/nsm/issues/26)) |

### What was wrong

`TriplanarDecoder.padding` scales query coordinates before they index the feature planes
(`normalize_coordinates`). It is **not a learned parameter**, so nothing in a checkpoint
constrains it: strict `load_state_dict` succeeds at any value, and the model then samples
the feature planes at the wrong scale. `loader._get_triplanar_params` defaulted it to 0.1
with a `config.get`, so a config that never mentioned `padding` produced a working,
plausible, wrong model.

**Measured.** A model built at `padding=0.35`, saved, and loaded through `load_model` with
a config omitting the key computes a maximum absolute SDF difference of **0.063**. The
output is `tanh`-bounded to (−1, 1), so that is ~3% of the full range.

### What changed

`padding` is a required key for the triplanar branch. A config without it raises `KeyError`
with the value to write, including the note that a config predating the key belongs to a
model trained at the constructor default — so `"padding": 0.1` reproduces such a model
exactly, and stating it restores bitwise-identical output.

This is option 1 of the three the issue lists. Options 2 (write it into the checkpoint) and
3 (a public "build the model this config describes" call) are not done: option 2 would put
a key in the state dict that no shipped checkpoint has, and option 3 is the model-registry
work in `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md` §8.1.

**`kneepipeline` is not covered by this fix and does not need to be.** It hand-rolls the
config→constructor mapping (`steps/run_nsm.py:94-112`, 15 of 16 meaningful arguments) and
never calls `load_model`, so the refusal does not reach it — and both models it loads were
trained at the default. Closing that gap is what option 3 is for; `SCOPE.md` §3.1 tracks it.

### How to tell whether one of your runs is affected

Was the model trained at a `padding` other than 0.1, and did the config or caller you
loaded it with state that value? If not, every SDF that model computed is wrong by up to
~3% of the output range. Re-run with `padding` stated; nothing about the checkpoint needs
to change.

*Pinned by:* `test_model_roundtrip.TestPaddingIsNotInTheCheckpoint`.

---

## 17. Face arrays were reshaped without validation, so a non-triangle mesh built garbage

| | |
|---|---|
| **Affected** | Five functions in `mesh/`, on any input that was not an (M, 3) triangle array: `correspondence_metrics.self_intersection_count` / `foldover_count` given a non-triangular mesh, and `interpolate.build_mesh_laplacian` / `compute_feature_mask` — and through them `interpolate_points(tangent_laplacian=True)` — given a VTK-style flat `faces` array |
| **Unaffected** | Every call that passed an all-triangle mesh, or an already-(M, 3) array: the reshape and the replacement return the identical array, asserted |
| **Severity** | Silent on some inputs, a bare `ValueError` on others, and which one you got depended on the cell count |
| **Fixed in** | `mesh-package-sweep`, Aug 2026 ([#57](https://github.com/gattia/nsm/issues/57)) |

### What was wrong

Each site took a face array and called `reshape(-1, 4)[:, 1:]` or `reshape(-1, 3)`. A
VTK-style array is `[n, i0, …, in, n, …]`, so the reshape succeeds exactly when its flat
length happens to divide — a fact about the cell count mod 3 or mod 4, not about the mesh
being triangular. Measured:

| input | flat length | `reshape(-1, 4)` | `reshape(-1, 3)` |
|---|---|---|---|
| 3 quads | 15 | `ValueError` | **5 fabricated rows** |
| 4 quads | 20 | **5 fabricated rows** for 4 cells | `ValueError` |
| 96 triangles, VTK-style `.faces` | 384 | correct (96) | **128 fabricated rows** |
| 4 triangles + 4 quads | 36 | **9 fabricated rows** for 8 cells | `ValueError` |

The rows are interleaved cell-size markers and vertex indices read as triangles, so they
index real vertices and every downstream computation succeeds. On a 4-quad strip
`self_intersection_count` returned `0` and `foldover_count` returned
`near_degenerate: 2`. On `pv.Sphere(8, 8).faces` passed as the `faces=` argument,
`build_mesh_laplacian` built a 373-non-zero smoothing operator where the correct one has
288, and `compute_feature_mask` pinned 50 vertices where the correct answer is 8 — so the
interpolated correspondence was wrong, not absent.

### How to tell whether one of your runs is affected

Two questions, and both have to be "no":

1. Did you pass `interpolate_points(..., faces=)` anything other than an (M, 3) array —
   `mesh.faces` rather than `mesh.regular_faces`, most likely? If so the tangent-Laplacian
   smoothing used the wrong neighbourhood graph and pinned the wrong vertices; re-run with
   `regular_faces`. Without `tangent_laplacian=True` neither function is reached.
2. Did you score a mesh that was not all-triangles? `mesh.is_all_triangles` answers it. If
   not, `self_intersection_count` and `foldover_count` are meaningless for it — but only
   for cell counts that happened to divide by 4; other counts raised. Triangulate and
   re-score. `triangle_health` in the same result is unaffected either way: it goes
   through `TriangleProperties`, which has always refused a non-triangle cell.

Under the fix every one of those calls raises a `ValueError` naming what to pass instead,
so a re-run cannot silently repeat the mistake.

*Pinned by:* `testing/NSM/mesh/test_mesh_contracts.py`, §1.

---

## 18. `score_correspondence` measured the round trip against the wrong mesh

| | |
|---|---|
| **Affected** | `score_correspondence(roundtrip_points=…)` called **without** `source_mesh`. Both `roundtrip_distance` and `forward_backward_disagreement` |
| **Unaffected** | Every call that passed `source_mesh`, and every call that did not pass `roundtrip_points` (those two keys already said `{"skipped": True}`) |
| **Severity** | Silent — a plausible number where the neighbouring key in the same dict correctly reported a skip |
| **Fixed in** | `mesh-package-sweep`, Aug 2026 ([#54](https://github.com/gattia/nsm/issues/54)) |

### What was wrong

Both metrics measure how far a forward-then-backward warp lands from where it started, so
the reference positions are the **source** mesh's. With `source_mesh=None` the code
substituted `warped_mesh` — which measures the warp itself and reports it as a round-trip
error. Measured on a 1.5× scaling with a 0.001 round-trip displacement: mean
`roundtrip_distance` **0.2500** against a true **0.0017**, a factor of 144. In the same
returned dict, `foldover_count` correctly reported
`{"skipped": True, "reason": "source_mesh not provided"}`.

### How to tell whether one of your runs is affected

Did the call that produced the numbers pass `source_mesh`? If not, discard
`roundtrip_distance` and `forward_backward_disagreement` from that result — every other
key in it is unaffected. Under the fix those two keys skip with a reason instead.

*Pinned by:* `test_mesh_contracts.test_roundtrip_metrics_skip_without_a_source_mesh`.

---

## 19. The adaptive-meshing fallback built its grid where `search_bounds` was not

| | |
|---|---|
| **Affected** | `create_mesh_adaptive` calls that set `search_bounds` away from the default `(-1.0, 1.0)`, left `voxel_origin` unset, and hit the fallback — which fires when the coarse pass finds no zero crossing |
| **Unaffected** | Every run at the defaults, and that is every NSM-owned one: `reconstruct_mesh` builds `search_bounds` from `recon_grid_origin`, which defaults to `1.0` and which no NSM config overrides. The two-pass path never reads `voxel_origin` at all |
| **Severity** | Silent — a mesh from a grid that did not cover the requested region |
| **Fixed in** | `mesh-package-sweep`, Aug 2026 ([#60](https://github.com/gattia/nsm/issues/60)) |

### What was wrong

The fallback forwarded `voxel_origin` — its own `(-1, -1, -1)` default — alongside a
`voxel_size` derived from `search_bounds` a few lines above. The two disagreed by
construction whenever `search_bounds` was not centred on the origin at unit half-width.
Measured with `search_bounds=(0.0, 4.0)` and `n_pts_per_axis=17`: the fallback grid spanned
`[-1, 3]` on every axis, so it searched a region the caller had not asked about and missed
most of the one they had. The value was one of **17 positional arguments** in that call.

`voxel_origin` now defaults to `None`, meaning "take it from `search_bounds`"; at the
default `search_bounds` that reproduces `(-1, -1, -1)` exactly, which is why no run at the
defaults moves. An explicitly passed origin still wins.

### How to tell whether one of your runs is affected

Did you pass a non-default `search_bounds` (or a `recon_grid_origin` other than `1.0`)
*and* leave `voxel_origin` unset? If so, only reconstructions that logged
`"Coarse pass found no surface. Falling back."` are affected — the two-pass path never
used the parameter. Those reconstructions should be re-run.

*Pinned by:* `test_mesh_contracts.test_fallback_grid_covers_search_bounds` and
`test_default_search_bounds_keep_the_historical_fallback_origin`.

## 20. `reconstruct_mesh` accepted a misspelled parameter and used the default instead

| | |
|---|---|
| **Affected** | Any `reconstruct_mesh`, `get_mean_errors` or `reconstruct_latent` call that passed a keyword the signature does not name — a misspelling, a renamed parameter, or one copied from a sibling function |
| **Unaffected** | Calls whose keywords all spell a real parameter, which is every NSM-internal one and kneepipeline's. `batch_size_latent_recon` (`reconstruct_mesh`) and `max_batch_size` (`reconstruct_latent`) were and remain deliberately accepted |
| **Severity** | Silent — the run completed, reported nothing, and used the default for the parameter the caller believed they had set |
| **Fixed in** | `reconstruct-mesh-internals`, Aug 2026 (plan §8.0.J); the `reconstruct_latent` site in `latent-fit-internals`, Aug 2026 (plan §8.0.K) |

### What was wrong

`reconstruct_mesh` takes 58 named parameters and a `**kwargs` that was inspected for
exactly one key. Every other key reached the end of the function unread. Measured across
five misspellings of real parameters — `n_pts_per_axes`, `num_iteration`, `calc_assd_`,
`latent_reg_wieght`, `clamp_distance` — all five completed a reconstruction with no
exception, no warning and no log record. With 58 near-synonymous names, this is the
likeliest way to call the function wrongly and it was the only way that produced no signal
at all.

It now raises `TypeError`, naming the unknown key.

**The same hole was one call level down and was fixed a slice later.**
`reconstruct_latent` takes 38 named parameters and a `**kwargs` read for exactly one key,
`max_batch_size`. Measured across seven misspellings — `num_iteration`,
`latent_reg_wieght`, `clamp_distance`, `lattent_size`, `optimiser_name`, `n_iterations`,
`lr_` — all seven completed a fit with no exception, no warning and no log record.
`reconstruct_mesh` splats a dict of 36 keys into it, all of them real parameters, so a
caller who reached this second site had called `reconstruct_latent` directly.

### How to tell whether one of your runs is affected

Re-run the same call under fixed code. If it raises `TypeError`, that keyword was being
ignored, and the run used the default shown in the signature for whatever you meant to
set. The two that change a result rather than a diagnostic are the grid
(`n_pts_per_axis`, default 256) and the fit (`num_iterations` 1000, `lr` 5e-4,
`latent_reg_weight` 1e-4, `clamp_dist` None); a misspelling among those means the run was
not configured as recorded and should be re-run. For `reconstruct_latent` the fit
parameters are the whole list — there is no grid.

*Pinned by:* `test_reconstruct_mesh_contracts.TestUnknownKeywordsAreRefused` and
`test_reconstruct_latent_internals.TestUnknownKeywordsAreRefused`.


## 21. `reconstruct_latent` returned the number 100 instead of a loss

| | |
|---|---|
| **Affected** | Any `reconstruct_latent` call with `convergence="recon_loss"` — the mode `NSM/configs/default_config.json` ships as `convergence_type_recon` — that read the first element of the returned `(loss, latent)` |
| **Unaffected** | `convergence="overall_loss"` and `convergence="num_iterations"`, and every `reconstruct_mesh` / `get_mean_errors` caller: `reconstruct_mesh` binds the returned loss and never reads it, so no reconstruction result, metric or mesh is touched. The fitted latent was always correct |
| **Severity** | Silent — a plausible-looking constant where a loss was expected |
| **Fixed in** | `latent-fit-internals`, Aug 2026 (plan §8.0.K) |

### What was wrong

`loss` and `recon_loss` were both initialised to the literal `100` and each did two jobs:
the sentinel the next step is compared against, and the value returned. Under
`convergence="recon_loss"` only `recon_loss` was ever updated, so `loss` was still `100` at
the `return`. Measured exactly: the returned loss is the int `100`, not a tensor.

The same sentinel had a second failure. It was not worse than every loss, so a fit whose
losses never dropped below 100 recorded no step at all and raised
`UnboundLocalError: local variable 'latent_' referenced before assignment` — after running
every iteration it was asked for. That half always crashed, so it costs nobody a result;
it is recorded here because it is the same line.

`loss` is now recorded with the latent it belongs to, and the sentinel is `float("inf")`.

### How to tell whether one of your runs is affected

Only a number you logged is affected, never a latent or a mesh. If you have a
`reconstruct_latent` loss recorded as exactly `100` — or `100` for every subject in a
cohort — that is this, and the fit itself was fine. Re-running is not necessary; the
recorded loss is simply not a loss.

*Pinned by:* `test_reconstruct_latent_internals.TestTheReturnedLossIsALoss`.


## 22. `hybrid_optimizer` decayed its learning rate to zero, and ignored `optimizer_name`

| | |
|---|---|
| **Affected** | Any `reconstruct_latent` or `reconstruct_mesh` call with `hybrid_optimizer=True`, `n_lr_updates` set, and `adam_iterations` larger than `num_iterations` |
| **Unaffected** | Every run at the default `hybrid_optimizer=False`, which is both shipped configs and kneepipeline. Hybrid mode has never been on a production path |
| **Severity** | Silent — the Adam phase stopped moving the latent partway through and reported nothing |
| **Fixed in** | `latent-fit-internals`, Aug 2026 (plan §8.0.K) |

### What was wrong

`adjust_lr_every` was derived from `num_iterations`, but with `hybrid_optimizer=True` the
loop runs `adam_iterations + lbfgs_iterations` and `num_iterations` is read for nothing
else. So `n_lr_updates` meant a different thing in each mode. Measured at
`num_iterations=10, adam_iterations=100, n_lr_updates=2, lr_update_factor=10`: **11 decays
ending at exactly 0.0**, where the same 100 Adam steps scheduled over their own horizon
take one. An Adam phase at learning rate 0.0 leaves the latent where the previous step put
it, for the rest of the phase.

Separately, `optimizer_name` was not consulted at all in hybrid mode — the loop derives its
optimizer from the step number. `hybrid_optimizer=True` with a non-default
`optimizer_name` now raises rather than discarding one half of the pair.

### How to tell whether one of your runs is affected

If `adam_iterations <= num_iterations` the schedule was already right. Otherwise the
learning rate reached `lr * lr_update_factor ** -(adam_iterations // (num_iterations //
n_lr_updates))`; if that is at or below float underflow, the tail of the Adam phase did
nothing and the latent is whatever the last non-zero-LR step produced. Re-run affected
fits — the fitted latent is not the one the configuration describes.

*Pinned by:*
`test_reconstruct_latent_internals.TestTheLearningRateScheduleSpansThePhaseItSteps`.


## 23. An unrecognised `convergence` silently meant "num_iterations"

| | |
|---|---|
| **Affected** | Any `reconstruct_latent` or `reconstruct_mesh` call whose `convergence` was not exactly `"num_iterations"`, `"overall_loss"` or `"recon_loss"` — a typo, a capitalisation such as `"Recon_Loss"`, `None`, or `""` |
| **Unaffected** | The three exact spellings, which is every NSM-internal call, both shipped configs and kneepipeline (`convergence_type_recon` is `"recon_loss"`) |
| **Severity** | Silent — the fit ran to completion with early stopping disabled, and reported nothing |
| **Fixed in** | `latent-fit-internals`, Aug 2026 (plan §8.0.K) |

### What was wrong

The convergence block is `if convergence == "overall_loss": ... elif convergence ==
"recon_loss": ... else: <treat as num_iterations>`. The `else` is a real branch, not a
missing one, so an unrecognised value selected it. Measured across `"Recon_Loss"`,
`"recon_los"`, `""`, `None` and `"banana"`: all five completed and returned a latent
bit-identical to `convergence="num_iterations"`.

The consequence is not a wrong number but a missing behaviour: `convergence_patience`
never applies, the fit runs every one of `num_iterations` steps, and the returned latent is
the last one rather than the best one. For a fit that would have converged and stopped
early, the difference is however far the latent drifted afterwards.

`convergence` is now case-folded and then refused, alongside `optimizer_name` and
`loss_type`.

### How to tell whether one of your runs is affected

Check the spelling in the config or call that produced the run, against the three values
above; case matters only in the sense that it used to. If it does not match one of them
exactly, the run had no early stopping. Re-running matters if `convergence_patience` was
meant to be doing something — compare the recorded step count against `num_iterations`: an
affected run used all of them.

*Pinned by:*
`test_reconstruct_latent_internals.TestUnknownValuesAreRefusedWhereTheyAreNamed`.


## 24. A multi-surface draw was weighted towards whichever surface had the most vertices

| | |
|---|---|
| **Affected** | `reconstruct_latent` / `reconstruct_mesh` fits with more than one surface where **some surface had fewer points than `n_samples // n_surfaces`** — i.e. the budget was large enough to exhaust the smallest surface |
| **Unaffected** | Single-surface fits at any setting, and any multi-surface fit whose every surface has at least its share — which is the shipped configuration (`n_samples_latent_recon: 20000` over surfaces of tens of thousands of vertices) and kneepipeline. Both draw identically before and after |
| **Severity** | Silent — the fit ran, and sampled space unevenly around the surfaces |
| **Fixed in** | `latent-fit-internals`, Aug 2026 (plan §8.0.K) |

### What was wrong

Each surface was given `n_samples // n_surfaces` points, capped at what it had:
`min(share, count)`. Nothing redistributed what a small surface could not use, so the draw
was **neither of the two things it could sensibly be** — not balanced, because a surface
below its share contributed less than the others while they kept theirs; and not the
requested size, because the shortfall was dropped without a word. On a 300/90 cloud,
`n_samples=390` drew `[195, 90]`.

That matters because `pts_surface` does not route points to losses — every surface's SDF is
evaluated at every drawn point — so what it controls is where in space the samples come
from. An uneven draw weights the fit towards whichever surface has the most vertices.

A top-up from the whole cloud was written into the draw loop and was unreachable
(`n_samples_` was rebound above it, so `current_filled` always equalled it; measured at 0
executions across 49 surface-count and budget combinations). That was the other design —
draw the requested count and accept the imbalance — and it has been deleted rather than
repaired.

Every surface now contributes the same count, held to what the smallest contributing
surface can supply. The cost is stated rather than hidden: with unequal surfaces the whole
cloud is unreachable, and raising `n_samples` past `n_surfaces × smallest_surface` does
nothing.

### How to tell whether one of your runs is affected

Compare `n_samples // n_surfaces` against your smallest surface's point count. If every
surface has at least that many, the draw is unchanged and so is the result. If any surface
is smaller, that run sampled unevenly: the deficit surface contributed all of itself while
the others contributed the full share, so the fit was weighted towards the larger surfaces
by the ratio between them. Re-run if the imbalance was large — with four surfaces of
62,530 / 48,407 / 82,213 / 35,808 vertices and `n_samples` above ~143,000, the old draw took
every vertex of each (228,958 points, ratio 2.3:1 largest to smallest) where the new one
takes 35,808 from each.

*Pinned by:* `test_reconstruct_latent_internals.TestTheMultiSurfaceDrawIsBalanced`.

---

## 25. The logged latent-norm stats were the last *split*'s whenever `batch_split` > 1

| | |
|---|---|
| **Affected** | The `mean_vec_length` and `std_vec_length` metrics of any run with `batch_split` above 1. `train_epoch` computed both inside the split loop and accumulated them outside it, so whichever split ran last was the one counted. Nov 2024 (`0638e31`, the commit that added the metric) → Aug 2026 |
| **Unaffected** | `batch_split: 1`, which is the shipped `default_config.json` and what the training regression baselines run — the split loop runs once and the two forms agree exactly, bit for bit. Weights, gradients and checkpoints at any `batch_split`: the stats sit outside the loss path |
| **Severity** | Silent, wandb-only, and sometimes `NaN` — see below |
| **Fixed in** | `train-epoch-internals`, Aug 2026 (plan §8.0.L) |

### What was wrong

This is [§12](#12-the-logged-latent-norm-stats-were-the-last-batchs-scaled-down-by-the-batch-count)
at its second site. That fix changed `=` to `+=` on the **batch** loop; underneath it is a
**split** loop, and the two statistics were computed inside that one:

```python
for split_idx in range(...):
    ...
    mean_vec_length = torch.mean(torch.norm(batch_vecs, dim=1))   # rebound every split
step_mean_vec_length += mean_vec_length.item()                    # only the last survives
```

`batch_split` exists to bound memory — `torch.chunk` partitions the batch and the
per-surface losses are divided by the whole batch's point count, so the loss is invariant
to it. These two metrics were not. Measured on a 4-subject fixture, same seed and data,
only `batch_split` changing: `mean_vec_length` 0.1445 / 0.2026 / 0.3201 for 1 / 2 / 4 and
`std_vec_length` 0.1031 / 0.1213 / 0.0, against a `loss` invariant across the same three
runs to 1.5e-08.

The `std` is worse than wrong where a split holds a single row: `torch.std` of one value is
undefined, so the epoch's payload carried `NaN`. On the same fixture `batch_split` 6 and 16
both reported `nan`.

Both statistics are now collected across the splits and reduced over the whole batch, which
is what "the epoch mean over batches" has always meant and what removes the `NaN` as well
as the drift.

### How to tell whether one of your runs is affected

Read `batch_split` from the run's config. At 1 the series is unchanged. Above 1 the series
is the last split's statistic per batch, averaged over batches: not recoverable by scaling,
because which subjects landed in the final chunk depends on the shuffle. `NaN` in the
series means some batch's final chunk held one row. Nothing else about the run is affected
— re-training buys only the metric.

*Pinned by:*
`test_train_epoch_internals.TestTheLatentNormStatsAreTheEpochMean` (independence from
`batch_split`, the epoch mean computed from the embedding at latent LR 0, and the `NaN`).

## 26. `model_params_config.json` recorded no subject list, or a previous run's

| | |
|---|---|
| **Affected** | The `list_mesh_paths` entry of `model_params_config.json` for any run whose config carried a `list_mesh_paths` key of its own — which the shipped `NSM/configs/default_config.json` does, as `null`. `save_model_params` placed its argument first and then merged the config over it. → Aug 2026 |
| **Unaffected** | Every other key in the file, and everything else a run produces: weights, latent codes, optimizer state, metrics. A config with no `list_mesh_paths` key recorded the correct list |
| **Severity** | Silent, and it removes provenance rather than corrupting a result |
| **Fixed in** | `utils-model-params`, Aug 2026 (plan §8.0.M) |

### What was wrong

```python
dict_save = {"list_mesh_paths": list_mesh_paths}
dict_save.update(config)          # a config key of the same name wins
```

The argument is the training dataset's mesh-path list, taken from
the dataset's own `list_mesh_paths` attribute at the call site. The config's copy is not: it is either the
shipped default's `null`, or an earlier run's list, round-tripped back in from that run's
saved file. Both used to overwrite the argument.

Two reachable cases, both measured:

- **`default_config.json` carries `"list_mesh_paths": null`.** Every run started from it
  recorded `null` — no subject list at all — in the file `load_model`,
  `examples/load_trained_model.py` and both consumer scripts read.
- **A config round-tripped from a previous run's saved file carries that run's subjects.**
  `NSM/configs/generate_sdf_default_config.py`'s header lists `list_mesh_paths` among the
  machine paths it had to sanitize out of the `647_nsm_femur_v0.0.1` config, so this is the
  shape a real saved config has. Re-training then recorded the previous run's subjects
  against this run's weights.

The argument is applied after the merge now. Where the config's own value disagrees and is
not `None`, the override is logged, because that is the case where the config really did
say something else.

### How to tell whether one of your runs is affected

Open the run's `model_params_config.json` and read `list_mesh_paths`. `null` means the list
was lost. A list means it is either correct or an earlier run's — compare it against the
dataset the run actually trained on; if the file was produced from a config derived from
another run's saved file, assume it is that run's. Nothing else in the file is affected, and
nothing about the weights is: the entry is provenance, not an input to training or to
reconstruction, and no NSM code path reads it back.

*Pinned by:*
`test_utils.TestTheRecordNamesItsSubjects` (the shipped default, the round trip, the
control with no config key, and the fact that no regression test read the value).

## 27. A single-joint cartilage validation function scored the femur's meshes

| | |
|---|---|
| **Affected** | Runs whose `recon_val_func_name` named `compare_cart_thickness_tibia`, `_patella` or `_femur` while the model produced **more than two surfaces** whose first pair was **not that joint's** bone and cartilage. → Aug 2026 |
| **Unaffected** | Two-surface models, which is what both shipped ShapeMedKnee configs are; `compare_cart_thickness_femur` on a femur-**first** multi-surface list, which sliced the correct pair and scored it — right numbers by position rather than by contract, a shape the fix removes anyway (ruled fixed-layout by design, `SCOPE.md` §2.5, CHANGELOG § Unreleased *Breaking*); `compare_cart_thickness_whole_joint`, which took its own slices; every metric other than `cart_thick_*`; and everything about the weights |
| **Severity** | Silent, and it produces NaN rather than a wrong number |
| **Fixed in** | `slice-n-prime-cartilage-func`, Aug 2026 (plan §8.0.N′) |

### What was wrong

All three single-joint functions took `orig_meshes[:2]` and `recon_meshes[:2]` and nothing
checked what they sliced from. Handed a six-mesh whole-joint list — femur, tibia, patella
— `compare_cart_thickness_tibia` therefore scored the **femur's** bone and cartilage, and
looked up region indices 2 and 3 on a femoral label array that has neither.

Measured: eight keys, every value NaN, exit code 0. pymskt's `get_cart_thickness_mean`
warns per region ("No data for region 2 - returning mean as nan") and returns NaN;
`get_mean_errors` averages the column and logs it.

The effect is confined to NaN because the canonical label sets are disjoint: femoral
subregions are 11–15 and the tibial indices are 2 and 3, so the wrong meshes can never
contain the requested region and the function cannot return a plausible-but-wrong number.

### How to tell whether one of your runs is affected

Read `recon_val_func_name` and `objects_per_decoder` together in
`model_params_config.json`. A single-joint function with `objects_per_decoder > 2` is the
affected shape — unless the list's first pair is that joint's own bone and cartilage, the
femur-first case above, whose numbers are correct — and its `cart_thick_*` metrics in
that run are NaN throughout — including
the derived `cart_thick_*_corr` and `cart_thick_*_RMSE`. Nothing else in the run is
touched: this is a validation metric, never a training signal.

The same configuration now raises `ValueError` naming the function, the count it needs and
the layout it assumes.

*Pinned by:*
`test_cartilage_func.TestTheMeshListLength` (the whole-joint list into the tibia wrapper,
the wrong-length pairs, and the four-surface `["bone", "cart", "med_men", "lat_men"]`
layout into the whole-joint function).


## 28. ASSD downcast the caller's meshes to float32 before measuring them

| | |
|---|---|
| **Affected** | `compute_recon_loss(..., calc_assd=True)` — and therefore `get_mean_errors` and `reconstruct_mesh` with `calc_assd`, which is what both shipped configs set — called with meshes whose `point_coords` are **float64**. → Aug 2026 |
| **Unaffected** | Everything NSM produces or reads itself: VTK stores points at single precision, so `create_mesh`'s output and a mesh pymskt reads from disk are already float32 and the cast was a no-op. Bit-identical on both shipped ShapeMedKnee configs and on `kneepipeline`. The chamfer path never cast at all |
| **Severity** | Silent, small, and in the direction of *less* precision |
| **Fixed in** | `slice-n-phase2-close`, Aug 2026 (plan §8.0.N, [#55](https://github.com/gattia/nsm/issues/55)) |

### What was wrong

Before measuring ASSD, `recon_evaluation.compute_recon_loss` did this:

```python
# make sure the points for the meshes are the same types
mesh.point_coords = mesh.point_coords.astype(np.float32)
orig_meshes[mesh_idx].point_coords = orig_meshes[mesh_idx].point_coords.astype(np.float32)
```

Two things are wrong with it and only one of them is about precision.

**The stated reason does not hold.** `get_assd_mesh` calls pymskt's `pcu_sdf`, which casts
the query points *and* the mesh vertices to `float64` itself
(`get_faces_vertices(..., points_dtype=np.float64)`, `_as_c_contig(pts, np.float64)`). The
caller's dtype never reached the computation. Measured on a sphere pair, float32/float64,
float64/float32 and float64/float64 all return the identical value — so the cast could
never supply agreement, only remove precision on the way in.

**It mutated the caller's objects.** Not copies: `mesh` is the caller's reconstruction and
`orig_meshes[mesh_idx]` is the caller's ground-truth mesh, and both came back downcast.
Conditional on the flag, too — a caller that scored chamfer only, then added ASSD later,
lost precision on meshes it still held and had no way to notice.

### How to tell whether one of your runs is affected

Your ASSD numbers are affected only if the meshes you passed were float64. If they came
from `create_mesh`, from `reconstruct_mesh`, or from reading a `.vtk` through pymskt, they
were float32 and nothing changed. If you built them yourself at double precision — from a
numpy array, or a mesh library that keeps doubles — the ASSD you recorded came from a
float32 round-trip of your points.

**The size of it:** measured on a sphere pair perturbed below float32 resolution, the ASSD
moves by **7.2e-09**. That is far below any tolerance NSM reports against, so this is a
correctness note rather than a call to re-run anything.

*Pinned by:* `test_caller_object_mutation.TestSite2TheAssdDowncast`, which asserts the
caller's dtypes survive the call, that the chamfer path still does not touch them, and that
the reported ASSD is unchanged for the float32 meshes production actually passes.

## 29. `reconstruct_mesh`'s random-point sampling widened from sigma 0.001 to 0.01

| | |
|---|---|
| **Affected** | `reconstruct_mesh(get_rand_pts=True)` **without** an explicit `sigma_rand_pts`. Aug 2026 → |
| **Unaffected** | `get_rand_pts=False`, which is what both shipped ShapeMedKnee configs set, so the sampling does not happen at all; any caller that passes `sigma_rand_pts` explicitly, which `kneepipeline`'s `steps/run_nsm.py` does; `get_mean_errors`, whose default was already 0.01 |
| **Severity** | Silent — a 10× wider Gaussian around the surface, so a different sample set |
| **Changed in** | `slice-n-phase2-close`, Aug 2026 (plan §8.0.N, [#56](https://github.com/gattia/nsm/issues/56)) |

### What was wrong

`sigma_rand_pts` is the width of the Gaussian the off-surface points are drawn from. It
passed through three layers under one name with two values: `reconstruct_mesh` defaulted to
**0.001**, `get_mean_errors` to **0.01**, and the shipped config's `sigma_rand_pts_recon` is
**0.01**. So the same knob meant two different things depending on which layer you entered
at, and neither end said so.

Resolved to 0.01 — the value the ShapeMedKnee configuration uses and the one every path
that reads a config already took. This is a **default** change, not an algorithm change:
a caller who passed the value explicitly is unaffected, in either direction.

### How to tell whether one of your runs is affected

Two conditions, both required: you called `reconstruct_mesh` (not `get_mean_errors`) with
`get_rand_pts=True`, and you did not pass `sigma_rand_pts`. If either is false, nothing
moved. If both are true, your off-surface points were drawn at a tenth of the width they
will be drawn at now; pass `sigma_rand_pts=0.001` to reproduce the old draw.

*Pinned by:* `test_reconstruct_mesh_contracts.TestTheKnobsThatDifferByLayer`, which reads
the defaults off the signatures rather than restating them.
