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

| Defect | Severity | Issue |
|---|---|---|
| Cache key does not cover what changes cached content | **High** — silently wrong training data | [#19](https://github.com/gattia/nsm/issues/19) |
| Sigma coordinate space depends on `scale_jointly` | **High** — ~100× over/under-sampling | [#3](https://github.com/gattia/nsm/issues/3) |
| `padding` absent from checkpoints | **High** — silent wrong-scale sampling | [#26](https://github.com/gattia/nsm/issues/26) |
| `include_surf_in_pts` appends a leaked loop variable | **High** | [#17](https://github.com/gattia/nsm/issues/17) |
| Parameters accepted and never read | Medium — **read the traps first** | [#20](https://github.com/gattia/nsm/issues/20) |
| `xyz_in_all` accepted and never read | Medium — silent no-op | [#20](https://github.com/gattia/nsm/issues/20) |
| A `None` surface cannot build | Medium — advertised feature, unusable | [#67](https://github.com/gattia/nsm/issues/67) |
| Every VAE layer stored twice | Medium — 1.92× checkpoints | [#27](https://github.com/gattia/nsm/issues/27) |
| `reconstruct_mesh` early return drops requested keys | Medium | [#29](https://github.com/gattia/nsm/issues/29) |
| `reconstruct_mesh` raises `KeyError: 'pts'` on one branch | Medium | [#15](https://github.com/gattia/nsm/issues/15) |
| `n_pts_random` accepted and discarded | Medium | [#16](https://github.com/gattia/nsm/issues/16) |
| `sample_difficulty_lx` shipped but unimplemented | Medium | [#18](https://github.com/gattia/nsm/issues/18) |
| `enforce_minmax` clamps predictions | Medium — config semantics | *none — a docs/design call, see below* |
| `Pool` deadlocks after an in-process build | Low — hangs, does not corrupt | [#25](https://github.com/gattia/nsm/issues/25) |
| `train_deep_sdf` returns nothing | Low — blocks observability | [#28](https://github.com/gattia/nsm/issues/28) |

---

## `datasets/sdf_dataset.py`

### Cache key omits parameters that change what is cached

`MultiSurfaceSDFSamples.get_hash_params` does not include `mesh_to_scale`, `uniform_pts_buffer`
or `subsample`, all of which change what gets written. Two runs differing only in one share an
`md5`, and with `load_cache=True` — what both shipped configs use — the second silently trains
on the first's data.

Not equally severe: `mesh_to_scale` invalidates **every** array (it decides which surface
drives centering and normalization, so the two runs are in different coordinate frames);
`uniform_pts_buffer` moves the points; `subsample` touches only the index arrays. Fix
`mesh_to_scale` first.

**Measured.** One subject cached at `subsample=64` and reloaded at `512`: the small
surface's interior fraction in a batch fell 0.258 → 0.059, a **4.4× under-representation**
of interior samples, with `equal_pos_neg=True` set throughout. `sdf_pos_neg_idx` sizes the
repeated index arrays for the `subsample` in force when the cache was written
(`MultiSurfaceSDFSamples.sdf_pos_neg_idx`); reload with a larger one and `__getitem__` tops
the batch up with uniform random points (`MultiSurfaceSDFSamples.__getitem__`). In a real
dataset the small surface is the cartilage.

The reload guard that should have caught this compares `len(data["pos_idx"])` against the
number of *meshes* (`MultiSurfaceSDFSamples.get_sample_data_dict`), never against the
subsample the arrays were built for.

**How to tell whether you are affected:** if you reused a cache directory across runs that
differed in `mesh_to_scale`, `uniform_pts_buffer` or `subsample`, the later run trained on the
earlier one's data. Different cache directory per configuration, and you are fine.

*Fix:* [#19](https://github.com/gattia/nsm/issues/19), bundled with the two entries below
because all three invalidate every cached `.npz` — one regeneration, not three. *Pinned by:*
`test_dataset_cache.TestUnhashedParametersCollide` (5 tests).

### Cache key omits mesh content

The key is `md5(params + mesh paths)`. Edit a mesh in place and the key does not move, so
the stale `.npz` is served and you train on the old geometry. Same class as the two entries
either side of it: something that changes cached content is not in the key.

**How to tell whether you are affected:** if you have ever edited or re-exported a mesh
without moving or renaming it, any run after that reused the pre-edit samples.

*Fix:* [#19](https://github.com/gattia/nsm/issues/19). *Not currently pinned by a test.*

### `reference_mesh` is hashed by memory address

`SDFSamples.create_hash` stringifies every hash parameter, and `reference_mesh` is one of them
in both classes' `get_hash_params`. `str(Mesh(...))` begins `Mesh (0x7f478a24ce20)`, so the key
is per-object: two `Mesh` instances from the same file hash differently, and the same instance
hashes differently in the next process. **The cache can never hit** — you pay full regeneration
every run. Passing the reference as a path string is stable, and is the workaround people are
implicitly relying on.

*Fix:* [#19](https://github.com/gattia/nsm/issues/19). *Pinned by:*
`test_dataset_cache.TestReferenceMeshHashing`.

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

### `scale_jointly` cannot run in memory, and would drop its buffer if it could

`scale_jointly=True` with `store_data_in_memory=True` raises `KeyError: 'new_pts_0'` at
construction, on both dataset classes: the in-memory branch of
`norm_and_scale_all_meshes` reads the flattened `new_pts_0`-style keys that exist only in
the `.npz` cache layout, while in-memory sample dicts hold `new_pts` as a list. The
combination has never constructed, so no results are affected.

The same branch also omits `joint_scale_buffer` — the disk branch grows the shared scale
by `1 + buffer`, the in-memory branch does not — so a fix for the `KeyError` alone would
quietly put in-memory runs in a different coordinate frame than disk-backed ones. The pin
asserts the buffered domain and `raises=KeyError`, forcing both halves to land together.

*Fix:* [#69](https://github.com/gattia/nsm/issues/69). *Pinned by:*
`test_dataset_cache.TestScaleJointlyInMemory::test_an_in_memory_dataset_lands_inside_the_buffered_domain`.

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

*Fix:* [#25](https://github.com/gattia/nsm/issues/25). *Worked around in:*
`test_dataset_cache.TestSeedDerivation`, which builds its two datasets in separate
subprocesses.

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

### `F401` is project-ignored, so unused imports do not appear in `make lint`

`.flake8`'s `extend-ignore` has carried `F401` since before the Aug 2026 lint work. `make
lint` reports zero violations, and separately there are **43** unused imports it will never
show. "flake8 is at zero" is true and does not mean the imports are gone.

*Fix:* not filed. Removing the ignore is a judgement call — several are deliberate
re-exports in `__init__.py` files, which is the usual reason `F401` gets ignored wholesale.

## `models/triplanar.py`

### `padding` is not in the checkpoint, and the mismatch is silent

`TriplanarDecoder.padding` scales query coordinates before they index the feature planes
(`TriplanarDecoder.normalize_coordinates`). It is **not a learned parameter**, so a
checkpoint trained at one value loads
cleanly under strict `load_state_dict` at another and then samples at the wrong scale.

**Measured.** A model built at `padding=0.35`, saved, and loaded through `load_model` with a
config that omits `padding` (`loader._get_triplanar_params` defaults it to 0.1) loads without
error and computes a maximum absolute SDF difference of **0.063**. The output is `tanh`-bounded
to (−1, 1), so that is ~3% of the full range, not a rounding artefact. Stating `padding` in the
config restores bitwise-identical output.

`kneepipeline/steps/run_nsm.py:94-112` passes 15 of `TriplanarDecoder`'s 16 meaningful
arguments, and `padding` is the one it omits — so the shipped consumer is exposed.

**How to tell whether you are affected:** if the model was trained at a `padding` other than
0.1 and your config or caller does not state it, every SDF it computes is wrong by up to
~3% of the output range. Stating `padding` in the config restores bitwise-identical output.

*Fix:* [#26](https://github.com/gattia/nsm/issues/26). *Pinned by:*
`test_model_roundtrip.TestPaddingIsNotInTheCheckpoint`.

### `normalize_coordinates` ignores its own `padding` argument

The signature is `TriplanarDecoder.normalize_coordinates(self, query, plane, padding=0.1)`
and the body reads `self.padding`. Accepted, no effect, at any value. Same defect class as
the entry above and as `get_pts_center_and_scale` — which is why they should be swept
together rather than one at a time.

> ⚠️ **The obvious fix is worse than the bug**, and the test pinning this rewards the wrong
> one. Read [#20](https://github.com/gattia/nsm/issues/20)'s traps before changing anything
> here.

*Pinned by:*
`test_model_roundtrip...::test_normalize_coordinates_must_honour_its_padding_argument`.

### Every VAE layer is stored twice in the state dict

`VAEDecoder.__init__` registers each layer twice — once in `self.layers`, a `ModuleList`,
and again in `self.decoder = nn.Sequential(*self.layers)`. Both are child modules,
so `state_dict()` emits every VAE tensor under two aliased names.

Loading is unaffected — the names alias one parameter. Two things are:

- **Checkpoint size.** All three shipped models store 39.96M elements for 20.80M
  parameters, **1.92×**. The 275 MB files would be about 143 MB.
- **Checkpoint surgery.** Editing by key silently loses the edit if only one name is
  written. Not hypothetical: the first draft of `test_the_comparison_can_fail` did exactly
  that and looked like a passing round trip.

**How to tell whether you are affected:** every NSM checkpoint is. Your models are correct —
this costs disk, not accuracy — but any tooling that edits a checkpoint by key needs to write
both names.

*Fix:* [#27](https://github.com/gattia/nsm/issues/27), which is a checkpoint-format break in
both directions and needs a migration shim. *Pinned by:*
`test_model_roundtrip.TestAliasedCheckpointEntries`.

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

## `models/deep_sdf.py`

### `xyz_in_all` is accepted and never read

`deep_sdf.Decoder.__init__` takes `xyz_in_all`, documents it as "for deepSDF decoder, include
XYZ at each layer", and never stores it. `forward` computed `xyz = input_[:, -3:]` and never
used it either — the vestige of the unimplemented feature, removed when `NSM/` was brought
to zero flake8 violations. `default_config.json` ships the key and `loader` plumbs it
through in four places, so a config setting `xyz_in_all: true` is silently a no-op.

Same class as `normalize_coordinates`' `padding` and `get_pts_center_and_scale`'s `center=`
— an argument accepted and discarded — so it belongs to the same sweep.

*Fix:* [#20](https://github.com/gattia/nsm/issues/20). *Not pinned by a test:* found by
reading, not by a failure.

## `train/train_deep_sdf.py`

### `enforce_minmax` clamps the prediction, not just the target

`train_epoch` clamps `pred_sdf` as well as the target, and `torch.clamp` passes no
gradient outside its bounds. Every sample predicted outside ±`clamp_dist` therefore
contributes **exactly zero gradient**, however wrong it is.

**Measured.** On a freshly built triplanar decoder, **44.6%** of predictions already fall
outside ±0.1 before the first step. The shipped `default_config.json` uses `clamp_dist: 0.1`;
both ShapeMedKnee configs use `1.0`.

Whether that stalls a given run is configuration-dependent — an earlier claim that it always
does was **false**, and was withdrawn after being run. The defect is that the name and the
docs describe a target transform while the behaviour is a training-dynamics knob. This is a
documentation-or-decision call, not a bug fix, so it has **no issue** — it belongs with the
config work in `SCOPE.md` §2.2.
*Pinned by:* `test_training_regression.TestClampedPredictionGradients`.

### `train_deep_sdf` returns nothing

`train_deep_sdf` ends in a bare `return`. `train_epoch` builds a full `log_dict` per epoch and
`train_deep_sdf` forwards it only to `wandb`, so a caller without a wandb key can learn
nothing about a run except by reading checkpoints back off disk. The regression harness has
to wrap `train_epoch` to observe anything (`testing/NSM/regression/_harness.py`) — fixing
this deletes that wrapper.

*Fix:* [#28](https://github.com/gattia/nsm/issues/28).

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

## `reconstruct/main.py`

### The early return drops keys the caller asked for

When the decoder's mean shape has no zero level set, `reconstruct_mesh` returns early at
with only `{mesh, latent, assd_*}`, ignoring `return_registration_params`,
`return_timing` and `orig_mesh`. The two result shapes are not interchangeable and the
consumer reads `result["center"]` unconditionally (`kneepipeline/steps/run_nsm.py:230`).

Sharper than the missing keys: **the result looks successful.** `mesh` is `[None, None]`,
`assd_*` are `nan`, and `latent` is a correctly-shaped `(1, latent_size)` tensor of zeros —
the untouched `mean_latent`, never fitted. A caller checking "did I get a latent" gets yes.
*Fix:* [#29](https://github.com/gattia/nsm/issues/29). *Pinned by:*
`test_reconstruction_regression.TestDecoderWithNoZeroLevelSet` (5 tests).

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
| **Affects** | Uniform ("random") training samples: (a) any run with a nonzero `uniform_pts_buffer` — the parameter exists since Jan 2025 (`48c5f60`) and the shipped `default_config.json` sets `dataset_uniform_pts_buffer: 0.2`; (b) any **single-surface** run with `norm_pts=True` |
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

Check the run's dataset settings: a nonzero `uniform_pts_buffer`
(`dataset_uniform_pts_buffer` in configs), or a single-surface dataset with
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
