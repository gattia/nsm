# Known issues history

A record of bugs that silently changed NSM's numerical behaviour, and the exact date and
configuration ranges each one affects.

**Why this file exists.** For research code, "which of my results are affected by this
bug?" is a question that has to be answerable years later, by someone who was not there.
A fix commit and a code comment cannot answer it. Every entry below should let a reader
determine, for a run they have on disk, whether that run is affected and what to do about it.

**When to add an entry.** Any time a fix changes the numerical output of training or
reconstruction for inputs that previously ran without error. Bugs that always crashed do
not need an entry — nobody has results from them.

---

## 1. Learning-rate schedules applied swapped (model ↔ latent codes)

| | |
|---|---|
| **Affects** | May 2023 → Aug 2026 |
| **Affected optimizers** | `Adam`, `AdamW` |
| **Unaffected optimizers** | `schedule_free_AdamW`, `schedule_free_SGD` |
| **Severity** | Silent — wrong numerics, no error, no warning |
| **Fixed in** | `fix-lr-schedule-mapping`, Aug 2026 |
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

### Why schedule-free runs are unaffected

The training loop skips the LR adjustment entirely for those optimizers:

```python
if not ("schedule_free" in config["optimizer"]):
    adjust_learning_rate(config["lr_schedules"], optimizer, epoch)
```

They therefore kept `get_optimizer()`'s correct initial assignment for the whole run.
**Do not swap the schedule entries in a `schedule_free_*` config.**

### How to tell whether one of your runs is affected

1. Check `model_params_config.json` in the experiment directory.
2. If `"optimizer"` is `"Adam"` or `"AdamW"`, and the run predates the fix → **affected**.
3. If `"optimizer"` starts with `"schedule_free"` → **not affected**.
4. If `"lr_schedule_convention"` is present, the run was configured after the fix and its
   mapping is explicit.

For an affected run, the learning rates actually used were:

- latent codes ← `LearningRateSchedule[0]`
- model/decoder ← `LearningRateSchedule[1]`

### How to reproduce an affected run under fixed code

Add one key to the run's config, leaving `LearningRateSchedule` untouched:

```json
"lr_schedule_convention": "legacy_swapped"
```

This swaps the two entries internally so the effective learning rates match the historical
run exactly. Verified by `TestHistoricalEquivalence` in
`testing/NSM/test_lr_schedules.py`, which asserts equality against a reimplementation of
the pre-fix mapping across a range of epochs.

For new runs, use `"lr_schedule_convention": "v2"` — index 0 = model, index 1 = latent codes.

### Migration guard

A pre-fix config run on fixed code would otherwise train with a different mapping than it
did historically, with no error. So an `Adam`/`AdamW` config that does not declare
`lr_schedule_convention` now **raises** with a message explaining both options.
`schedule_free_*` configs default to `v2` silently, having never been ambiguous.

### What this did to the shipped defaults

The default config was inherited from DeepSDF's reference `specs.json`, which lists the
network LR first and the latent LR second:

| | Entry 0 | Entry 1 |
|---|---|---|
| Intended | model `0.0005` | latent `0.001` |
| **Actually applied** | **latent `0.0005`** | **model `0.001`** |

So the reference convention — latents learn *faster* than the decoder — was inverted. Every
affected run trained its decoder at 2× the intended rate and its latents at half.

This fix swaps the two entries in the shipped default configs, which **preserves the
historical effective behaviour** (model `0.001`, latent `0.0005`) rather than restoring
DeepSDF's intent. That is a deliberate choice for continuity with the tuned production
models — see the open action below. It is not an endorsement of those values.

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

- `NSM/utils.py` — `get_optimizer`, `adjust_learning_rate`, `resolve_lr_schedule_convention`
- `testing/NSM/test_lr_schedules.py` — regression and equivalence tests
- `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md` §4 — this fix as the migration template

---

## 2. Sigma sampling coordinate space depends on `scale_jointly`

| | |
|---|---|
| **Affects** | Ongoing — not yet fixed |
| **Severity** | Silent — ~100× over/under-sampling with no error |
| **Tracking** | Issue #3; `planning/BREAKING_CHANGE_PROPOSAL.md` |

`SDFSamples` interprets `sigma_near` / `sigma_far` in two different coordinate spaces
depending on the `scale_jointly` flag:

- `scale_jointly=False` — sigma sampling happens **after** per-mesh normalization, so
  sigma is in normalized `[-1, 1]` cube units (typical values 0.01–0.1).
- `scale_jointly=True` — sigma sampling happens **before** normalization, so sigma is in
  original mesh units, e.g. mm (typical values 0.5–5.0).

The same sigma value therefore means two very different things, and using values tuned for
one mode in the other changes effective sampling density by roughly 100× — silently.

**How to tell whether a run is affected:** check `scale_jointly` and the sigma values
together in `model_params_config.json`. Small sigmas (< 0.2) with `scale_jointly=True`, or
large sigmas (> 0.5) with `scale_jointly=False`, indicate a probable mismatch.

**Planned fix:** standardize on original coordinate space with an explicit
`sigma_coordinate_space` parameter and a migration path. Scheduled into Phase 4 of
`.claude/plans/NSM_CODE_HEALTH_REFACTOR.md`. It must ship with a migration guard of the
same shape as issue 1's.

---

## 3. `train_deep_sdf_multi_head` optimizes only the last model

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

The module now emits a `DeprecationWarning`. Use `NSM.train.train_deep_sdf` with
`objects_per_decoder > 1` instead. Whether to repair or delete this file is a Phase 0
decision in `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md`.
