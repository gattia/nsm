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
