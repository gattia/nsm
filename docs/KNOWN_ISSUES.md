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
- `NSM/_lr_migration.py` — the migration error text. Not permanent API; delete the file
  once no config still in use predates the `Target` key.
- `testing/NSM/test_lr_schedules.py` — regression and equivalence tests
- `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md` §4 — this fix as the migration template

---

## 2. Sigma sampling coordinate space depends on `scale_jointly`

| | |
|---|---|
| **Affects** | Ongoing — not yet fixed |
| **Severity** | Silent — ~100× over/under-sampling with no error |
| **Tracking** | Issue #3; `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` |

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

---

## 4. Sampling was never seeded — no run before Aug 2026 is reproducible

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
