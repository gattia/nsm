# Test harness notes — findings

**Companion to `planning/TEST_HARNESS_HANDOFF.md` §4.1.** Written 2026-08-15 while building
the Phase 3 §7.1 regression harness (`testing/NSM/regression/`), against tag `v0.1.0` on
branch `plan-code-health-refactor`.

Everything below was **executed**, not read. Each entry gives what was expected, what
happened, the `file:line`, and the test that demonstrates it. Nothing here was fixed — the
harness pins current behaviour, bugs included, which is what makes a later fix visible as a
test changing.

**This document is the evidence.** What to do about each of these is tracked as a work list
in [`DEFECT_WORKLIST.md`](DEFECT_WORKLIST.md), ordered by file and function so the entries
surface while the code is being refactored function by function. Deliberately not GitHub
issues yet: the refactor is expected to churn this code, so whatever is still open at the end
is what is worth filing. The one exception is the sampler seed, which is upstream and filed as
[gattia/pymskt#54](https://github.com/gattia/pymskt/issues/54).

`docs/AUDIT_FINDINGS.md` was deliberately not edited. Entries marked **[register]** confirm
or refine something already in it; entries marked **[new]** were not on the list.

Measured environment: Linux, CPU + one CUDA device, Python 3.9.25, torch 2.8.0+cu128,
numpy 2.0.2, pytest 8.4.2.

---

## Contents

- [1. Confirmed defects](#1-confirmed-defects) — 12 entries
- [2. Refined or corrected claims](#2-refined-or-corrected-claims) — 3 entries
- [3. Testability findings](#3-testability-findings) — 2 entries
- [4. Measurements worth keeping](#4-measurements-worth-keeping)
- [5. Out of scope, seen and left alone](#5-out-of-scope-seen-and-left-alone)

---

## 1. Confirmed defects

### 1.1 The dataset cache key omits three parameters that change what it stores **[register]**

*Expected:* the register lists `mesh_to_scale`, `uniform_pts_buffer` and `subsample` as
absent from `get_hash_params`, flagged as inference and as the highest-severity finding on
the list.

*Happened:* all three confirmed. `MultiSurfaceSDFSamples.get_hash_params`
(`NSM/datasets/sdf_dataset.py:1973-1999`) lists eleven scalars plus the per-surface point
counts, probabilities and sigmas. It does not include any of the three. Two datasets
differing only in one of them produce an identical `md5`, and with `load_cache=True` — the
setting both shipped configs use — the second silently gets the first's file.

They are not equally severe, which the register could not have known:

| Parameter | Cached arrays that differ |
|---|---|
| `mesh_to_scale` | `pts`, `sdfs`, `new_pts_0`, `new_pts_1` — **everything**, because it decides which surface drives centering and normalization. The two runs are in different coordinate frames. |
| `uniform_pts_buffer` | `pts`, `sdfs`, `pos_idx_*`, `neg_idx_*` — it sets the bounds the uniform points are drawn from. |
| `subsample` | `pos_idx_*`, `neg_idx_*` only. The points are unaffected. |

*Consequence of the `subsample` case, measured.* `sdf_pos_neg_idx` repeats the index arrays
just far enough for the `subsample` in force when the cache was written
(`sdf_dataset.py:2029-2030`). Reload with a larger one and there are too few entries:
`__getitem__` takes what exists and tops the batch up with uniformly random points
(`:2122-2127`). One subject, cache written at `subsample=64` and reloaded at `512`: the
small surface's interior fraction in a batch fell from 0.258 to 0.059 — **a 4.4×
under-representation of interior samples**, with `equal_pos_neg=True` set throughout. In a
real dataset the small surface is the cartilage.

The reload guard that could have caught this compares `len(data["pos_idx"])` against the
number of *meshes* (`:1764-1771`), never against the subsample the arrays were built for.

*Tests:* `test_dataset_cache.TestUnhashedParametersCollide` — one test per parameter, plus
`test_a_changed_parameter_must_not_reuse_the_previous_runs_cache` and
`test_equal_pos_neg_must_hold_after_a_subsample_change`.

### 1.2 A `Mesh` object as `reference_mesh` is hashed by memory address **[register]**

*Expected:* `SCOPE.md` §4 states the cache never hits in this case.

*Happened:* confirmed. `create_hash` stringifies every hash parameter
(`sdf_dataset.py:1437`) and `reference_mesh` is one of them (`:1406`, `:1981`).
`str(Mesh(...))` begins `Mesh (0x7f478a24ce20)`, so the key is per-object: two `Mesh`
instances built from the same file hash differently, and the same instance hashes
differently in the next process. Passing the reference as a path string is stable and is
the workaround.

*Tests:* `test_dataset_cache.TestReferenceMeshHashing`.

### 1.3 `random_seed` seeds nothing, and the near-surface sampler cannot be seeded **[new]**

This one shaped the whole harness, so it is written up at length.

*Expected:* the handoff's spec (§3) says "2–3 analytic meshes, 8 epochs, CPU, fixed seed".
`SDFSamples(random_seed=...)` is documented as "Random seed. Defaults to None."
(`sdf_dataset.py:791`).

*Happened:* `random_seed` is stored on the instance (`:873`) and appended to the cache key
(`:1433-1434`). That is all it does. `grep -rn "np.random.seed\|manual_seed\|default_rng"
NSM/` returns **nothing** outside `train/deprecated/` — NSM calls no seeding function
anywhere.

That leaves two sampling paths with very different reproducibility and nothing at the
constructor to tell them apart:

- **Uniform** (`sigma_near`/`sigma_far` of `None`) → `get_rand_uniform_pts` →
  `np.random.uniform` (`:29-53`, `:300-303`). Seedable by the caller with
  `np.random.seed`; verified bit-identical across builds and across processes.
- **Near-surface** (`sigma` set — what every shipped config uses) →
  `Mesh.rand_pts_around_surface`, which cannot be seeded by a caller at all. It has **two**
  independent unseedable draws, and the first draft of this note found only one of them:

  1. *Base surface points.* `pymskt.mesh.meshTools.rand_sample_pts_mesh:1543` calls
     `pcu.sample_mesh_random(points, faces, n_pts)`. The signature is
     `sample_mesh_random(v, f, num_samples, random_seed=0)` and its own docstring says
     **"Passing in 0 will use the current time"** — so the default is not "seed 0", it is
     "seed from the wall clock". `sample_mesh_poisson_disk`, the `bluenoise` branch, has the
     same parameter and default.
  2. *Perturbation offsets.* `pymskt/mesh/meshes.py:361-363` uses
     `np.random.default_rng().multivariate_normal`. `default_rng()` with no argument seeds
     from OS entropy and is independent of the legacy global state, so `np.random.seed()`
     does not reach it either. This one is easy to miss because it reads as numpy randomness.

  Verified: same `np.random.seed`, same inputs, different points; and both become
  reproducible once given an explicit seed. **Reported upstream as
  [gattia/pymskt#54](https://github.com/gattia/pymskt/issues/54)**, which is where the fix
  belongs — NSM cannot seed through this from the outside.

*Consequences.*

1. **A cold-cache rebuild of the training data for any shipped config cannot be
   reproduced.** 647 and 551 both use numeric `sigma_near`/`sigma_far`.
2. **A warm cache hides it.** Two runs with the same `random_seed` get the same cache key,
   so the second reuses the first's `.npz` and looks perfectly reproducible. Point them at
   different cache directories and the illusion goes. This is the landmine shape: the
   parameter that appears to control reproducibility is the parameter that arranges for
   you not to notice its absence.
3. **The harness had to work around it.** Every fixture runs on the uniform path. That is a
   restriction on the harness, not on NSM, and it is liftable the day `pymskt` threads a
   seed through — a one-argument change there, or an NSM-side change to
   `read_meshes_get_sampled_pts`, which is out of scope for a tests-only diff.

*Tests:* `test_dataset_cache.TestSeeding` — four tests: the uniform path is reproducible
(passes), the warm cache makes `random_seed` look like it works (passes — that is a cache hit
behaving correctly, and it is why nobody noticed), and two strict xfails for the unseedable
sampler and for `random_seed` itself. Both xfails turn the suite red the day pymskt#54 lands.

### 1.4 `store_data_in_memory=True` raises on the first item (multi-surface only) **[new]**

*Expected:* an advertised constructor argument; `store_data_in_memory` is a documented key
in both shipped configs.

*Happened:* `MultiSurfaceSDFSamples.__getitem__:2158-2162` reads `time_` and `size`, which
are only bound in the `store_data_in_memory is False` branch (`:2050-2057`). With
`store_data_in_memory=True` and the default `test_load_times=True`, the first `__getitem__`
raises `UnboundLocalError: local variable 'time_' referenced before assignment`.

The single-surface `SDFSamples.__getitem__:1563` guards the same block correctly
(`if (self.test_load_times is True) and (self.store_data_in_memory is False)`), so the two
classes disagree about the same option.

The apparent workaround — also passing `test_load_times=False` — produces items with only
`{"xyz", "gt_sdf"}`, and `train_epoch` reads all four timing keys unconditionally
(`NSM/train/train_deep_sdf.py:578-581`). So there is no combination of these two flags that
both constructs and trains.

*Tests:* `test_dataset_cache.TestConfigurationsThatDoNotRun` — three tests covering the
crash, the partial workaround, and why the workaround does not reach training.

### 1.5 `p_near_surface=0` crashes inside `point_cloud_utils` **[new]**

*Expected:* a legal request for "no near-surface samples".

*Happened:* `get_pt_sample_combos` emits a `[0, sigma]` combo and
`get_sample_data_dict:1820-1841` calls the sampler with it regardless, so `pcu` raises
`ValueError: Invalid input point cloud with zero points`. Same for
`p_further_from_surface=0`. An instance of the register's "constructible-but-uncallable
configuration" class.

*Test:* `test_dataset_cache.TestConfigurationsThatDoNotRun::test_zero_sampling_probability_must_sample_nothing`.

### 1.6 `get_pts_center_and_scale` ignores `center=` and `scale=`, and mutates its input **[register]**

*Expected:* the register lists `sdf_dataset.py:87` — `center=` / `scale=` rebound before
they are read — and `:91` — silent in-place mutation of caller data.

*Happened:* both confirmed by execution. `center` is overwritten at `:88` with
`np.mean(pts, axis=0)` and `scale` at `:94`, so `center=False, scale=False` centres and
scales anyway. `pts -= center` at `:91` writes through to the caller's array. All three
in-repo call sites pass `np.copy(...)`, so the convention exists only as a habit at the
call sites — a fourth caller written without that habit gets silently corrupted input.

*Tests:* `test_dataset_cache.TestPointCenteringAndScaling`.

### 1.7 `LOC_SDF_CACHE` is read at import time **[new]**

*Expected:* an environment variable that redirects the SDF cache.

*Happened:* it is read inside a **default argument** —
`loc_save=os.environ.get("LOC_SDF_CACHE", ...)` at `sdf_dataset.py:820-822` and
`:1609-1611` — so it is evaluated once when the module is imported. Setting it afterwards
has no effect and the caller silently writes to `~/.cache/nsm_sdf_cache`.

The downstream consumer does exactly this: `kneepipeline/steps/run_nsm.py` sets
`os.environ["LOC_SDF_CACHE"] = ""` *after* `from NSM.reconstruct import reconstruct_mesh`
on the line above. Harmless there, because `reconstruct_mesh` uses
`read_meshes_get_sampled_pts` directly and never constructs a dataset — but the line does
not do what it looks like it does.

*Test:* `test_dataset_cache.TestCacheLocationDefault`.

### 1.8 `reconstruct_mesh`'s early return drops keys the caller asked for **[new]**

*Expected:* `return_registration_params=True` means `center`, `scale` and `icp_transform`
are in the result.

*Happened:* when the decoder's mean shape has no zero level set, `reconstruct_mesh` returns
early at `NSM/reconstruct/main.py:946-966` with only `{mesh, latent, assd_*}` — ignoring
`return_registration_params`, `return_timing`, and `orig_mesh`. The two result shapes are
not interchangeable, and the consumer reads `mesh_result["center"]` unconditionally
(`kneepipeline/steps/run_nsm.py:230`).

Worse than the missing keys: the result *looks* fine. `mesh` is `[None, None]`, `assd_0`
and `assd_1` are `nan`, and `latent` is a correctly-shaped `(1, latent_size)` tensor of
zeros — the untouched `mean_latent`, never fitted. A caller checking "did I get a latent"
gets yes.

*Tests:* `test_reconstruction_regression.TestDecoderWithNoZeroLevelSet` — four tests. The
precondition is stated directly with a small test double rather than an under-trained
model, because whether a real model reaches that state is configuration-dependent (see
§2.2).

### 1.9 `padding` is not in the checkpoint, and the mismatch is silent **[register]**

*Expected:* `SCOPE.md` §3.1 item 2 — `padding` is not a learned parameter, so a checkpoint
trained at a different value loads cleanly under strict `load_state_dict` and then samples
the feature planes at the wrong scale.

*Happened:* confirmed, with a number. A model built at `padding=0.35`, saved, and loaded
through `load_model` with a config that omits `padding` (`loader.py:133` defaults it to
0.1) loads without error and computes a **maximum absolute SDF difference of 0.063**. The
decoder's output is `tanh`-bounded to (−1, 1), so that is ~3% of the full output range —
not a rounding artefact. Stating `padding` in the config restores bitwise-identical output.

`kneepipeline/steps/run_nsm.py:94-112` passes 15 of `TriplanarDecoder`'s 16 meaningful
arguments and `padding` is the one it omits.

*Tests:* `test_model_roundtrip.TestPaddingIsNotInTheCheckpoint`.

### 1.10 `TriplanarDecoder.normalize_coordinates` ignores its own `padding` argument **[new]**

*Expected:* the signature is `normalize_coordinates(self, query, plane, padding=0.1)`
(`NSM/models/triplanar.py:312`).

*Happened:* the body reads `self.padding` (`:322`). The parameter is accepted and has no
effect at any value. A second instance of the same class of defect as §1.9 and in the same
place, which is the sort of thing `CLAUDE.md`'s "fix the class, not the instance" is aimed at.

*Test:* `test_model_roundtrip.TestPaddingIsNotInTheCheckpoint::test_normalize_coordinates_must_honour_its_padding_argument`.

### 1.11 Every VAE layer is stored twice in the state dict **[new]**

*Expected:* a checkpoint holds one entry per parameter.

*Happened:* `VAEDecoder` registers each layer twice — once in `self.layers`, a `ModuleList`
(`triplanar.py:58-97`), and again in `self.decoder = nn.Sequential(*self.layers)` (`:99`).
Both are child modules, so `state_dict()` emits every VAE tensor under two names that alias
the same storage.

Loading is unaffected: the names alias one parameter, so whichever is applied last wins and
it is the same data. Two things are affected.

**Checkpoint size.** Measured on the three shipped models:

| Model | Entries | Aliased groups | Elements stored | Parameters | Ratio |
|---|---|---|---|---|---|
| `647_nsm_femur_v0.0.1/model/2000.pth` | 58 | 22 | 39,957,764 | 20,801,924 | **1.92×** |
| `551_nsm_femur_bone_v0.0.1/model/1150.pth` | 58 | 22 | 39,957,250 | 20,801,410 | **1.92×** |
| `231_nsm_femur_cartilage_v0.0.1/model/2000.pth` | 58 | 22 | 39,957,764 | 20,801,924 | **1.92×** |

The 275 MB files would be about 143 MB. Every NSM checkpoint ever shipped is roughly twice
the size it needs to be.

**Checkpoint surgery.** Editing a checkpoint by key — pruning, quantizing, patching a layer
— silently loses the edit if only one of the two names is written. Not hypothetical: the
first draft of `test_the_comparison_can_fail` in `test_model_roundtrip.py` did exactly that
and looked like a passing round trip. That test now perturbs every float tensor, with a
comment saying why.

*Tests:* `test_model_roundtrip.TestAliasedCheckpointEntries` — three tests, including the
demonstration that editing one alias is reverted by the other.

### 1.12 `enforce_minmax` clamps the prediction, not just the target **[new]**

*Expected:* `clamp_dist` bounds the SDF targets, as in DeepSDF's clamped L1.

*Happened:* `train_epoch` clamps `pred_sdf` as well (`NSM/train/train_deep_sdf.py:401`), and
`torch.clamp` passes no gradient outside its bounds. Every sample the decoder predicts
outside ±`clamp_dist` therefore contributes **exactly zero gradient**, however wrong it is.

Measured on a freshly built triplanar decoder (`tanh` output, zero latent, 2048 uniform
query points): **44.6%** of predictions already fall outside ±0.1 before the first step. The
shipped `NSM/configs/default_config.json` uses `clamp_dist: 0.1`; both ShapeMedKnee configs
use `1.0`. The harness uses 1.0.

This makes `clamp_dist` a training-dynamics knob, not just a target transform, which is not
what its name or its documentation suggests. It belongs with the config-documentation work
in `SCOPE.md` §2.2.

*Tests:* `test_training_regression.TestClampedPredictionGradients` — the gradient mechanism
in both directions, plus the measured dead fraction, baselined so a change to weight
initialization or `final_activation` shows up.

---

## 2. Refined or corrected claims

### 2.1 kneepipeline's "seed after `.cuda()`" rule does not reproduce here

`kneepipeline/CLAUDE.md` Known Issue 3 and `steps/run_nsm.py:176-181` state that
`torch.manual_seed()` must be called **after** `model.cuda()` because "`model.cuda()`
consumes CUDA random state", and attribute ~0.08 BScore differences to getting it wrong.

On torch 2.8.0+cu128 this is not observable. Both orderings produce an identical CUDA random
stream, in one process and across fresh processes, with and without a module transfer
between the seed call and the draw.

That does not make the consumer's ordering wrong — it is harmless, and the effect may have
been real on the torch build in use when it was written, or may have had a different cause.
It does mean the stated *reason* is not currently true. Pinned as a tripwire rather than
silently dropped: if a torch upgrade makes `.cuda()` consume RNG state again,
`test_gpu.TestSeedOrderingAroundCudaTransfer` goes red, which is exactly when someone needs
to know. Worth raising in the consumer repo; it is that repo's document, not NSM's.

### 2.2 My own first claim about `clamp_dist=0.1` was wrong, and the correction matters

The first version of §1.12 claimed that at `clamp_dist=0.1` a fresh triplanar decoder
saturates completely and **never** forms a zero level set. That came from one observed run
and was wrong as a general statement.

Executed: under the harness's config (3 subjects, 256 samples/object, model LR 0.005),
`clamp_dist=0.1` trains fine — loss 0.101 → 0.028 over 8 epochs — and produces surfaces at
both 8 and 60 epochs. Under a different config (2 subjects, 1024 samples, model LR 0.01) the
decoder did saturate to a constant +1.0 and produced no surface at all, at 50, 100 and 200
epochs.

So the true statement is the narrow one in §1.12: the clamp discards ~45% of the gradient
signal at init, and whether that stalls training is configuration-dependent. The broad
statement was removed from the tests along with the assertion it supported — a test
comparing "relative loss drop at 0.1 vs at 1.0" is meaningless anyway, since changing the
clamp changes the loss being measured.

Recorded because the handoff's caveat about the register applies to this document too, and
because it is the second time in this project's audit that a read-and-report claim survived
until someone ran it (`ARCHITECTURE.md` §7.1 is the first).

### 2.3 The three unhashed cache parameters are not equally severe

The register groups `mesh_to_scale`, `uniform_pts_buffer` and `subsample` under one finding.
All three are real, but `mesh_to_scale` invalidates every cached array while `subsample`
touches only the index arrays. If this gets fixed incrementally, `mesh_to_scale` is the one
to fix first. See the table in §1.1.

---

## 3. Testability findings

These are the two places where "this behaviour is untestable as written" applies, per the
handoff's instruction to report rather than solve them. Neither blocked the harness; both
forced a shape it would not otherwise have.

### 3.1 `train_deep_sdf` returns `None`, so no loss history is observable

`NSM/train/train_deep_sdf.py:272` is a bare `return`. `train_epoch` builds a full `log_dict`
per epoch (`:608-631`) and `train_deep_sdf` forwards it only to `wandb`
(`:265-266`). A caller without a wandb key can obtain nothing about a run except by reading
the checkpoints back off disk.

The harness therefore wraps `module.train_epoch` for the duration of a run
(`_harness.run_training`) rather than re-implementing the loop, so what it records is what
the real trainer did. If `train_deep_sdf` ever returns its history, the wrapper can be
deleted and the harness gets simpler. This is a small, self-contained API improvement with
an immediate testing payoff, and it belongs to whoever touches `train/` in Phase 4.

### 3.2 There is no public "build the model this config describes"

`load_model` requires a checkpoint path, which a fresh model does not have. The only public
alternative is to hand-roll the config→constructor mapping, which is exactly what the
downstream consumer does and how it loses `padding` (§1.9).

The harness reaches into `NSM.models.loader._get_triplanar_params`, a private function, so
that the model it trains is byte-identical to the model `load_model` builds — verified by
`test_model_roundtrip.TestModelConfigMapping`. That coupling is deliberate and documented at
the import: when the decoder registry of plan §8.1 lands, this is what should fail loudly.

`SCOPE.md` §3.1 already calls closing this gap "the single highest-value API change
available". This is a second, independent vote for it: a first-party test suite cannot build
a model from a config either.

---

## 4. Measurements worth keeping

**Harness cost.** 117 tests (97 passed + 20 xfailed), ~20 s locally with a GPU present, ~13 s
without (the 8 GPU tests skip). Full suite 159 + 1 skip in 13.1 s → **256 passed + 1 skipped +
20 xfailed in 33 s**. The 2-minute budget is not close to being at risk.

**Why 20 xfails rather than 20 passing characterization tests.** The handoff asked for current
behaviour to be pinned, bugs included. The first version did that by asserting the *broken*
behaviour, so those tests passed — and a suite reporting "274 passed" then says the opposite of
the truth about a library with twelve known defects in it. They now assert the behaviour NSM
*should* have, marked `xfail(strict=True)`:

- the report distinguishes "correct" (`passed`) from "known defect" (`xfailed`);
- `strict` means a fixed defect XPASSes and turns the suite **red**, naming its worklist item,
  so the fix cannot land without someone updating the test and ticking the list. Verified by
  temporarily honouring `center=`/`scale=` in `get_pts_center_and_scale`: both worklist #6
  xfails went `XPASS(strict)` and the suite failed;
- the pinning is not lost. Several assert the premise still holds before asserting the defect,
  so the mark cannot pass vacuously, and the measured numbers live in this document.

Known cost: an xfail that starts failing for an unrelated reason still reports `xfailed` and
hides it. Mitigated by keeping each body minimal and pointed at one thing.

**Coverage,** as a by-product and not a goal:

| Module | Before | After |
|---|---|---|
| `datasets/sdf_dataset.py` | 7% | **54%** |
| `train/train_deep_sdf.py` | 10% | **65%** |
| `reconstruct/main.py` | 24% | **42%** |
| `utils.py` | 55% | **75%** |
| `mesh/main.py` | 62% | **68%** |
| **Total** | **34%** | **53%** |

`reconstruct_mesh` went from one executed line — its `def` — to four distinct scenarios per
CPU run (six with a GPU present).

**The harness has been seen to fail.** Not only through the two in-suite deliberate breaks,
but against the real defect: `adjust_learning_rate` in `NSM/utils.py` was temporarily
reverted to the pre-Aug-2026 positional mapping — `lr_schedules[i]` to `param_groups[i]`,
against `get_optimizer`'s `[latent, model...]` ordering — and the harness went red with
**14 failures**, including:

```
learning_rates_per_epoch.latent: differs from baseline (rtol=0.0, atol=0.0).
  worst element [1]: baseline 0.0009000000000000001 observed 0.005 (abs diff 4.100e-03)
```

The failures spanned both halves: the per-epoch learning rates, the loss trajectory and its
components, the latent-norm trajectory, and — through the trained decoder — the fitted
latent, mesh geometry, vertex counts, ASSD and registration params of the reconstruction.
`NSM/utils.py` was restored afterwards and is untouched in the diff.

One detail from that run is worth keeping, because it looks like a flaw and is not: the
three `TestDeliberateBreak::test_swapping_lr_targets_fails_*` self-checks also failed. With
the positional mapping restored, transposing the two `Target` labels no longer changes
anything — the two wrongs cancel and the swapped run reproduces the correct baseline
exactly. Those tests failing is the harness reporting *"the deliberate break has stopped
breaking anything"*, which is precisely the signal you want when the LR path stops honouring
`Target`.

**Break magnitudes, and where the tolerances came from.** Tolerances were sized from these
rather than picked, so every numeric baseline has at least an order of magnitude between its
tolerance and the smallest signal it must catch. Transposing the two LR `Target` labels moves
the loss trajectory by **1.7 relative** and the final latent norms by **5.4e-2**. Moving one
input mesh vertex by 0.25 (a quarter of the bone radius) moves the fitted latent by
**7.0e-3**, the vertex-position deciles by **4.3e-3** and the surface centroid by **9.0e-4**.

An earlier draft used a 0.05 perturbation; that cleared the geometry tolerance by a factor of
1.4, which is not a margin. Worth recording as a method note: a deliberate break has to be
sized against the tolerance it is meant to defeat, and the only way to know is to measure it.

**CPU determinism.** The whole CPU path is exactly reproducible: two `reconstruct_mesh` calls
in one process with the same seed give bitwise-identical latents and metrics, and the 8-epoch
training run reproduces bitwise across processes. That is what allows `rtol=atol=0` on the
learning-rate baseline.

**GPU divergence, measured** — the handoff asks that this be stated plainly, so: **the CPU
baselines in this harness do not bound GPU divergence, and are not close to doing so.** Same
weights, same seed, same input, CUDA instead of CPU:

| Quantity | CPU vs GPU difference | Harness CPU tolerance |
|---|---|---|
| Fitted latent (max abs) | 4.0e-2 | 1e-4 |
| Bone surface centroid (max abs) | 3.1e-3 | 1e-4 |
| `assd_0` / `assd_1` | ~2% | 1e-4 relative |
| Reconstructed vertex count | 3344 → 3356, 1072 → 1078 | 2% |

A GPU run is a different numerical experiment. GPU results would need their own baselines on
pinned hardware — and those baselines could not be exact, because the same reconstruction run
twice on the same GPU is **not** bitwise identical; it agrees to about one float32 ulp
(~1e-8). The CPU path is exact. `test_gpu.py` pins all of this.

---

## 5. Out of scope, seen and left alone

Per handoff §4, noted and not touched. Each is its own PR.

| Thing | Where |
|---|---|
| `testpaths = ["tests"]` names a directory that does not exist; collection works by pytest 8's fallback | `pyproject.toml:95` |
| `addopts = "-k 'not train_test.py'"` filters a file that no longer exists, and `-k` matches test names not filenames, so it never worked | `pyproject.toml` |
| `black --check` fails on 9 files | `NSM/reconstruct/main.py`, `NSM/mesh/main.py`, `NSM/losses.py`, `NSM/models/triplanar.py`, others |
| `make lint` reports 445 flake8 violations, including 4 F821 undefined names; the CI lint job is `continue-on-error` | repo-wide |
| `.github/workflows/docs.yml` invokes `make requirements dev` and `make docs`, neither of which is a target | `.github/workflows/docs.yml` |
| `testing/testing_h5_vs_np_loading/save_and_load_h5_vs_np.py:1` is a shell command in a `.py` file, which breaks any AST tooling over the repo | that file |

**Platform.** The numeric baselines are pinned to **Linux-x86_64**, per the maintainer's call
that development is Linux-x86_64 and remote. Each baseline file records its stack under
`generated_on`. The CI matrix also runs `macos-latest` and no macOS machine was available
here, so the harness gates rather than gambles: a different OS/architecture **skips** the
numeric baselines with a reason naming both platforms, while a different torch or numpy
**goes red** — a dependency bump that moves training output is exactly what this harness is
for. Structural and exact-arithmetic assertions run everywhere. Regenerating on a foreign
platform refuses rather than clobbering the pinned file, and `TestBaselinePlatformPin`
exercises the gate so it cannot decay into a blanket skip. Adding macOS later means adding a
per-platform baseline file, not loosening tolerances.

For the record, the new files are `black --check` clean and produce **zero** flake8
violations, so none of the above got worse.

No CI change was needed or made: `.github/workflows/build-test.yml` already runs `make test`,
which is `pytest testing/ -v`, which collects `testing/NSM/regression/`.
