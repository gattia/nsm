# Numerical regression harness

Its job is to **fail when NSM's training or reconstruction output changes**. The defects
found while building it are entries in `docs/KNOWN_ISSUES.md`, each naming the test that
pins it.

```bash
pytest testing/NSM/regression/ -q      # about a minute; the GPU test skips without CUDA
pytest testing/NSM/regression/ -q -rx  # also list the strict xfails and their issues

# Release-time only: the real shipped checkpoints, 275 MB and 260 MB, never in CI.
NSM_SHIPPED_MODELS=/path/to/NSM_MODELS pytest testing/NSM/regression/test_shipped_checkpoints.py
```

## Design constraints

- **It runs in CI on every PR, so it must stay fast.** That is why the fixtures are three
  tiny analytic subjects and eight CPU epochs.
- **Reconstruction goes through `reconstruct_mesh`**, the function the downstream consumer
  calls, the way it calls it: a list of mesh paths, every argument by name.
- **The order of the returned `mesh` list is a contract.** Index 0 is bone and 1 is
  cartilage. The consumer hardcodes this and NSM declares it nowhere, so a test asserts it
  by geometry.
- **CPU baselines do not bound GPU divergence.** `test_gpu.py` records by how much, and is
  skipped without CUDA.

## Green means "nothing changed"

It does not mean the library is correct. A known defect is a strict xfail: an assertion of
the behaviour NSM should have. Fix the defect and the xfail passes, which `strict=True`
reports as a failure, so whoever fixed it removes the mark and closes the issue.

| Outcome | Meaning |
|---|---|
| `passed` | Correct behaviour, or a number that has not moved |
| `xfailed` | A known defect, still present, listed in `docs/KNOWN_ISSUES.md` (Open) |
| `failed` | A regression, or a defect that was fixed |

An xfail that starts failing for a different reason still reports `xfailed`. Each xfail body
is kept small and pointed at one thing for that reason.

## What is asserted

| Module | Pins |
|---|---|
| `test_training_regression.py` | 8 CPU epochs, fixed seed: each param group's learning rate at every epoch, the loss and its components, the latent norms, checkpoint contents, resume |
| `test_reconstruction_regression.py` | A full `reconstruct_mesh` on the committed decoder: the keys the consumer reads, the `mesh` order, the fitted latent, geometry, ASSD and registration. Plus a freshly trained decoder, against an untrained control |
| `test_dataset_cache.py` | Cache round trip and repair, which parameters reach the cache key, and what `random_seed` does and does not seed |
| `test_model_roundtrip.py` | `save_model` then `load_model` is bitwise identical; `padding` must be stated; pre-#27 aliased checkpoints still load |
| `test_gpu.py` | Skipped without CUDA. A GPU reconstruction has the same structure and broadly the same metrics |

Two deliberate breaks must go **red**, and are asserted to on every run:

- `test_training_regression.TestDeliberateBreak` transposes the two learning-rate `Target`
  labels, the bug in `docs/KNOWN_ISSUES.md` §1. The LR, loss and latent baselines must all
  reject the result.
- `test_reconstruction_regression.TestDeliberateBreak` moves one of the bone sphere's 530
  vertices by a quarter of its radius. The latent and geometry baselines must reject it.

## Baselines

Versioned files in `baselines/*.json`, one per test module, each
`{"schema_version": N, "generated_on": {...}, "values": {...}}`. A missing key fails.

```bash
NSM_REGENERATE_BASELINES=1 pytest testing/NSM/regression/     # rewrite
pytest testing/NSM/regression/                                # then verify
```

`test_baselines_are_not_being_regenerated` fails whenever the variable is set, so a
regeneration run cannot pass for a checking one. Bump `SCHEMA_VERSION` in `_harness.py`
only when the meaning of a stored key changes, not when a number moves.

**Tolerances** are sized from the deliberate breaks and live in one block in
`_harness.py`. A copy in `test_gpu.py` once carried `1e-4` for both reconstruction
tolerances against real values of `5e-4` and `3e-4`.

**The margin is asserted.** `_harness.headroom()` reports how many times its tolerance a
deviation is, and both `TestDeliberateBreak` classes require at least `MIN_HEADROOM` (10).
Learning rates have no margin to measure: they are compared exactly. `METRIC_RTOL` and
`COUNT_RTOL` are not break detectors.

The reconstruction break is **one vertex**, the smallest change this fixture can express,
and it clears the floor with 34.8x to spare. Do not enlarge it to fix a failure: 5 vertices
measure 69x and 10 measure 119x, so a bigger break can only make a failing one pass.

### Platform

The numeric baselines are pinned to **Linux-x86_64** (Python 3.9.25, torch 2.8.0+cu128,
numpy 2.0.2), recorded under `generated_on`. The gate (`_harness.platform_matches`) is
asymmetric:

- **A different OS or architecture skips the numeric baselines.** CI also runs
  `macos-latest`, which has no baseline. Exact assertions (learning rates, result keys,
  mesh order, cache keys, checkpoint round trip) still run everywhere.
- **A different torch or numpy goes red.** A dependency bump that moves the output is what
  this harness exists to report.

Regenerating on another platform refuses rather than overwriting. To support a second
platform, add a per-platform baseline file. `TestBaselinePlatformPin` exercises the gate.

## The reconstruction decoder is a committed asset

`assets/reconstruction_decoder.pt` (74 KB) is the one decoder every reconstruction test runs
on. It is loaded, not retrained. Retraining it each run pinned a 60-epoch gradient-descent
trajectory, which amplifies a last-bit arithmetic difference. Measured between torch
2.8.0+cu128 and 2.7.1+cu126:

| what varied | drift in the geometry baseline |
|---|---|
| everything (retrained each run) | **763x `GEOMETRY_ATOL`** |
| torch only, same decoder weights | 0.005x |
| decoder weights only, same torch | 763x |

The weights diverge 6.3e-07 by epoch 10, 1.7e-05 by 20 and 1.4e-02 by 30, so past epoch 30
the two stacks hold different models. Training output is pinned by
`baselines/training.json` at 8 epochs, where it moved ~1e-8 across the same bump.

The asset carries its provenance under `generated_on`, inside the checkpoint so the two
cannot separate. `TestTheCommittedDecoder` checks it. It is loaded with `strict=True`, and a
missing or unloadable asset is an error naming the regeneration command, never a skip.

### Regenerating it

Needed when the architecture changes.

```bash
NSM_REGENERATE_RECON_DECODER=1 pytest testing/NSM/regression/  # retrain and rewrite the asset
NSM_REGENERATE_BASELINES=1 pytest testing/NSM/regression/      # then rebaseline against it
pytest testing/NSM/regression/                                 # then verify
```

The second step is required: every reconstruction baseline is fitted to these weights. That
is why the decoder has its own switch, and why
`test_the_reconstruction_decoder_is_not_being_regenerated` fails while it is set.

`TestAFreshlyTrainedDecoder` checks what freezing the decoder gave up: that a model straight
out of `train_deep_sdf` can be reconstructed from. It pins no number. The trained decoder's
ASSD must beat an untrained control's by 3x (measured 9.8x and 17.5x). It trains through
`_harness.train_reconstruction_decoder`, so the regeneration path runs every time.

## How it stays deterministic

`SDFSamples(random_seed=...)` seeds every draw on both sampling paths, so the fixtures use
the near-surface path production uses. `build_dataset` also seeds numpy's global stream,
which a `random_seed=None` call still draws from. `TestSeeding` and `TestSeedDerivation`
pin both, plus the derivation properties that fail silently: different seeds give
different data, the near and far passes draw different points, list order and mesh
location do not change a subject's data, and `multiprocessing=True` builds the same cache.

A subject's seed comes from the bytes of its meshes, not from its cache hash. The hash
contains the absolute path, and pytest's temporary paths change every run: keyed on the
hash, two consecutive runs disagreed in the second decimal of every loss.

## The fixtures

Three subjects, each a bone sphere and a small oblate cartilage ellipsoid offset in +z,
written by `pyvista` with no sampling and no meshfix. The offset makes the surfaces
identifiable by centroid, which is how the `mesh` order is asserted. The surfaces are
disjoint because `remove_overlapping_points` drops every point inside two surfaces: nested,
the inner one would have no negative samples, which `sdf_pos_neg_idx` refuses.

## Before editing

- `_harness.build_model` imports the private `loader._get_triplanar_params`, because NSM
  has no public "build the model this config describes" call.
- `loc_save` is always passed explicitly, so no test writes into the developer's cache.
- `reconstruct_mesh` moves a decoder to its device in place. Hand it a copy of a
  session-scoped fixture, as `test_gpu.py` does.
- This directory has no `__init__.py`. That puts it on `sys.path` and makes
  `from _harness import ...` work.
