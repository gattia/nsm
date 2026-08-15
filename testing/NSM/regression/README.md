# Numerical regression harness

Phase 3 §7.1 of `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md`, built to the spec in
`planning/TEST_HARNESS_HANDOFF.md` §3.

Its job is to **fail when NSM's training or reconstruction output changes**, so the Phase 4
decomposition can proceed without silently altering results. Findings that came out of
building it are in `planning/TEST_HARNESS_NOTES.md`.

```bash
pytest testing/NSM/regression/ -q      # 97 passed, 20 xfailed; ~20 s (~13 s with no GPU)
pytest testing/NSM/regression/ -q -rx  # ... and list what the 20 xfails are
make test                              # runs it along with everything else
```

CI needs no change: `.github/workflows/build-test.yml` already runs `make test`, which is
`pytest testing/ -v`.

## Green does not mean "the library is correct"

It means **nothing changed**. Twenty of the assertions here describe behaviour NSM *should*
have and does not, and they are marked `xfail(strict=True)` rather than written to assert the
broken behaviour — because a test that passes *because* something is broken makes a green
suite say the opposite of the truth.

So the report separates three things:

| Outcome | Meaning |
|---|---|
| `passed` | Behaviour that is correct, or a number that has not moved |
| `xfailed` | A known defect, still present. Listed in `planning/DEFECT_WORKLIST.md` |
| `failed` | Either a regression, **or a defect that got fixed** — see below |

`strict=True` is what makes this work in both directions. Fix a defect and its xfail starts
passing, which pytest reports as `XPASS(strict)` — a **failure**. The suite goes red and names
the worklist item, so whoever fixed it has to come back, delete the mark, and tick the list.
Without `strict`, a fixed defect would silently keep reporting `xfailed` forever and the mark
would rot.

The known cost, stated plainly: an xfail that starts failing for a *different* reason still
reports `xfailed` and hides it. Mitigation is to keep each xfail body minimal and pointed at
one thing, which they are — several assert the premise still holds before asserting the defect,
so the mark cannot pass vacuously.

Not every worklist item has an xfail. Some (#10 `clamp_dist` semantics) are a documentation or
design decision rather than a statable correctness assertion, and are carried as measurements.

## What is asserted

| Module | Pins |
|---|---|
| `test_training_regression.py` | 8 epochs, CPU, fixed seed: per-param-group learning rate at **every** epoch, loss trajectory and its components, latent-norm trajectory, checkpoint contents |
| `test_reconstruction_regression.py` | A full `reconstruct_mesh` call: the eight result keys the consumer reads, the **order** of the `mesh` list, fitted latent, mesh geometry, ASSD, registration params |
| `test_dataset_cache.py` | Cache round-trip, which parameters reach the cache key and which do not, and what NSM's `random_seed` actually does (13 xfail) |
| `test_model_roundtrip.py` | `save_model` → `load_model` is bitwise identical; `padding` is not in the checkpoint; the state dict aliases every VAE layer (5 xfail) |
| `test_gpu.py` | Skipped without CUDA. The seed-ordering constraint the consumer depends on, and **how far a GPU run diverges from these CPU baselines** |

Two of these are deliberate breaks that must go **red** on a broken build and are asserted
to do so on every run:

- `test_training_regression.TestDeliberateBreak` transposes the two learning-rate
  `Target` labels — the exact shape of the bug in `docs/KNOWN_ISSUES_HISTORY.md` §1 — and
  asserts the LR, loss and latent baselines all reject the result.
- `test_reconstruction_regression.TestDeliberateBreak` moves one vertex of an input mesh
  by 0.25 (a quarter of the bone radius) and asserts the latent and geometry baselines
  reject the result.

## Baselines

Versioned artifacts in `baselines/*.json`, one file per test module, each
`{"schema_version": N, "generated_on": {...}, "values": {...}}`. A missing key fails; it is
never a silent pass.

```bash
NSM_REGENERATE_BASELINES=1 pytest testing/NSM/regression/     # rewrite
pytest testing/NSM/regression/                                # then verify
```

A regeneration run is not a passing run: `test_baselines_are_not_being_regenerated` fails
whenever the environment variable is set, so a CI job that somehow inherits it goes red
rather than quietly rebaselining.

Bump `SCHEMA_VERSION` in `_harness.py` only when the *meaning* of a stored key changes,
not when a number moves.

**Tolerances** are sized from the deliberate breaks, not chosen by taste. Learning rates are
compared exactly — they are `Initial * Factor ** (epoch // Interval)` in Python floats.
Everything else leaves at least an order of magnitude between the tolerance and the smallest
signal it has to catch:

| Baseline | Tolerance | Smallest break signal | Headroom |
|---|---|---|---|
| loss trajectory | `rtol=1e-3` | 1.7 relative (LR swap) | 1700× |
| training latent norms | `atol=1e-4` | 5.4e-2 (LR swap) | 540× |
| fitted latent | `atol=5e-4` | 7.0e-3 (moved vertex) | 14× |
| mesh geometry / deciles | `atol=3e-4` | 9.0e-4 (moved vertex) | 3× |
| surface metrics | `rtol=2e-3` | — | not a break detector |

### Platform

**The numeric baselines are pinned to Linux-x86_64**, the platform development happens on.
Each file records the stack it came from under `generated_on`
(Linux-x86_64 / CPU / Python 3.9.25 / torch 2.8.0+cu128 / numpy 2.0.2).

The gate is deliberately asymmetric (`_harness.platform_matches`):

- **A different OS or architecture skips the numeric baselines**, with a reason naming both
  platforms. The CI matrix also runs `macos-latest`; there is no macOS baseline, and
  inventing one by loosening tolerances until both platforms fit would leave a harness that
  detects nothing. Structural and exact-arithmetic assertions — per-epoch learning rates,
  result keys, mesh ordering, cache keys, checkpoint round-trip — still run everywhere,
  because they are identity or exact Python float arithmetic.
- **A different torch or numpy goes red**, never skips. A dependency bump that moves
  training output is precisely what this harness exists to report; the failure message names
  the version difference.

Regenerating on a platform other than the pinned one **refuses** rather than clobbering the
committed baseline. To support a second platform, add a per-platform baseline file — do not
overwrite this one. `TestBaselinePlatformPin` exercises the gate itself, so it cannot decay
into a blanket skip unnoticed.

## How it stays deterministic

**NSM seeds nothing.** `SDFSamples(random_seed=...)` is documented as "Random seed" and is
only ever appended to the cache key (`test_dataset_cache.TestSeeding`). The harness
therefore seeds `numpy` and `torch` itself at each entry point.

That is enough on the **uniform** sampling path (`sigma_near`/`sigma_far` of `None`), which
draws through `np.random.uniform`. It is not enough on the near-surface path, which cannot be
seeded by a caller at all: `pymskt.Mesh.rand_pts_around_surface` has two independent draws
that bypass `np.random.seed()` — `pcu.sample_mesh_random(..., random_seed=0)`, where
`random_seed=0` means "seed from the current time", and `np.random.default_rng()` with no
argument, which seeds from OS entropy. Reported upstream as
[gattia/pymskt#54](https://github.com/gattia/pymskt/issues/54).

Every fixture here uses the uniform path for that reason, and `test_dataset_cache.TestSeeding`
pins both halves so the restriction can be lifted the day that issue lands.

## Shape of the fixtures

Three synthetic "subjects", each a bone (sphere) plus a cartilage (small oblate ellipsoid,
offset in +z), written by `pyvista` with no sampling and no meshfix. The offset is not
cosmetic: it makes the two surfaces identifiable by centroid, which is what lets the
reconstruction module assert the result `mesh` list **order** rather than merely its
length.

The surfaces are disjoint solids rather than nested shells because
`MultiSurfaceSDFSamples.remove_overlapping_points` drops every point interior to two
objects — nesting them leaves the inner surface with no negative samples and
`sdf_pos_neg_idx` then divides by zero.

## Things worth knowing before editing this

- `_harness.build_model` imports `loader._get_triplanar_params`, a private function,
  because NSM has no public "build the model this config describes" call. That keeps the
  model the harness trains identical to the model `load_model` builds. When the decoder
  registry of plan §8.1 lands, this import is what should fail loudly.
- `run_training` wraps `train_epoch` because `train_deep_sdf` returns `None` — no loss
  history is observable from the public entry point. If that ever changes, the wrapper can go.
- `loc_save` is always passed explicitly. The constructor default is
  `os.environ.get("LOC_SDF_CACHE", ...)` evaluated as a *default argument*, so it is bound
  when `sdf_dataset` is imported and setting the variable inside a test comes too late —
  the test would write into the developer's real `~/.cache/nsm_sdf_cache`.
- `reconstruct_mesh` calls `decoders[i].to(device)`, which moves a module **in place**.
  Anything handing it a session-scoped fixture must hand it a copy; `test_gpu.on_cuda` does.
- This directory has no `__init__.py`, matching `testing/NSM/models/`, `reconstruct/` and
  `datasets/`. That is what puts it on `sys.path` and makes `from _harness import ...` work.
