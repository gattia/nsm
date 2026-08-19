# Numerical regression harness

Phase 3 §7.1 of `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md`.

Its job is to **fail when NSM's training or reconstruction output changes**, so the Phase 4
decomposition can proceed without silently altering results. The findings that came out of
building it are entries in `docs/KNOWN_ISSUES.md`, each naming the test that pins it.

```bash
pytest testing/NSM/regression/ -q      # ~1 min; some tests skip without a GPU
pytest testing/NSM/regression/ -q -rx  # ... and list the strict xfails and their issues
make test                              # runs it along with everything else
```

CI needs no change: `.github/workflows/build-test.yml` already runs `make test`, which is
`pytest testing/ -v`.

## What it was built to do

Four constraints shape this harness, none of them visible from the assertions themselves.

- **It runs in CI on every PR and must stay under two minutes.** Past that people skip it
  and the whole exercise is wasted. That budget is why the fixtures are three tiny analytic
  meshes and eight CPU epochs rather than anything realistic, and it is the tightest
  constraint here — the suite has roughly doubled since the budget was set.
- **Reconstruction goes through `reconstruct_mesh`, not `reconstruct_latent`.**
  `reconstruct_mesh` is what the downstream consumer calls, and before this harness it had
  exactly one executed line in the entire suite: its `def`. It is called the way the
  consumer calls it — a *list* of mesh paths, every argument by name.
- **The order of the returned `mesh` list is a contract, so it is asserted.** Index 0 =
  bone, index 1 = cartilage is hardcoded by the consumer and declared nowhere: not in the
  signature, not in the docstring, not in the result dict. Nothing but this assertion would
  notice it inverting.
- **CPU baselines do not bound GPU divergence.** They are not a weaker form of a GPU check;
  they say nothing about one. `test_gpu.py` is the only thing here that does, and it is
  skipped without CUDA.

## Green does not mean "the library is correct"

It means **nothing changed**. The strict-xfail assertions here describe behaviour NSM *should*
have and does not, and they are marked `xfail(strict=True)` rather than written to assert the
broken behaviour — because a test that passes *because* something is broken makes a green
suite say the opposite of the truth.

So the report separates three things:

| Outcome | Meaning |
|---|---|
| `passed` | Behaviour that is correct, or a number that has not moved |
| `xfailed` | A known defect, still present. Listed in `docs/KNOWN_ISSUES.md` (Open) |
| `failed` | Either a regression, **or a defect that got fixed** — see below |

`strict=True` is what makes this work in both directions. Fix a defect and its xfail starts
passing, which pytest reports as `XPASS(strict)` — a **failure**. The suite goes red and names
the issue, so whoever fixed it has to come back, delete the mark, and close the issue.
Without `strict`, a fixed defect would silently keep reporting `xfailed` forever and the mark
would rot.

The known cost, stated plainly: an xfail that starts failing for a *different* reason still
reports `xfailed` and hides it. Mitigation is to keep each xfail body minimal and pointed at
one thing, which they are — several assert the premise still holds before asserting the defect,
so the mark cannot pass vacuously.

Not every known defect has an xfail. Some (`clamp_dist`, `KNOWN_ISSUES.md`) are a documentation or
design decision rather than a statable correctness assertion, and are carried as measurements.

## What is asserted

| Module | Pins |
|---|---|
| `test_training_regression.py` | 8 epochs, CPU, fixed seed: per-param-group learning rate at **every** epoch, loss trajectory and its components, latent-norm trajectory, checkpoint contents |
| `test_reconstruction_regression.py` | A full `reconstruct_mesh` call on the **committed decoder** (below): the eight result keys the consumer reads, the **order** of the `mesh` list, fitted latent, mesh geometry, ASSD, registration params. Plus one un-baselined smoke test that a *freshly trained* decoder can be reconstructed from at all |
| `test_dataset_cache.py` | Cache round-trip, which parameters reach the cache key and which do not, and what `random_seed` does and deliberately does not seed (11 xfail) |
| `test_model_roundtrip.py` | `save_model` → `load_model` is bitwise identical; `padding` is not in the checkpoint; the state dict aliases every VAE layer (5 xfail) |
| `test_gpu.py` | Skipped without CUDA. The seed-ordering constraint the consumer depends on, and **how far a GPU run diverges from these CPU baselines** |

Two of these are deliberate breaks that must go **red** on a broken build and are asserted
to do so on every run:

- `test_training_regression.TestDeliberateBreak` transposes the two learning-rate
  `Target` labels — the exact shape of the bug in `docs/KNOWN_ISSUES.md` §1 — and
  asserts the LR, loss and latent baselines all reject the result.
- `test_reconstruction_regression.TestDeliberateBreak` dents an input mesh — one of the bone
  sphere's 530 vertices, displaced by a quarter of its radius — and asserts the latent and
  geometry baselines reject the result.

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

**Tolerances** are sized from the deliberate breaks, not chosen by taste — a number that has
never been shown to catch anything is not a tolerance. They all live in `_harness.py`, in one
block, imported by every module that compares against them. They were duplicated once and the
copy was wrong: `test_gpu.py` carried its own `CPU_LATENT_ATOL` and `CPU_GEOMETRY_ATOL`, both
`1e-4`, against real values of `5e-4` and `3e-4`, so it asserted GPU divergence against a bound
five times tighter than the one it named.

The margin is **asserted, not written down**. `_harness.headroom()` reports how many times its
tolerance an observed deviation actually is; both `TestDeliberateBreak` classes require at least
`MIN_HEADROOM` (10) and print the measured multiple when they fail, so a fixture change that
weakens a break goes red on the run that weakens it. A hand-transcribed table used to stand
here, under the claim that every tolerance was "at least an order of magnitude" below the break
it catches, and both of its reconstruction rows were wrong — by 4× and 8×, in the direction
that made the breaks look weaker than they were. That error had already cost something: the
dent below was widened from one vertex to twenty to escape a margin that was never real. The
one margin genuinely under 10× is not in this suite at all — it is `test_gpu`'s surface-centroid
divergence, 2.4× `GEOMETRY_ATOL`, and it was invisible for as long as that module compared
against its own wrong copy of the number.

Two things have no headroom to measure and are not asserted against `MIN_HEADROOM`: learning
rates, compared exactly (`Initial * Factor ** (epoch // Interval)` in Python floats), and
`METRIC_RTOL` / `COUNT_RTOL`, which are not break detectors. Headroom is taken over the
**largest** element a break moves, because `np.allclose` rejects a value as soon as any one
element is out of tolerance.

The reconstruction break is **one vertex of the bone sphere's 530**, displaced by a quarter of
its radius — the smallest geometry change this fixture can express. It clears the floor with
34.8× to spare. Enlarging it is not the response to a failure here: 5 vertices measure 69× and
10 measure 119×, so raising that number can only make a failing break pass, which is loosening
a tolerance wearing a different hat. If this margin ever drops, the fixture or the tolerance is
what moved.

### Platform

**The numeric baselines are pinned to Linux-x86_64**, the platform development happens on.
Each file records what it came from under `generated_on`
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

## The reconstruction decoder is a committed asset

`assets/reconstruction_decoder.pt` (74 KB) holds the one decoder every reconstruction test
runs on. It is **loaded, not retrained**.

Until Aug 2026 the `reconstruction_model` fixture retrained it in-session, 60 epochs every
run — so `baselines/reconstruction.json` pinned a gradient-descent trajectory rather than
`reconstruct_mesh`. Gradient descent amplifies a last-bit arithmetic difference
exponentially, and a trajectory is therefore not portable across dependency versions.
Measured between torch 2.8.0+cu128 and 2.7.1+cu126, by decomposition:

| what varied | drift in the geometry baseline |
|---|---|
| everything (the old fixture) | **763× `GEOMETRY_ATOL`** |
| torch only, same decoder weights | 0.005× |
| decoder weights only, same torch | 763× |

The weights diverge 6.3e-07 by epoch 10, 1.7e-05 by 20, 1.4e-02 by 30, saturating near
3.9e-02: past epoch 30 the two stacks hold *different models*. Surviving a torch bump by
widening the tolerance would have meant a tolerance 12× larger than the deliberate break it
exists to detect — a detector that swallows its own signal. Training output is pinned
directly, and better, by `baselines/training.json`, at 8 epochs where it has not yet
diverged: it moved ~1e-8 across the same bump, 0.0002× its tolerance.

With the decoder frozen, the suite passes identically on both torch versions and the
residual drift is 0.005× `GEOMETRY_ATOL`.

**Provenance lives inside the checkpoint**, under `generated_on`, for the same reason
`baselines/*.json` carry theirs inside: a sidecar file can be separated from the weights it
describes, or left stale against them. `TestTheCommittedDecoder` asserts it is there.

The asset is read with `torch.load(..., weights_only=True)` into a model built by
`_harness.build_model`, then `load_state_dict(..., strict=True)`. Strict is deliberate: an
architecture change must fail loudly rather than half-load. A missing or unloadable asset is
an **error naming the regeneration command**, never a skip.

### Regenerating it

Needed when the architecture changes, and essentially never otherwise.

```bash
NSM_REGENERATE_RECON_DECODER=1 pytest testing/NSM/regression/  # retrain + rewrite the asset
NSM_REGENERATE_BASELINES=1 pytest testing/NSM/regression/      # then rebaseline against it
pytest testing/NSM/regression/                                 # then verify
```

The second step is not optional, and is why this is a **separate switch** from
`NSM_REGENERATE_BASELINES`: every reconstruction baseline is fitted to these exact weights.
Driving both from one variable would make that step invisible.
`test_the_reconstruction_decoder_is_not_being_regenerated` turns a run with the variable set
red, for the same reason its baseline equivalent does, and regenerating on a platform other
than the pinned one **refuses** rather than clobbering — mirroring `BaselineStore.flush`.

`TestAFreshlyTrainedDecoder` buys back the one thing freezing the decoder cost: nothing else
now checks that a model straight out of `train_deep_sdf` can be reconstructed from. It
trains and reconstructs and asserts only that a surface comes back and the latent has the
right shape — **no numeric baseline**, since those are precisely the chaotic numbers. It
costs ~2.5 s, and it goes through `_harness.train_reconstruction_decoder`, so the
regeneration path above is executed on every run instead of rotting between uses.

## How it stays deterministic

`SDFSamples(random_seed=...)` seeds every draw on both sampling paths, so the fixtures run on
the **near-surface** path production uses (`sigma_near=0.01`, `sigma_far=0.03` — the shipped
ShapeMedKnee widths, 0.743 mm and 2.35 mm against an ~80 mm femur, expressed in this
harness's max-radius-1 coordinates). `build_dataset` passes its `seed` there, and seeds
`torch` and `numpy` globally as well; the `numpy` seed still matters because `random_seed=None`
deliberately leaves sampling on the legacy global stream. `TestSeeding` and
`TestSeedDerivation` pin both halves, plus the derivation properties that are silent when
they break: different seeds give different data, the near and far passes draw different base
points, mesh list order does not change a subject's data, moving the meshes does not either,
and `multiprocessing=True` produces the same cache as `multiprocessing=False`.

**A subject's seed is derived from the bytes of its meshes**, not from its cache hash — the
cache hash contains the mesh's absolute path, so keying on it meant meshes written to
`/tmp/pytest-of-<user>/pytest-<n>/` were reseeded on every run and no baseline could
reproduce. Measured at the time: two consecutive runs of the training module disagreed in
the second decimal of every loss. The fixtures write to ordinary pytest `tmp_path`
directories.

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

Both surfaces must end up with samples of **both** signs, so that failure mode is worth
re-checking whenever the sigmas or the geometry move. At the fixture's 2000 points per
surface, all three subjects come out near 3110 positive / 890 negative for the bone and
3215 / 785 for the cartilage.

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
