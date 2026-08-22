# Changelog

Notable changes to NSM, newest first.

**What belongs here:** anything that changes what a caller must write, what an existing
call does, or what an existing result would be. Internal refactors, test coverage and
documentation do not, unless they change one of those three.

**Breaking changes lead each release**, because that is the only section a reader upgrading
across versions has to act on. NSM is pre-1.0, so a breaking change bumps the **minor**.

**When this is updated:** as part of the release, in the same PR that bumps
`NSM.__version__` and immediately before the tag is cut. A release with no changelog entry
is a release nobody downstream can evaluate.

**Related, and deliberately separate:** `docs/KNOWN_ISSUES.md` § History answers *"is a run
I already have on disk affected?"* — that is about results. This file answers *"does my
code still work?"* — that is about API. A change can warrant an entry in both.

**Installing a specific version.** NSM is not on PyPI (the `nsm` name there is an unrelated
package), so pin the tag directly:

```
nsm @ git+https://github.com/gattia/nsm@v0.2.0
```

---

## Unreleased

### Breaking

- **Four dead symbols are deleted** (audit disposition, maintainer-approved
  2026-08-22): `symmetric_chammfer` (`NSM/utils.py` — a `pass` stub returning `None`),
  `sdf_gradients` (`NSM/mesh/interpolate.py` — its return was mostly fabricated zero
  padding), `find_object_bounds_random_sampling` (`NSM/mesh/main.py` — non-deterministic,
  superseded by `coarse_bounds_from_sign_change`), and the `NSM/configs/deep_sdf_config`
  scratch file. All four had zero callers in this repo and in `kneepipeline`; rulings and
  evidence in `docs/SCOPE.md` §2.8.

- **`default_config.json` is replaced wholesale** (#48, maintainer decision): it is now a
  sanitized snapshot of the ShapeMedKnee `647_nsm_femur_v0.0.1` training config — the
  values that actually produced a shipped model — instead of a hand-written DeepSDF-era
  dict that could not drive `train_deep_sdf` at all (five unconditionally-read keys were
  missing, starting with `prefetch_factor`). The `LearningRateSchedule` entries carry
  `Target`s reproducing what 647 historically trained under — **the larger LR drives the
  latents** (History §1); `mesh_names` and `padding` are added explicitly; run identity,
  machine paths and derived keys are stripped. Pinned by
  `test_default_config_trains.py`, which runs the real trainer from the shipped file.

- **EMD is removed** (#53, maintainer decision): the `calc_emd` parameter of
  `compute_recon_loss`, `reconstruct_mesh` and `get_mean_errors`, the `emd` config key,
  the vendored `NSM.dependencies.sinkhorn` module (the whole `NSM.dependencies`
  package), and the `pykeops` requirement. No result ever existed: the only caller
  passed numpy arrays, which pykeops rejects at the boundary, in every version since the
  function was written — `calc_emd=True` always raised. A call that passes `calc_emd`
  now fails with `TypeError`; a config carrying `emd` is silently ignored. Both shipped
  ShapeMedKnee configs set `emd: false` and are unaffected.

- **`get_pts_center_and_scale` no longer takes `center` or `scale`.** Both were shadowed by
  the values computed from them before they were read, so neither had any effect at any
  value. They are removed rather than made authoritative: every caller passes
  `scale=norm_pts`, which defaults to `False` everywhere and is unset in the shipped
  configs, so an argument that worked would stop scaling on a default run and change the
  coordinate frame of every dataset and checkpoint NSM has produced. **No numerical output
  changes** — the arguments were inert, and the committed regression baselines are
  unmoved. Delete the arguments from any call; centering and scaling were always
  unconditional and still are.

- **`subsample` is required and validated on both dataset constructors** (#43).
  `MultiSurfaceSDFSamples` documented `subsample=None` as its default, but `None` could
  never construct — it crashed in `get_samples_per_sign` on a cold cache and skipped
  joint normalization on a warm one — so construction now refuses anything but a
  positive int, by name. No working call changes.

### Fixed — affects results

- **`get_optimizer` now passes `weight_decay` to `Adam`** (#47). It always passed it to
  `AdamW` and `schedule_free_AdamW`; the `Adam` branch silently dropped it, so every
  `optimizer: "Adam"` run trained with zero weight decay whatever the config said. An
  `Adam` run that sets `weight_decay` now trains differently — the committed training
  baselines moved and were regenerated (loss trajectory ~0.03% at epoch 1 to ~3% by
  epoch 6 at `weight_decay: 1e-4` on the CPU harness). Both shipped ShapeMedKnee configs
  use `AdamW` and are unaffected. To reproduce the old behaviour exactly, set
  `weight_decay: 0`. See `docs/KNOWN_ISSUES.md` § History §4.

- **Multi-surface overlap removal now counts, and multi-decoder reconstruction indexes
  by a running surface offset** (#44). `remove_overlapping_points` removed "sign sum ==
  −2" points — correct only at exactly two surfaces (nothing removed at 3 or 5; only
  inside-3-of-4 at 4); it now removes points inside two or more surfaces.
  `reconstruct_latent` scored every decoder after the first against the first decoder's
  ground truth; each decoder now reads its own slice of the flat `sdf_gt`. Two-surface,
  single-decoder runs — the shipped configuration — are bit-identical before and after
  (regression baselines unmoved). See `docs/KNOWN_ISSUES.md` § History §5.

- **The uniform sampling cube is symmetric, and the single-mesh sampler no longer clips
  its draws** (#40). Both samplers rebound `mins` before `maxs` read it, so a nonzero
  `uniform_pts_buffer` grew the cube more above than below (at the shipped `0.2`:
  `[-1.200, +1.220]` instead of `±1.200` on a normalized object); the single-mesh
  sampler additionally clipped all random draws — near-surface Gaussians included — to
  `±(1 + buffer/2)` under `norm_pts=True`. Both now share `get_buffered_cube_mins_maxs`
  and neither clips. Cached datasets built with a nonzero buffer, or single-surface with
  `norm_pts=True`, resample differently — **and the cache key does not know** (#19), so
  delete old `.npz` files to pick up the fix. Multi-surface buffer-0 runs are
  bit-identical (regression baselines unmoved). Also from #40: `read_mesh_get_sampled_pts`
  returns `pts_surface` as an int64 array, matching the multi-mesh sampler, instead of a
  Python list. See `docs/KNOWN_ISSUES.md` § History §6.

### Fixed

- **A surface with no positive or no negative SDF samples raises a `ValueError` naming
  the surface** (#41) instead of `ZeroDivisionError` — e.g. one surface nested inside
  another loses every interior point to overlap removal. A surface nothing draws from (a
  missing/`None` surface, or one allotted no subsample share) yields empty index lists
  and is handled.

- **`MultiSurfaceSDFSamples` accepts `joint_scale_buffer`** (#43) and forwards it to
  joint normalization. It was refused with `TypeError`; the parent's default (0.1)
  happens to equal the production value, which is why nothing noticed. Not yet in the
  cache key — that is #19's business (it does not change cached bytes).

- **`cyclic_anneal_linear` no longer NaNs runs shorter than its cycle count.**
  `floor(n_epochs / n_cycles)` was 0 for `n_epochs < 5`, so `epoch % 0` returned NaN and
  the NaN regularization weight silently NaN'd the entire training loss — the run
  completed and exited 0. Degenerate runs now pin the weight at `min_`; any run with
  `n_epochs >= 5` is bit-identical. No History entry: the degenerate path never produced
  a usable result.

- **`add_plain_lr_to_config` no longer raises `KeyError: 'Initial'` on a Constant
  schedule** (#48). `get_learning_rate_schedules` accepts Constant entries (which carry
  `Value`); the logging helper now reads them too.

- **`get_pts_center_and_scale` no longer mutates its input.** It copies first. The three
  in-repo callers each carried a defensive `np.copy(...)`; those are removed, since the
  copy now happens inside. A caller written without one is no longer silently corrupted.

---

## v0.2.0

Sampling can now be reproduced, and a numerical regression harness exists to make the
decomposition work in `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md` safe to start.

### Breaking

- **`include_seed_in_hash` removed** from both dataset constructors. Nothing set it, and
  once the seed began affecting sampled data it became a way to poison the cache. Passing
  it now raises `TypeError`. There is no replacement; delete the argument.
- **`mskt>=0.1.21` is now required** (was unpinned). Older versions do not accept
  `**kwargs` on `Mesh.rand_pts_around_surface`, so they raise `TypeError` on the first
  sample rather than quietly sampling unseeded. That is the intended backstop, not an
  incidental floor — do not relax it.

### Changed — affects results

- **`SDFSamples(random_seed=...)` now actually seeds sampling.** It was previously stored
  and hashed into the cache key but never used, so no run was reproducible. Callers who
  passed a seed were receiving unseeded data despite the argument and now receive seeded
  data.
  - **`random_seed=None`, the default, is bit-for-bit unchanged**, verified by comparing
    both the sampled arrays and the cache keys. Existing `.npz` caches stay valid.
  - The seed is derived per `(subject, sampling pass, surface)` rather than used directly,
    and the subject component keys on **mesh content**. See `docs/KNOWN_ISSUES.md`
    § History 3 for how to tell whether a run you already have is affected.
  - Fixes a related defect: `Pool` workers previously inherited one copy of the global
    NumPy state, so subjects were correlated rather than independent.

### Fixed

- **`os.sched_setaffinity` is now guarded**, so building a dataset with
  `multiprocessing=True` works on macOS and Windows. The same file already guarded
  `sched_getaffinity`; the `set` variant was missed. This raised `AttributeError` on any
  non-Linux platform.
- **Four undefined names** that raised `NameError` instead of doing their job:
  `deep_sdf.py` raised on a name that does not exist rather than the intended error;
  `reconstruct_latent_S3.py` crashed while formatting a bad-shape message and again on
  `log_wandb=True` with no `wandb` import; the deprecated trainer returned an unbound name
  after a completed run.

### Added

- **A numerical regression harness** under `testing/NSM/regression/`, pinning training and
  reconstruction against committed baselines, with deliberate-break tests that assert it
  can still detect a change. The reconstruction decoder is committed as a fixed asset
  because gradient descent amplifies last-bit arithmetic differences: retraining moved the
  geometry baselines 763× their tolerance across a torch version bump, while holding the
  weights fixed moved them 0.005×.
- **A doc-reference test** asserting that symbols cited in `docs/` still exist.

### Internal

No API effect: documentation restructured into `docs/` and `.claude/plans/`; `flake8` taken
to zero and made to gate CI; `make lint` / `make autoformat` aligned with `gattia/pymskt`;
`make docs` renders an API reference with pdoc.

---

## v0.1.0

The state before the code-health refactor. Retroactively summarised — this file did not
exist at the time, so `git log` is the authority for anything not listed.

### Breaking

- **`LearningRateSchedule` entries must declare `"Target"`** (`"model"` or `"latent"`).
  A config omitting it on either entry raises, with a message printing the paste-ready
  annotation that reproduces that run's historical behaviour. Exactly two entries, one per
  target; **entry order is ignored**.
  - Adam/AdamW and `schedule_free_*` migrate to **opposite** annotations, because the two
    families were affected differently. The error picks the right one from
    `config["optimizer"]`; do not hand-write it.

### Fixed — affects results

- **Learning-rate schedules were applied swapped** (model ↔ latent) on every Adam/AdamW run
  from May 2023 to August 2026. `get_optimizer` built groups as `[latent, model...]` while
  `adjust_learning_rate` reassigned by position each epoch. There is now no positional
  indexing anywhere in the LR path. See `docs/KNOWN_ISSUES.md` § History 1, including how
  to reproduce an affected run under fixed code.
