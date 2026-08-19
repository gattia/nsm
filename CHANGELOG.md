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

- **`get_pts_center_and_scale` no longer takes `center` or `scale`.** Both were shadowed by
  the values computed from them before they were read, so neither had any effect at any
  value. They are removed rather than made authoritative: every caller passes
  `scale=norm_pts`, which defaults to `False` everywhere and is unset in the shipped
  configs, so an argument that worked would stop scaling on a default run and change the
  coordinate frame of every dataset and checkpoint NSM has produced. **No numerical output
  changes** — the arguments were inert, and the committed regression baselines are
  unmoved. Delete the arguments from any call; centering and scaling were always
  unconditional and still are.

### Fixed

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
