# Plan: NSM code-health audit and refactor

**Repo:** `gattia/nsm` (NSM). **Created:** 2026-08-14. **Restructured:** 2026-09-21.

> **This file is the whole plan. Do not read the history file to find out what to do.**
> Everything executed — the old 1,450-line State narrative and the eighteen slice
> statements §8.0.A–R — moved verbatim to `NSM_CODE_HEALTH_REFACTOR_HISTORY.md` on
> 2026-09-21, because at 5,152 lines this file could no longer answer "what is next".
> Go there only to look up what one past slice did.

**Where the old section numbers went.** Other files cite this plan by section, so the
numbering is preserved rather than renumbered. Sections that moved are in the history file
under the same number.

| Cited as | Now in |
|---|---|
| §1 Why · §3 Phase 0 · §4 Phase A · §5 Phase 1 · §6 Phase 2 · §7 Phase 3 (incl. §7.1 harness, §7.5 validation runs) | history file, same numbers |
| §8.0 slice index · §8.0.A through §8.0.R | history file, same numbers |
| §8.0.P · §8.0.S · §8.0.T | **this file**, as Steps P, S and T below — restated and re-measured |
| §8.1 · §8.2 · §8.3 · §9 · §10 · §10.1 · §11 | **this file**, unchanged |

## State

**Updated:** 2026-09-22 · **Status:** open

- **Next:** **Slice T**, the test-suite trim. Nothing gates it and it is the last slice
  before Step Close. Re-measure the table in Step T before starting — slice S moved both
  numbers, and the ratio is the point.
- **Blocked on:** nothing in this repo. **One thing is owed outside it before the release
  is deployed**, and it is not a blocker on the merge: `kneepipeline/steps/run_nsm.py:211`
  passes `verbose=True` to `reconstruct_mesh` and that parameter no longer exists, so
  pulling this tree into the production checkout without deleting that line fails the next
  NSM fit. The consumer change is inert against `main` today and can land at any time.
  Merging S does not deploy S — the pull is the deploy.
- **Done:** Phases 0–3 complete. Eighteen slices executed, A through R, each with its own
  PR. v0.2.0 (PR #36) and v0.3.0 (§8.0.O) shipped, and **v0.4.0 is cut and waiting for its
  tag**. Both post-v0.3.0 validation runs passed — §7.5a on the production box (PR #102),
  §7.5b on the maintainer's cluster (PR #105). **§8.0.R merged 2026-09-21** as `d9ae062`
  (PR #107). **Step 0 closed 2026-09-22:** its last item validated R against production and
  passed — R moves the production BScore less than re-running the same code does.
  **Slice P closed 2026-09-22**, as PRs **#110** (`2c0a9cf`) and **#111** (`a7b3351`):
  `train/deprecated/` and `two_stage` deleted, #18 closed won't-fix, the multi-head trainer
  removed (#51 closed), the `grad_clip` entry closed as working by design. `NSM/` 15,106 →
  13,555 lines. **Slice S executed 2026-09-22** on `slice-s-v0-4-0-signatures`, eight
  commits: `pts_surface` required, `max_batch_size` deleted, the two decoder constructors
  refusing unknown keywords, `regions_label` deleted, the whole `verbose=` bridge deleted,
  `NSM.mesh` exporting its four submodules, and the CHANGELOG cut to v0.4.0. `NSM/` 13,555
  → **13,326**; suite 1180 → **1177** passed, 4 skipped, 3 xfailed, 110 s. Tree pulled and
  the celery worker restarted 2026-09-22 (PID 986906, stamp `v0.3.0-84-ga7b3351`).
- **Surprises:**
  - **A slice can close a finding in the same breath as deferring it.** All five
    `reconstruct_latent` sites §8.0.K deferred to R were already fixed by §8.0.K's own
    review round and §8.0.J's kwargs refusal. The row recorded the deferral and lost the
    fix, so R was scheduled to re-fix a "200× step size" that had already moved. **Build a
    slice row by sweeping what names it, and re-run it before scheduling it.** *Slice S hit
    the same thing twice more: its item (3) was already closed by R, and the 2026-09-21
    re-measure rebuilt the item table from the §8.0.S row alone and so lost the two items
    §8.0.N′ had deferred to S by name.*
  - **A stale test name in `docs/` survives every check the repo has.**
    `docs/KNOWN_ISSUES.md` named `TestTheLbfgsParametersAreReadOnBothPaths` as what pins
    the hybrid/LBFGS entry, and no such class has ever existed — the behaviour is pinned,
    under another name, in another file. `test_docs_references` checks that every
    `module.symbol` in `docs/` resolves in `NSM/`, which is the library and not the suite.
    Found by running the claim rather than reading it.
  - **The "accepted and never read" class is empty on the function surface.** Every
    candidate was a polymorphic hook whose *sibling* implementation reads the parameter, so
    #20's delete-the-parameter remedy needs judging across every implementation. Every live
    instance is one frame up, at the config layer, in `models/loader.py`. That finding is
    what started `NSM_CONFIG_SECTIONS_AND_MODEL_REGISTRY.md`.
  - **Not every unread argument is a parameter, and the remedy does not transfer.**
    `compare_cart_thickness`'s `orig_cart` was scheduled for deletion alongside
    `regions_label` and turned out to be element 1 of a list whose length is a fixed-layout
    contract shared with the reconstruction's list. Deleting it would have made the two
    lists different shapes. "Delete the argument" is a rule about signatures; a positional
    slot in a list argument needs the contract read first.
  - **A one-release deprecation does not reach a consumer that imports you as a
    directory.** The `verbose=` bridge's whole design was a release of overlap so nobody
    lost output without notice, and its own header records why the notice could not land —
    a `DeprecationWarning` is invisible under Python's default filter outside `__main__`,
    and the consumer calls NSM from inside a module. The deletion is loud, which is the
    good case, but the consumer still finds out by failing. For an unpinned consumer the
    announcement has to be a message to its maintainer, not a warning in the code.
  - **A validation A/B does not need the production tree checked out to the old ref.**
    §7.5a checked out each ref in the working tree the production worker imports, and made
    "restore `main`, verified" the last step of every session. Step 0's R validation ran
    both versions from one tree instead: the old checkout sits in scratch and wins
    `import NSM` through `PYTHONPATH` plus a `sitecustomize.py` that drops the editable
    install's meta-path finder, which otherwise beats `PYTHONPATH`. The production tree is
    never modified, so there is nothing to restore and no window where an arriving job
    would run the wrong code.
  - **A deletion slice is mostly a documentation slice, and the sweep needs its own
    sweep.** P deleted 1,108 lines of `NSM/` and touched 23 files to do it: every carve-out
    that existed only for the deleted code, every ruling that assumed it (`SCOPE.md` §1,
    §2.2, §2.6, §2.7, §2.9 and §5, plus the `ARCHITECTURE.md` coverage and duplicate-name
    tables), and two `KNOWN_ISSUES` entries about code that no longer exists. A History
    entry outlives the code it describes on purpose — it answers a question about runs —
    but a *dotted citation* in one does not: `test_docs_references` requires every
    `module.symbol` in `docs/` to resolve in the current tree, which is a good rule that a
    deletion slice will trip. **The first pass still missed six**, all prose rather than
    code: see Step P.
  - **Porting a feature means reading it, not moving it.** #18 had a maintainer-endorsed
    "port this" on the issue since Aug 2026, and the first version of P carried it out
    faithfully — including the defect that had made its author switch it off in Feb 2024,
    which nothing in the issue, `SCOPE.md` or the plan recorded because nobody had run it.
    `CLAUDE.md`'s "never inherit a rationale along with the code" is the rule that would
    have caught it, and what it needs in practice is a *behavioural* check: the four tests
    the port shipped all passed with and without the one-word fix, because they asserted
    the forward value and the defect was in the gradient.
  - **The test suite grew faster than the library it tests.** Slice T was scheduled at a
    1.02 test-to-source ratio and a 109-second suite; measured 2026-09-21 it is **1.10 and
    180 seconds**. Every slice since has added to what T was created to trim.

---

## What is left

Five steps, in order. Nothing here is blocked. Steps 0 and P are small, S is a release,
T is the one with judgment in it.

### Step 0 — clear the decks (no code)

- [x] **Check out `main` and pull.** Done 2026-09-21. The working tree sat on the unmerged
      `slice-r-parameter-surface` from 2026-09-03, and `kneepipeline` imports this directory
      rather than an installed package, so twenty jobs between 09-15 and 09-21 ran branch
      code. **Their numbers are fine** — that branch is now `main`, verified byte-identical
      in `NSM/` — but see the next item.
- [x] **Restart the celery worker.** Done 2026-09-22, `systemctl restart
      knee-pipeline-worker.service`, verified idle first (no step children, no active or
      reserved celery tasks, only the 102 MiB bare CUDA context). New PID 803823, pings
      OK, and the stamp it will now compute is `v0.3.0-64-g066223f` against the stale
      `v0.3.0-46-gcd83ccc` the twenty exposed jobs recorded. **Restarted again after
      pulling #109** (`v0.3.0-70-gfcf170f`), which is the standing rule this exposes:
      **every pull of this tree needs a worker restart before the next job runs, or that
      job's stamp lies.** Measured 2026-09-22: `environment_versions()` is imported and
      called *inside* the manifest writer, not at worker startup, so `lru_cache` freezes
      the stamp at the **first job after a restart** and holds it for that worker's life.
      A commit made after a restart but before any job is therefore harmless. That rule
      has no permanent home yet and wants one in the website repo. The website computes each job's `nsm`
      version stamp with `git describe` but caches it with `@lru_cache` for the life of the
      worker process (`backend/services/version.py`). The worker has been up since
      2026-09-01, so all twenty of those manifests record `nsm: v0.3.0-46-gcd83ccc` while
      the tree was at `ee396a2`. **The plan's own "merging is not deploying; the pull is
      the deploy" note is true for the code and false for the record.** Restarting is safe
      now: the worker is idle, holding only its 102 MiB bare CUDA context with no step
      children. The durable fix belongs in the website repo — drop the cache, or key it on
      the tree's HEAD — and is worth an issue there.
- [x] **Validate §8.0.R against production, once.** *(Executed 2026-09-22 on the production
      box; **PASS**. Archived job `8ff02ee4` — §7.5a's job 1, the one production NSM
      comparison point the archive holds — re-run through `steps.run_nsm`
      (`nsm_type: "both"`) + `steps.compute_bscore` in the production env, three times:
      current `main`, pre-R `main` (`b0b5d3f`, the R merge's first parent), then `main`
      again as a same-code control.*

      | | bone+cart | bone-only |
      |---|---|---|
      | **main vs pre-R — R's own contribution** | **1.15e-05** | **3.23e-05** |
      | main vs main, same code — this box's noise today | 3.64e-05 | 5.99e-05 |
      | main vs the archived production value | 3.78e-05 | archive has none |

      *R's A/B difference is **smaller than the same-code repeat**, so its numerical
      contribution is below the noise floor rather than merely inside it. Latent L2 across
      the three runs is 7.4e-04–7.8e-04 (bone+cart) and 1.4e-03 (bone-only) on 512-dim
      vectors of norm 7.3, with no pair standing out; ASSD agrees to 7e-05 mm. Nothing near
      the 0.08 seed-ordering signature, and bone+cart's documented ~0.004 `fix_mesh` band is
      100× away.*

      ***Two premise corrections.*** *"Both BScore variants against the archived values"
      cannot be done: every production job runs `nsm_type: bone_and_cart`, so every archived
      `bscore_results.json` in the 343-job archive holds one key. Bone-only is validated by
      the A/B instead, which is the stronger comparison anyway — it holds the meshes, the
      environment and the box fixed and varies only the NSM tree. And the row's worry about
      production having moved to mskt 0.1.21 since §7.5a does not reach this path: the
      prepared meshes this run fed the fit came out **byte-identical to the 2026-08-17
      archive**, because `generate_meshes` — the step PR #102 measured the upgrade moving —
      is not re-run.*

      ***The production tree was never checked out to an old ref.*** *Both NSM versions ran
      from one working tree: the pre-R clone sat in scratch and won `import NSM` through
      `PYTHONPATH` plus a scratch `sitecustomize.py` that drops the editable install's
      meta-path finder. Verified in the fit subprocess itself, which is where it matters —
      it resolved to the clone and `_takes_latent_and_xyz`, R's memoized dispatch, was
      absent there. §7.5a's "`checkout main` is the last step of every session" is a rule
      for a risk this arrangement does not take.*

      R shipped without a §7.5a-style check,
      and those twenty jobs were its only production exposure. Re-run one archived job on
      `main` and compare both BScore variants against the archived values; the measured
      run-to-run band on that box is ~1e-04. This is cheap and it is the last unvalidated
      slice.
- [x] **Committed the `CLAUDE.md` attribution rule** (`555fcf4`), uncommitted on this tree
      since 2026-09-17.
- [x] **PR #109 merged** 2026-09-22 (`fcf170f`). Records the first real-data test of
      `NSM_TRAINING_IDEAS.md` Idea 13: a drop-in activation in the triplanar conv stack
      does not help. It needed repair before it could merge — its one plan-file hunk
      amended §7.5b, which the 2026-09-21 split moved to the history file, so it
      conflicted. Resolved on its own branch by merge commit `7cf519d`, no force-push.
- [x] **PR #108 closed unmerged** 2026-09-22. Its first commit `9d2224a` deletes
      `NSM/models/modulated_periodic_activations.py` entire, which the maintainer ruled on
      2026-09-04 **stays** as the ShapeMed-Knee paper's MPA baseline; the PR predated the
      ruling by a day and still carried it. Its second commit `3d0775e` (delete
      `two_stage`, −414 lines) is approved and **is now owed to slice P** — see there. It
      also conflicted with #107 on `NSM/models/loader.py` and `docs/SCOPE.md`, so closing
      rather than rebasing saved that work too.
- [x] **Archive the handoff files.** Done 2026-09-22: the three files moved to
      `.claude/handoff/archive/`, superseded by this plan and by the posted #107
      description. The directory is gitignored, so none of it is in the repo. Handoff files
      are scratch with a lifetime of days — `CLAUDE.md` names them as something this repo
      does not keep, and the plan's State block is the real handoff.

### Step P — slice §8.0.P: quarantine and delete — **executed, PR #110 open**

*Four commits on `slice-p-quarantine-and-delete`, **284 lines added and 1,608 deleted**
(`NSM/` itself: 13,998 from 15,106). Suite 1189 passed / 4 skipped / 3 xfailed, `make lint`
clean, and the production consumer verified by importing its two symbols in the
kneepipeline environment, which imports this tree.*

**The slice was executed twice.** The first version ported the inverse-Lx weighting per
#18 and was opened as PR #110 on 2026-09-22. The maintainer read the ported branch the
same day and ruled it out; the branch was rebuilt without it (force-push, old tip kept as
the local `backup-before-drop`). What each bullet turned out to be:

- ***#18 closes won't-fix, and the four config keys go with the file.*** *The port was
  rejected on four measured counts, not on taste. **The paper's curriculum is already
  fully implemented**: Curriculum DeepSDF (arXiv:2003.08593) has two components, equation
  5 surface accuracy and equation 6 sample difficulty, and both are live in
  `_surface_l1_loss`. The paper contains **no inverse-power weighting at all**, so
  `sample_difficulty_lx` is NSM's own experiment rather than a missing piece of a published
  method. **It was tried and switched off** — live from `5188417` (Aug 2023) to `e173adc`
  (14 Feb 2024, "Remove some hard sample difficulty weight"). **It did not work**: its
  weight is built from the loss it multiplies and was never detached, so autograd
  differentiates `l1 / (l1 ** lx + eps)` whole, which has a maximum at
  `(eps / (lx - 1)) ** (1 / lx)` — 0.01 at `lx=2, eps=1e-4` — above which the gradient
  inverts (+4800 at error 0.005, then −1200 at 0.02, −355 at 0.05, −4.0 at 0.5), and with
  `surface_accuracy_e` and `lx < 1` it returns NaN on a run that exits 0. **And porting cost
  ~150 lines** of code, tests and docs for an untuned experiment with no users.*
- ***Equation 6 is immune for a reason worth keeping.*** *Its weight is built from
  `torch.sign`, which passes no gradient, so it is constant by construction. The two
  branches Feb 2024 removed — inverse-Lx and `hard_sample_difficulty_power` — are exactly
  the two that lacked that property. That is now a rule in `CLAUDE.md`: a weight built from
  the prediction or the loss has to be detached.*
- ***`KNOWN_ISSUES` § History 31 answers a question about runs, not about code.*** *The
  keys doing nothing since Feb 2024 is inert and needed no entry. The Aug 2023 – Feb 2024
  window, when the branch was live and its gradient inverted, is not: a reader holding a
  run from then has a model that is not what its config describes.*
- ***#108's §1 line was wrong in the other direction too.*** *Measured 2026-09-22: **all
  three** advertised types — `triplanar`, `deepsdf`, `implicit` — fit a latent through
  `reconstruct_latent` end to end, so `SCOPE.md` §1's "only `TriplanarDecoder` survives the
  reconstruction path" is what needed rewriting, not the count in it. What survives of that
  bullet is that `load_model`'s type list is a hardcoded `if/elif` — §8.1. **Nothing pins
  this**, and it replaced a documented limitation; S or §8.1 should give it a test.*
- ***§8.0.S item (4) loses its only config route.*** *The evidence test removed with
  `two_stage` was the one config path reaching `TriplanarDecoder`'s unread `**kwargs`.*
- ***#51 needed nothing***, *as predicted — but see the next step.*

**A deletion slice is mostly a documentation slice, and its sweep needs its own sweep.**
The first version deleted 1,112 lines of `NSM/` and touched 22 files to do it, and still
left six references to the deleted code behind: a `per-file-ignores` entry in `.flake8`,
two exemption docstrings that said "these three" over a two-entry tuple, two orphan nodes
in `ARCHITECTURE.md`'s module diagram, an import left unused by the test deletions (which
`make lint` cannot see, because `.flake8` project-ignores F401), and a `SCOPE.md` §2.7
line still arguing for quarantine-not-delete three lines below the decision to delete. A
second config-key sweep also had its own copy of the exemption the first one dropped.
**Grep for the deleted name in prose as well as code, and count the things a sentence
claims to count.**

**Ruled 2026-09-22, executed next as its own PR: remove `train_deep_sdf_multi_head.py`**
(443 lines). Not a model type — a training loop taking N ordinary decoders, defining no
model class, 47% of its non-comment lines verbatim copies of `train_deep_sdf.py`. Broken
since 2023 (only the last decoder trains), last feature work 2025-01-27, and every 2026
commit to it is a refactor sweep that touched every file. `SCOPE.md` §2.1's
unsupported-until-needed ruling and #51's repair checklist are superseded: the checklist
moves to `SCOPE.md` or the issue closes there. It rides its own PR because #110 is already
23 files.

*Left for the maintainer: reviewing and merging #110, and approving the #18 close text.
The working tree is on `main`, so production is not running branch code; after the merge,
pull and restart the worker.*

### Step S — slice §8.0.S: v0.4.0, the public signatures — **executed 2026-09-22**

*Eight commits on `slice-s-v0-4-0-signatures`; `NSM/` 13,555 → 13,326; suite 1180 →
1177 passed, 4 skipped, 3 xfailed, `make lint` clean at every commit. Permanent
additions: **+39 lines** against the +60 budget — the two constructor refusals and
their shared helper, and four package imports. **The tag is the maintainer's action
after the merge**, as at v0.3.0: nothing in the PR bumps a version number, because
from v0.3.0 the version is the tag that setuptools-scm derives. The statement the
slice opened with follows, unedited.*

#### The statement, as commit 1

Six Breaking items earlier slices deferred to §8.0.O by name. v0.4.0 is already scheduled
by something else — `NSM/_verbose_deprecation.py` says *delete at v0.4.0* and v0.3.0
existing is what makes that due — so these do not need a boundary invented for them.

Everything below was re-run against `main` at `1e5fd4a` before it was written. Four of the
seven re-measured rows came back different from what the table said, and two of those
change what the slice does.

#### What the items turned out to be

| Item | Measured on `main`, 2026-09-22 | Disposition |
|---|---|---|
| (1) the four public signatures as one set | `reconstruct_mesh` 59 named + `**kwargs`, `reconstruct_latent` 40 + `**kwargs`, `create_mesh_adaptive` 26, `create_mesh` 17 | **Not in S.** Ruled 2026-09-22 to `NSM_CONFIG_SECTIONS_AND_MODEL_REGISTRY.md`; see § Decisions |
| (2) `reconstruct_latent(pts_surface=None)` | `reconstruct_latent_pts_surface_type_check(None)` raises `ValueError: pts_surface must be list, tuple, np.ndarray, or torch.Tensor`. The one in-library caller, `reconstruct/main.py:551`, always passes it | **In S.** Make it required |
| (3) the `lbfgs_*` prefix decision | **Already closed, by §8.0.R.** `test_parameter_surface.TestTheDeferredSitesAreClosed::test_the_lbfgs_triple_is_read_on_the_non_hybrid_path` asserts the constructed `(lr, max_iter, history_size)` is the requested `(1.0, 3, 7)` and not `(0.005, 10, 100)`. No signature change is owed | **Closed.** One docs repair rides along: `docs/KNOWN_ISSUES.md:252` names `TestTheLbfgsParametersAreReadOnBothPaths` as the pin, and that class does not exist anywhere in `testing/` |
| (4) refuse unknown `**kwargs` on `Decoder` / `TriplanarDecoder` | Still open. `Decoder(latent_size=8, dims=[16, 16], paddding=0.35, totally_made_up=7)` builds, and `TriplanarDecoder(paddding=0.35)` builds at `padding=0.1` | **In S**, with its evidence changed — see below |
| (5) delete `max_batch_size` | Still accepted-and-warned, `NSM/reconstruct/latent_fit.py:684`; it is the whole of `latent_fit._DEPRECATED_KWARGS` | **In S** |
| (6) submodule imports in `NSM/mesh/__init__.py` | **Additive, measured.** After a bare `import NSM.mesh`, `refine_mesh`, `correspondence_metrics`, `triangle_metrics` and `interpolate` are all unbound. Adding them binds four names, unbinds none, costs 0.002 s and pulls in no top-level module that `main` had not already loaded | **Rides in S as Added, not Breaking.** It does not need the boundary. It rides because the alternative is a deferral with no carrier, which is the failure §8.0's expiry note describes |
| (7) the `verbose` bridge | **27 decorated sites, not 13**, plus four private functions that take `verbose` undecorated, 31 in all. `NSM/train/train_deep_sdf.py:363` forwards `config["verbose"]` into `get_mean_errors` | **In S**, and it is the item with a consumer constraint on it — see below |

#### Two items the re-measured table lost

§8.0.N′ deferred `compare_cart_thickness`'s `regions_label` and `orig_cart` to "the S row,
by name, where the other seven are". The 2026-09-21 re-measure rebuilt the table from the
§8.0.S row alone and neither survived. Both are still true on `main`:

- `regions_label` is refused unless it is `"labels"`, because pymskt's two readers hardcode
  that name. Its own docstring says it is kept only until this slice removes it.
- `orig_cart` is unpacked from `orig_meshes` and never read again. It is pinned by
  `TestTheOriginalCartilageIsNeverRead`, which passes `None`, a string and an integer in
  its place and gets the same answer each time.

Both go. That is the `CLAUDE.md` remedy for an accepted-and-never-read argument, and it is
what put them in S in the first place.

#### Item (4): the evidence moved, the defect did not

The history file's argument for (4) was a *config* path — `_get_two_stage_params` copied
`config["triplanar_params"]` verbatim into the constructor, so a misspelled key built at
the default and said nothing. That path is gone: §8.0.P deleted `two_stage`, and the three
surviving translators in `models/loader.py` build an explicit `params` dict, so
`load_model` cannot hand a constructor a key it does not name.

What is left is direct construction, which is what the production consumer does
(`kneepipeline/steps/run_nsm.py:112` calls `TriplanarDecoder(**params)`). All 15 keys it
passes are named parameters, so the refusal does not reach it — checked against
`inspect.signature`, not assumed. The four deprecated `Decoder` keys (`xyz_in_all`,
`latent_noise_sigma`, `norm_layers`, `latent_dropout`) keep their current warn-or-refuse
behaviour; the refusal is for everything else.

#### Item (7): deleting the bridge breaks the production consumer, loudly

`kneepipeline/steps/run_nsm.py:211` passes `verbose=True` to `reconstruct_mesh`. Once the
parameter is gone, `refuse_unknown_kwargs` turns that into a `TypeError` on the next NSM
fit. It is loud rather than silent, which is the good case, but this tree is imported
unpinned by that consumer, so **the pull is the deploy** and the next job after a pull is
the one that fails.

The overlap release was supposed to be the warning, and for this consumer it never was:
the bridge's own header records that a `DeprecationWarning` is invisible under Python's
default filter outside `__main__`, and the consumer calls NSM from inside a module.

**Sequencing, stated so it is not discovered at deploy time.** The one-line removal of
`verbose=True` in `kneepipeline/steps/run_nsm.py` lands *before* this tree is pulled into
the production checkout. That consumer change is inert against `main` today — on v0.3.0 the
flag only routes NSM's records to stderr for the duration of the call — so it can land at
any time, and until it does the production tree stays where it is. Merging S does not
deploy S.

A second consequence inside the library: after the deletion nothing in `NSM/` reads
`config["verbose"]`. The key is not refused and every `model_params_config.json` on disk
keeps working, but it selects nothing, and the CHANGELOG entry that called it "an on-disk
format contract that keeps its meaning" needs correcting to say so. The replacement is the
one the deprecation has always named: `logging.getLogger("NSM").setLevel(logging.DEBUG)`
with a handler on it.

#### Permanent and transitional

Everything this slice adds is permanent: two constructor refusals and four package
imports. Everything else it does is deletion. Nothing transitional is created, and one
transitional module is retired — `NSM/_verbose_deprecation.py`, whose header has named
v0.4.0 as its delete-when since §8.0.G.

The two other transitional modules stay, and their conditions are not this release:
`NSM/_lr_migration.py` and `NSM/reconstruct/_config_migration.py` both say *delete once no
config still in use predates the change*, which is a statement about configs on disk and
not about a version number.

#### Size budget

Strongly net-negative in `NSM/`, which is what makes it a release slice rather than a
feature. The permanent additions are about **+40 lines** — two refusals and four imports,
with their messages. Anything past **+60** is scope creep and the deletion pass has to say
why. The deletions are not budgeted, because a budget on a deletion is an argument for
deleting less.

#### Sequence — one commit each, `make lint` clean and the suite green at every step

1. this statement
2. **(2)** `pts_surface` becomes required on `reconstruct_latent`
3. **(5)** `max_batch_size` deleted. `reconstruct_mesh`'s `batch_size_latent_recon` is a
   different key and **stays** — the production consumer passes it
   (`kneepipeline/steps/run_nsm.py:201`), so deleting it is not this slice's to do
4. **(4)** `Decoder` and `TriplanarDecoder` refuse what they do not name
5. **`compare_cart_thickness`** loses `regions_label` and `orig_cart`
6. **(7)** the `verbose` bridge and all 31 parameters deleted; `NSM/_verbose_deprecation.py`
   and its test file go
7. **(6)** `NSM/mesh/__init__.py` exports its four submodules
8. the release cut: CHANGELOG § Unreleased renamed to v0.4.0, the docs swept for claims this
   slice falsifies (including (3)'s missing test name), and this plan's State updated

Then **tag v0.4.0** — the maintainer's action after the merge, as at v0.3.0. Nothing in the
PR bumps a version number; from v0.3.0 the version *is* the tag.

#### What diverged from the statement

Three things, and the statement above is kept unedited so the difference is readable.

- **`orig_cart` is not deleted.** Step 5 of the sequence said it would be, following
  §8.0.N′, which had called it a public signature change. It is not a parameter: it is
  element 1 of a list whose length is the fixed-layout contract, and `reconstruct_mesh`
  hands the same surface layout to both sides (`func(sampled["orig_mesh"], meshes)`).
  Deleting it from the original side alone makes the two lists different shapes and
  renumbers `compare_cart_thickness_whole_joint`'s slicing on one side only. The ruling
  and its reasoning are in `docs/SCOPE.md` §2.5. `regions_label`, which was scheduled in
  the same breath, *was* a parameter and is gone.
- **Two tests became unable to fail and did not survive as they were.** Deleting the
  parameter made the AST sweep asserting no `reconstruct_mesh` record is gated on it
  vacuous — with no `verbose` in scope an `if verbose:` is a `NameError` — so it went. The
  `test_observability` sweep was rewritten to look for the *shape* instead of the name: a
  record reachable only through a parameter of its own function, whatever that parameter
  is called. That one was verified by introducing the shape and watching it go red, which
  is the check the first rewrite of it did not get — a widened predicate that matched any
  `if <anything>: logger.*` flagged fourteen pieces of ordinary control flow. Two
  `train_epoch` tests differing only in `config["verbose"]` collapsed into one.
- **`config["verbose"]` leaves the shipped default config.** The statement said the key
  would become unread and the CHANGELOG entry would say so. It does say so, and the key
  also goes from `default_config.json` and its generator, on §8.0.P's precedent for the
  four `sample_difficulty_lx` keys: a template that writes a key nothing reads manufactures
  the accepted-and-never-read class into every new config. Existing configs are not
  refused.

#### Owed outside this repo, before the release is deployed

`kneepipeline/steps/run_nsm.py:211` passes `verbose=True` to `reconstruct_mesh`. Delete
that line before pulling this tree into the production checkout, or the next NSM fit
raises `TypeError`. The change is inert against `main` today, so it can land at any time,
and it loses only the NSM `DEBUG` chatter the step's log currently carries — put it back
with `logging.getLogger("NSM").setLevel(logging.DEBUG)` if it is wanted. This is the
consumer PR `NSM_CONFIG_SECTIONS_AND_MODEL_REGISTRY.md` Step 4 already anticipates; it is
owed sooner than that, because the pull is the deploy.

### Step T — slice §8.0.T: trim the test suite

Scheduled by the maintainer 2026-08-29: *"they have gotten VERY bloated during this
refactor. Many are valuable, but they are also overkill."* It runs last because every
slice G–S adds to what it trims.

**Measured 2026-09-21** (recompute rather than trusting these — they move every slice):

```bash
python - <<'EOF'
import pathlib
for d in ("testing", "NSM"):
    f = list(pathlib.Path(d).rglob("*.py"))
    print(d, len(f), "files", sum(p.read_text(encoding="utf-8").count("\n") for p in f), "lines")
EOF
python -m pytest testing -q --durations=15
```

| | Scheduled (2026-08-29) | 2026-09-21 | After P (2026-09-22) | After S (2026-09-22) |
|---|---|---|---|---|
| `testing/` vs `NSM/` lines | 14,770 / 14,460 | 16,685 / 15,104 | 16,316 / 13,555 | **16,208 / 13,326** |
| ratio | 1.02 | 1.10 | 1.20 | **1.22** |
| suite wall clock | 109 s | 180 s | 110 s | **110 s** |

**P and S both moved the ratio the wrong way and neither moved the clock.** P cut 1,549
source lines against 369 test lines; S cut 229 source lines against 108 test lines. The
ratio T was created to attack is now its worst ever. The clock halved at P because the
deletions took slow tests with them, and has not moved since. Ten tests are ~50 s of
the 110; `--durations` names them, and the top three are
`test_dataset_cache.py::TestSeedDerivation::test_multiprocessing_does_not_change_the_data`
(10.4 s), `test_train_epoch_internals.py::...test_a_mismatched_weighting_is_still_refused_under_O`
(7.5 s) and four `test_observability.py` subprocess tests at ~6.3 s each.

**Target the wall clock, not a line count.** Rule 3 makes a docstring carrying a
measurement load-bearing, and a quarter of `testing/` is docstrings, so lines removed is
the wrong meter. Ten tests account for about 80 of the 180 seconds; `--durations` names
them. CI can publish the number, which satisfies rule 1.

Three criteria that are checkable instead of aesthetic:

1. **A test that cannot fail.** Revert the fix or delete the assertion and see if it goes
   red. Two were found by accident on 2026-08-29 alone, so the rate is not low.
2. **A characterization test whose strict xfail was retired inside its own slice, and whose
   plain sibling now asserts the same thing.** The pair was deliberate while the defect was
   live and is duplication afterwards. §8.0.R alone retired eleven such xfails.
3. **Overlap between the per-slice `test_*_internals.py` files and the regression harness.**
   Two coverage strategies that grew independently.

Biggest files, as a starting point rather than a target: `test_dataset_cache.py` 1,357,
`regression/_harness.py` 855, `test_reconstruct_latent_internals.py` 795,
`test_train_epoch_internals.py` 660, `test_lr_schedules.py` 628.

**Time-box it.** Two days, then stop and keep what is left. An unbounded aesthetic trim on
a green suite is how this slice fails to end.

### Step Close — retire this plan

Move both this file and `NSM_CODE_HEALTH_REFACTOR_HISTORY.md` to `.claude/plans/completed/`,
adding the two sections `CLAUDE.md` requires: **Delivered** (what shipped, with PR links)
and **Diverged** (where reality differed from the plan, and why). `Diverged` is the most
valuable thing in the file and exists nowhere else, so do not compress it — the Surprises
above and the eighteen slice statements in the history file are its raw material.

Then start `NSM_CONFIG_SECTIONS_AND_MODEL_REGISTRY.md`, which is already written and
waiting on its own §2 layout ruling.

---

## Decisions the maintainer owes

**Both were ruled 2026-09-22 and are struck through below.** Nothing is outstanding. They
are kept rather than deleted because each records an argument that outlives its answer.

1. ~~**Does slice S shrink the four public signatures, or does the config initiative?**~~
   **Ruled 2026-09-22: the config initiative.** S ships its other six items and leaves the
   59-parameter `reconstruct_mesh` alone; `NSM_CONFIG_SECTIONS_AND_MODEL_REGISTRY.md` takes
   item (1) when it gets there. The reasoning, kept because it is the argument rather than
   the answer: **shrinking it now and regrouping it later is two breaking changes for one
   result.**
   The 59-parameter `reconstruct_mesh` is exactly where the wanted "set the reference mesh"
   option lands, and the config plan's `recon` section is what that signature should
   mirror. Shrinking it now and regrouping it later is two breaking changes for one result.
   This is narrower than the config plan's own §5 proposal to absorb **all** of S, which
   should be declined — that would tie the refactor's ending to a new feature being
   designed and built.
2. ~~**Close the `grad_clip` entry in `docs/KNOWN_ISSUES.md` § Open as working by design?**~~
   **Ruled 2026-09-22: yes; executed in PR #111.** The entry leaves § Open and the ruling
   lands in `SCOPE.md` under a new *working by design* heading, because a won't-fix that
   closes nowhere becomes the eleventh document. The upstream comparison was **re-fetched
   and re-read** rather than inherited from the note below. `train_epoch` passes `model.parameters()` to `clip_grad_norm_`
   and never the latent embedding. Verified 2026-09-21 against the upstream reference:
   `facebookresearch/DeepSDF/train_deep_sdf.py:521` clips `decoder.parameters()` and never
   `lat_vecs`, so NSM matches it. The knob is `null` in the shipped default and in both
   production model configs, so nothing has ever set it. The latents already have their own
   L2 regularization with warmup and their own LR schedule. The 2026-08-22 note proposes an
   experiment; dropping it moves the row out of § Open. A test already pins the current
   behaviour (`test_parameter_surface.TestGradClipReachesTheModelOnly`).

**Also available, not required.** The 1,450-line State narrative now in the history file is
superseded by PR descriptions, `CHANGELOG.md` and `docs/KNOWN_ISSUES.md`. It was archived
rather than deleted so the choice stays open. Deleting it is safe — git history keeps it.

## Working conventions

Not repeated here. `CLAUDE.md` § Working on this repo is the authority: the `nsm-dev`
environment, `make lint` at zero, admin merge as the normal path on a protected `main`,
one commit per concern with review feedback landing on top, no AI attribution lines, and
nothing posted to the public tracker without the maintainer approving the exact text.

Two that bite on the slices below: plan-only text commits straight to `main` with no PR,
and a test that reads a file needs `encoding="utf-8"` explicitly.

---

## Not this plan

The three sections below are the register of things that were considered and ruled out of
the refactor. They are live decisions, not history, which is why they stay in the active
file. §8.3 is the index; §8.1 and §8.2 hold the evidence behind two of its rows.

### 8.1 Make the library plural — added 2026-08-15

> **Deferred 2026-08-26 — this is an upgrade, not the refactor.** All three bullets are
> new capability, and they are indexed in §8.3. The one thread that does *not* wait is the
> defensive half of #26, which §8.0.H carries: refusing to load a checkpoint whose config
> omits `padding` is a silent-wrong-answer fix, where "a third party adds a model type and
> it works everywhere" is a feature. Keep them apart — #26's issue text lists them as
> options 1 and 3 of the same fix, and taking option 3 first is how this section swallows
> the slice.

The name is *neural shape model**s***. Two defects currently make it singular in practice,
and both are structural, so they belong here rather than in the documentation pass.

- [ ] **A common decoder interface and a registration pathway.** Today
      `reconstruct/main.py:588` hardcodes a keyword-only `(latent=, xyz=)` call with no
      fallback, while `mesh/main.py:855-867` inspects the signature and falls back — two
      conventions in one pipeline. Three of `load_model`'s four advertised model types
      cannot be reconstructed. The goal is that a third party adds a model and it works in
      train, reconstruct, mesh and interpolate without editing NSM internals. One calling
      convention wins; every shipped checkpoint keeps loading.
      `models/loader.py` is where this hangs — see the open question in `docs/SCOPE.md` §2.6
      about whether the consumer could switch to `load_model` today.
- [ ] **A default config per model type, derived from the ShapeMedKnee configs.** The
      shipped `default_config.json` has 61 DeepSDF-shaped keys; the real production configs
      have 131. A triplanar model built from the shipped default silently falls back to a
      different architecture. Generate them the way the current one is generated, and pin
      them the way `testing/NSM/configs/test_default_config_sync.py` already pins it.
- [ ] **Config naming, validation and documentation.** The maintainer's assessment: the
      options are poorly documented and many names do not describe what the code does with
      them. Open sub-question — whether this needs a restructured config format or only
      renames plus validation. Every rename breaks a file someone has on disk, so the
      migration cost gets measured before the shape is chosen. The LR `Target` key
      (`NSM/utils.py` + `NSM/_lr_migration.py`) is the in-repo reference for how a config
      change fails loudly and hands the user a corrected copy of their own file.

**Sequencing note.** Both items change public behaviour, so both need §4-style migration
guards and §9 ledger entries, and both need §7.1 green first. The registry is the one that
unlocks the others: until there is one supported way to build a model from a config,
"a default config per model type" has no single consumer to be correct for.

### 8.2 Eikonal loss — gated 2026-08-15, needs repair

> **Deferred 2026-08-26 — the refactor's part of this is done.** Gating a never-runnable
> path behind `NotImplementedError` *was* the code-health outcome; making the loss work is
> research, tracked as `NSM_TRAINING_IDEAS.md` Idea 3. The three failures below stay here
> because they are the executed evidence, and whoever picks the research up needs them.
> Indexed in §8.3.

`eikonal_weight > 0` now raises `NotImplementedError` at both entry points
(`train_deep_sdf`, `reconstruct_latent`), with the message in `NSM/losses.py`.
`testing/NSM/test_losses.py` pins it and is written to **fail once the loss works** —
deleting that file is part of fixing this. No results are affected — the path always crashed, so per `CLAUDE.md` it gets no
`KNOWN_ISSUES.md` entry. Neither ShapeMedKnee config contains the key and
`kneepipeline` never passes it; production has never touched this code.

Three independent failures, in the order they must be fixed:

- [ ] **It crashes on the first backward pass.** `losses.py:54` reads
      `retain_graph=True if surf_idx < n_surfaces - 1 else False` — on the last (or only)
      surface that frees the forward graph the double-backward graph still needs, so the
      caller's `.backward()` raises. Verified for 1, 2 and 4 surfaces. One-line fix; the
      same pattern repeats in `compute_sdf_gradients`.
- [ ] **Triplanar models cannot use it at all.** It needs a second derivative through
      `grid_sample`, which PyTorch does not implement — verified on CPU and on a T4 with
      torch 2.8.0. The first-order gradient computes fine; the backward *through* it does
      not. This is not ours to fix, so the guard must stay for triplanar regardless: any
      future support is MLP-architectures-only until upstream changes.
- [ ] **It opposes clamped training,** which is the regime NSM actually uses
      (`enforce_minmax: true`, `clamp_dist: 1` in both production configs). A clamped
      target is flat outside the band, so its true gradient norm is 0, not 1. Measured on
      an analytic sphere at the generator's own suggested `0.1`: gradient norm converges
      (mean `|‖∇f‖−1|` 0.928 → 0.0091) while the zero level set is destroyed (surface error
      0.162 → 1.050). Unclamped, it is mildly helpful (0.0194 → 0.0172). If adopted, either
      restrict eikonal sample points to inside the clamp band or require
      `enforce_minmax: false`.
- [ ] Secondary: the eikonal term is computed on the **unclamped** prediction
      (`train_deep_sdf.py:510`) while L1 uses the clamped one (`:398`) — a second full
      forward pass, costing ~4x step time and ~3x memory when enabled.
- [ ] Secondary: `reconstruct/main.py` calls it under `torch.no_grad()` at `:723-724`,
      which raises independently of the above.

**When it is fixed, the test is the deliverable, not the fix** — an analytic sphere where
the gradient norm is measurably closer to 1 with the loss on, and a backwardable-regression
guard parametrised over 1/2/4 surfaces. Without that, "does it help" is unanswerable again
in a year.

**Rule:** every commit keeps the §7.1 harness green. Any intended numerical change gets a
§9 ledger entry and a §4-style migration guard.

---

### 8.3 Deferred until the refactor closes — drawn 2026-08-26

The plan's own §Scope banner has always said new science is out of scope. What it did not
say is that §8 had grown three items that are not code health either, and one of them had
reached the **Next** line. This is the list, so that deferring them is a decision with a
venue rather than a thing that keeps not happening.

| Deferred | Where it lives | Why it is not refactor |
|---|---|---|
| §8.1 — decoder interface, per-type default configs, config renames | here, banner above | New capability. Every bullet changes public behaviour and needs its own migration guard; none of them makes existing code more correct. |
| §8.2 — repairing the Eikonal loss | here + `NSM_TRAINING_IDEAS.md` Idea 3 | The loss is gated, which is the code-health answer. Making it work is an experiment with an unknown result — measured to *oppose* the clamped regime NSM actually trains in. |
| Idea 4 — the latent norm bound, training and recon sides | `NSM_TRAINING_IDEAS.md` Idea 4/7/11 | A training experiment. It sat in this plan's **Next** on 2026-08-25 while §8 had eleven unstarted slices; that is the specific failure this table exists to prevent. |
| Ideas 6, 10, 11, 12 | `NSM_TRAINING_IDEAS.md` | Same. Each is independently executable and none of them is blocked by the refactor. |
| #2 — `SDFSamples` slow loading | issue #2 | Performance. Real, but it is an optimisation, and §8.0.F just rewrote the cache path it would target. |
| §9's fourth bullet — whether the LR bug moved published results | §9 | A research assessment of finished work, not a change to the library. |
| #67 — training on subjects missing a surface | issue #67 + `SCOPE.md` §2 | *Added 2026-08-29.* New capability, and it needs data nobody has yet. The half that is code health — the reconstruct path, which works — is pinned by §8.0.N′'s end-to-end test. Nothing about the dataset half makes existing code more correct. |
| #35 — `reconstruct_latent_S3(log_wandb=True)` raises | issue #35 | *Added 2026-08-29.* Two-line fix on an opt-in logging path of a module SCOPE §2.4 rules deferred research. Revisit when something opens the file; creating a slice for it would be scheduling work ahead of six things that matter more. |

**The test for this table.** An item belongs in §8 if a reader would call the outcome
"the code was wrong and now it is right". If the honest description is "we tried something
and measured what happened", it belongs in the ideas file. #3 (sigma) stays in §8 under
that test — the same number means two things and one of them is wrong — which is why it is
§8.0.Q and not a row here. *Superseded for #3 on 2026-08-30: the maintainer moved it out
anyway. Passing the test says it is a fix, not that this plan must carry it — it is now
its own initiative in the two sigma plans, and this table's fence is unchanged for
everything else.*

---

## 9. Deliverable: the bug provenance ledger

New file `docs/KNOWN_ISSUES.md`. For science code this is a first-class artifact —
it answers "which of my results are affected?", which a code comment cannot.

Each entry: what was wrong, exact date range affected, which configs/optimizers/code paths,
observable consequence, how to detect it in an existing run, how to reproduce old behaviour.

- [x] Seed with the LR-schedule bug (May 2023 → Jul 2026, Adam/AdamW only, `schedule_free_*`
      unaffected). Prior to this it existed only as a docstring in a downstream fork — it
      must live somewhere durable and citable. *(`docs/KNOWN_ISSUES.md` § History 1, with
      the two opposite migrations by optimizer family. Ticked 2026-08-29; it has been the
      first entry in that file since Phase A.)*
- [x] Add the sigma coordinate-space ambiguity (issue #3). *(In `KNOWN_ISSUES` § Open
      since Phase 1 — `BREAKING_CHANGE_PROPOSAL.md`'s State records it. The § History
      entry lands with the fix, which is no longer this plan's to make: §8.0.Q re-homed
      to the two sigma plans on 2026-08-30.)*
- [ ] Add every subsequent finding from Phases 1–4.
- **Three findings from §7.5b's training runs (2026-08-31), recorded here rather than in
  `KNOWN_ISSUES.md`, each for a stated reason:**
  1. **`layer_split: False` meant "split at layer 0", and a real historical config relied
     on it.** Pre-fix, `False == 0` duplicated the whole MLP per object (4,742,148
     params); `main` coerces `False → None` (shared trunk, 2,371,588). This is #46 /
     § History 14 confirmed against a config someone actually trained with, which is what
     the entry was missing. Two consequences worth carrying: reproducing the historical
     architecture on `main` needs an explicit **`layer_split: 0`** (7.5b's MLP cells did,
     which is why both report 4,742,148 params and the comparison is architecture-matched),
     and `False == 0` makes the difference **invisible to dict-equality config diffing** —
     so "the configs are identical" does not imply the architectures are.
  2. **A pre-refactor crash the refactor fixed silently**, and deliberately *not* filed:
     at `bb2c6a3`, `scale_jointly=True` + `store_data_in_memory=True` raises
     `KeyError: 'new_pts_0'` in `norm_and_scale_all_meshes` — only the on-disk npz branch
     creates the numbered keys, and the in-memory branch reads keys nothing writes.
     `main` handles both. It is a different site from #22 (`UnboundLocalError` on `time_`,
     closed), so it was genuinely unrecorded. **No issue and no § History entry, per
     `CLAUDE.md`: it always crashed, so nobody holds results from it, and it is already
     fixed.** The ledger is the right and only home.
  3. **`mskt>=0.1.21` is a hard floor for `main`'s sampling path** (the `seed=` kwarg on
     `rand_pts_around_surface`), and the sampling library version must *match across refs*
     for cache comparability — 7.5b pinned both envs to 0.1.21 for exactly that reason.
     Already declared in `requirements.txt`; what is new is that a version *skew* between
     two refs invalidates a comparison even when both versions are individually supported.
- [ ] Assess whether the LR bug materially affected published/downstream results. Initial
      read: the hyperparameter search ran under the buggy mapping, so the chosen values were
      optimal *for that mapping*. The models are self-consistent; retuning under the fixed
      mapping is a separate exercise and is not a prerequisite for anything here.

---

## 10. Sequencing and risk

```
Phase 0 (scope)  ──►  Phase 1 (map)  ──►  Phase 2 (docs)
                            │                   │
                            └──────►  Phase 3 (tests) ──►  Phase 4 (decompose)
                                            ▲                      │
                                            └──── 7.3 just-in-time ┘
```

### 10.1 Versioning — decided 2026-08-15

The original text here said "before starting Phase 1, tag a release (`v2.x`)". That could
not be done as written: `pyproject.toml` derives the version from `NSM.__version__`, which
is the string literal `"0.0.1"` and has never been bumped, so there is no `v1` and `v2.x`
would invent history. Phase 1 ran without a tag; the tag is now a Phase 3 prerequisite
instead, since it is Phase 4 that breaks things.

- [x] **Tag `v0.1.0` — "the state before the refactor."** Have `kneepipeline` and `nsosim`
      pin it. This is the rollback point and it decouples their release cadence from this
      work. *(Tagged — the `Target` requirement is in it, see Surprises. The
      consumer-pinning half never happened: kneepipeline consumes a checked-out working
      tree (`DEPENDENCIES/nsm`), not a tag; nsosim is 0b's question.)*
- [x] **Not `1.0.0`.** That is a stability promise, and Phase 1 found 71 landmines with 30
      of them unverified on the production path. Claiming 1.0 and then breaking things in
      Phase 4 makes the number meaningless. `0.x` is honest and gives the same rollback
      guarantee. *(Holding: v0.1.0 and v0.2.0 both shipped under `0.x`.)*
- [x] **Bump on release, not on commit.** Under `0.x`: breaking changes bump the minor
      (`0.2.0` after Phase 4), additive changes bump the patch. *(Practiced: v0.2.0
      (PR #36); the pending Breaking set makes the next cut v0.3.0 — State § Versioning.)*
- [ ] **Move to `1.0.0` when there is something to promise** — when `__all__` exists (§3)
      and the §7.1 harness is green. That ties the version to a milestone rather than a date.
      *Both conditions are now met and 1.0.0 is still wrong, which is worth recording: the
      §3.2 stability **tiering** is what a 1.0 promise needs, and `__all__` as shipped is
      the mechanical export list, not that ruling. The gate was stated as the artifact when
      what it meant was the decision.*
- [x] **Derive the version from git tags.** Done in §8.0.O. *Uncommenting
      `[tool.setuptools_scm]` was not the whole change, and the two missing halves were
      measured rather than reasoned about: with no git metadata `pip wheel` **fails
      outright** (`fallback_version` fixes it), and `actions/checkout@v2` clones at depth 1
      with no tags, which does not fail — it builds `0.0.1.dev1+unknown.g<sha>`
      (`fetch-depth: 0` fixes it). `NSM.__version__` reads installed metadata, with a
      not-installed fallback because the one real consumer reaches NSM by `sys.path`.*

**Coordinate with downstream forks throughout.** At least one active fork carries modules
that do not exist upstream, so every week of unmerged refactor makes its merge worse.
Phases 0 and 1 in particular should be shared before execution — fork module usage is an
input to the dead-code call.

**Biggest risk:** Phase 3 stalls again. Mitigation — §7.1 is a single bounded artifact with
a clear done condition, delivered before any broad coverage push. If only §7.1 and §7.2 ever
land, the library is still meaningfully safer than it is today.

---

## 11. Related documents

| Document | Status | Relationship |
|---|---|---|
| `.claude/plans/NSM_CODE_HEALTH_REFACTOR_HISTORY.md` | Executed record | This plan's own history — the pre-2026-09-21 State narrative and all eighteen slice statements §8.0.A–R, verbatim. Retires to `completed/` with this file |
| `.claude/plans/NSM_CONFIG_SECTIONS_AND_MODEL_REGISTRY.md` | Open, blocked on its §2 | **The next initiative after this one closes.** Started by §8.0.R's finding that every live accepted-and-ignored parameter is in `models/loader.py`. Also owns the MPA loader fix, which is unblocked but deliberately held until after v0.4.0 |
| `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` | Own initiative since 2026-08-30 | Was "fold into Phase 4"; §8.0.Q re-homed to it instead — the *what and why* of #3 |
| `.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` | Own initiative since 2026-08-30 | Was "fold into Phase 4"; the *how* of #3 — its excerpts are stale, re-verify before executing |
| `.claude/plans/HYBRID_OPTIMIZER_REPORT.md` | Findings, Aug 2025 | Reference for `reconstruct/main.py` |
| `.claude/plans/NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` | Proposed | Blocked on stable interpolation API |
| `.claude/plans/NSM_TRAINING_IDEAS.md` | Open master list | Idea 3 (Eikonal loss) — **research, not Phase 3**; §8.3 supersedes this row's original claim, and §8.2 holds the executed evidence |
| `.claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md` | Complete 2026-05-22 | Target-state example |
| `docs/MULTI_SURFACE_REGISTRATION.md` | Current | Feature doc, verify in Phase 2 |
