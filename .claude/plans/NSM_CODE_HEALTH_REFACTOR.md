# Plan: NSM code-health audit and refactor

**Repo:** `gattia/nsm` (NSM). **Created:** 2026-08-14.

## State

**Updated:** 2026-08-23 · **Status:** open

- **Next:** the remaining §8 program is all grouped work awaiting its own
  statement: the evaluation-module split out of `reconstruct/main.py` rides with
  #5 (wandb-optional), class-side cache/build decomposition with #19/#27 (one
  migration release), `train_epoch`'s internal loss-pipeline decomposition only
  if a statement justifies it, multi_head's repair is #51, and the v0.3.0 cut
  ("soonish, or at the end of this cleanup") is the maintainer's call. In review:
  PR #77 (docs-only — the `enforce_minmax` Open entry gains its full gradient
  mechanics and stays open by maintainer ruling 2026-08-24; Idea 11 gains the
  clamp-form axis). Meanwhile the maintainer's research queue points at
  `NSM_TRAINING_IDEAS.md` Idea 4(a) first (see that file's State).
- **§8.0.D merged to `main` in PR #76 (2026-08-24):** #28, #42, #49, #52, #59
  closed by the merge.
- **§8.0.D landed on branch (2026-08-23):** statement → characterization → #42 →
  #49 → #59 → sweep → #28 → #52 → split → this update, suite green and lint
  clean at every commit (654→660 passed; 19→12 xfailed — every conversion a fix
  passing unmarked). #42: the warm-up unpacks the way `train_epoch` does, pinned
  via a stubbed `schedulefree` (AdamW + `train()`/`eval()`) since the crash was
  the trainer's, not the library's. #49: one boundary — `resume_epoch` names the
  last completed epoch, `>= 1` loads (History §11). #59: `+=`, and
  `add_plain_lr_to_config` loses its positional override (History §12 —
  wandb-only, ~×n_batches). Sweep: `train_epoch(return_loss, verbose)` deleted
  live-trainer-only (multi_head is #51's, `deprecated/` is 0b's). #28: history
  returned (wandb payload + epoch/lrs/targets/latent_norms per epoch, payload
  byte-identical); the harness's `train_epoch` wrapper deleted, baselines
  unmoved — the regression suite now tests the public contract. #52: names
  declared on `MultiSurfaceSDFSamples` in mesh-path-list order, trainer adopts
  or refuses at entry; deliberately not in the cache key. Split: four private
  helpers, `train_epoch` whole, net +57 of +80. **Diverged from the statement:**
  `_schedule_free_eval_warmup` also takes `optimizer` (the statement's signature
  omitted it, but `optimizer.eval()` lives in the block); #52's fix had to touch
  the pre-existing validation tests — a bare `MagicMock` dataset auto-creates a
  truthy `mesh_names` the adoption path would read, so those mocks now pin
  `mesh_names=None`; a stale `train_deep_sdf.py:401` line-number citation in the
  clamp characterization docstring rode along (symbols only, per #31).
- **§8.0.C merged to `main` in PR #74; the #48 remnants in PR #73 (both
  2026-08-23):** #15, #16, #29 closed by the merge.
- **§8.0.C landed on branch (2026-08-23):** statement → characterization → #15 →
  #16 → class sweep → #29 → split → this update, suite green and lint clean at
  every commit. #15: readers unified on `"pts"`; the two dataset-class probes
  collapsed, their dead `"gt_sdf"` fallback with them. The `"xyz"` transitional
  alias was then **deleted on the same branch** (maintainer, 2026-08-23:
  sooner-than-later — and since no tagged release ever carried it, waiting for
  v0.3.0 would have meant it never shipped at all). #15 is now a clean Breaking
  rename in the CHANGELOG; external `["xyz"]` readers get a loud `KeyError`. #16: honoured (History §9) — the
  memory said delete, but the measured blast radius is only never-shipped
  multi-object sampled runs and the intent is config-plumbed; the harness's
  sampled tests shrank ~1000× and the suite dropped ~11s. #29: raises
  `NoZeroLevelSetError` by name; `get_mean_errors` catches and scores NaN
  (History §10). The split: `latent_fit.py` + `wandb_logging.py`, `main.py` keeps
  `reconstruct_mesh` + evaluation + a permanent re-import block, net +67 of the
  +80 budget, both namespaces frozen by test. **Diverged from the statement:**
  the sweep had to reach all four trainer call sites and the default-config
  generator (the statement named only `train_deep_sdf.py:238`) — the plumbing was
  wider than enumerated; #29's planned NaN-latent-row into `Regress` was replaced
  by skip-plus-NaN-r² because sklearn refuses NaN design rows and degeneracy is
  decoder-level, so the failure is all-or-nothing; the three Open table rows
  retired together in the #29 commit rather than one per fix commit; a
  single-object end-to-end test rode with #16 (the branch was never runnable
  before #15+#16, so its first execution belonged with the fix that made it
  cheap).
- **#48 remnants on branch `recon-option-values`, draft PR #73 (2026-08-23):**
  `norm_penalty_type='barrier'` raises by name outside its `(min, max)` range
  instead of NaN — pre-fix runs *completed*, with `nan` in every loss readout and
  a finite gradient pushing the norm **away** from the range (verified by
  execution; History §8, CHANGELOG); `get_mean_errors` hands `Regress.add_latent`
  the fitted latent (detached/CPU/flat), not the result dict — born broken in
  `2811d27` (Jul 2023), never worked, always crashed → no History entry.
  Ride-alongs the work exposed: the docs-reference checker's `Regress` exemption
  was dead and its comment false (the class is in this repo) — deleted; SCOPE
  §2.5's "worth pinning down" question answered (never-worked, not
  broken-after-the-paper). Suite 505 passed / 14 xfailed.
- **Slice B landed (2026-08-23):** merged to `main` in PR #72 —
  `read_meshes_get_sampled_pts` orchestrates three private helpers
  (`_register_to_mean`, `_compute_shared_frame`, `_draw_surface_samples`); #17 and
  #69 fixed en route, both closed by the merge (#17: `KNOWN_ISSUES` § History §7
  records the one affected run class — multi-object fits with `get_rand_pts=True`
  on `scale_jointly=False` models, never training data; #69: fixed by unification,
  the in-memory mutate-in-place branch deleted net −48, disk numerics unchanged).
  #3 got its structural precondition only: sigma's frame-dependence is stated in
  `_draw_surface_samples`' docstring, the single site where sigma is consumed.
- **Slice A landed (2026-08-22):** merged to `main` in PR #71 — `sdf_dataset.py`
  holds the classes + permanent re-import block, the 13 leaf helpers in
  `NSM/datasets/utils.py`, the two readers in `NSM/datasets/mesh_sampling.py`;
  characterization in `test_dataset_helpers.py` / `test_sampled_pts_readers.py` /
  `test_import_compat.py` (frozen name list on both import paths).
  Characterization surprise, pinned and docstring-corrected: `unpack_numpy_data`
  accepts a dict only with `list_additional_keys=[]` — the default reads
  `data_.files`, NpzFile-only — despite its (now fixed) "NpzFile or dict" claim.
- **Scale preservation is an idea, not a defect** (maintainer, 2026-08-22): similarity
  registration deliberately matching subjects to the reference's size is a valid
  design; keeping true size is an *additional mode* worth having. It therefore lives
  as `NSM_TRAINING_IDEAS.md` Idea 6 (with the measured evidence and design sketch),
  not in `KNOWN_ISSUES.md` and not on the tracker; the durable fact — similarity =
  rigid + uniform scale, size does not survive registration — is stated in the
  `reference_mesh` / `register_to_mean_first` docstrings.
- **Versioning (2026-08-23):** released state is `v0.2.0` (tag + `__version__`
  agree). The Unreleased CHANGELOG section holds this branch's Breaking set, so
  the release that ships it is **v0.3.0** per §10.1 (breaking bumps the minor).
  Cut timing is the maintainer's call — "soonish, or at the end of this cleanup";
  nothing gates it now that the `"xyz"` alias is gone rather than
  waiting-to-be-deleted. §10.1's setuptools-scm item (derive the version from the
  tag) is still open and would naturally ride with the v0.3.0 release PR.
- **Blocked on:** nothing.
- **Context for whoever picks this up:** PR #68 carried one commit per concern, so
  `git log NSM/datasets/sdf_dataset.py` explains each fix. Decisions of record are in
  the PR body ("Of note"): the single-mesh clip was removed, not copied; the timing keys
  are optional diagnostics, not batch contract; `subsample=None` is refused, not
  resurrected; `joint_scale_buffer` deliberately stays out of the cache key until #19.
  The one results-affecting change is `docs/KNOWN_ISSUES.md` § History 6 (sampling cube),
  including the trap that pre-fix caches keep serving old points because the buffer is
  not in the cache key.
- **Deliberately deferred:** #19 (cache key, 6 xfails) and #27 (checkpoint aliasing). Both
  force downstream regeneration or migration, so they land together as one release rather
  than making consumers migrate twice. This is the argument that grouped #19 in the first
  place, applied one level up.
- **Done:**
  - Phase 0 scope — `docs/SCOPE.md`
  - Phase 1 map — `docs/ARCHITECTURE.md`, `docs/AUDIT_FINDINGS.md`
  - Phase A LR fix — PRs #9, #10, #11
  - Phase 3 §7.1 numerical regression harness — PRs #13, #14, merged to `main` in #30
  - Seeding fix + harness punch-list — `eaee68c`, `e432f9a`, `d2ba1c7`, `94b48f0`
  - **Everything above reached `main` for the first time in PR #30.** `main` had seen none
    of it before that.
  - Docs cite code by symbol, not line number, with a test — PR #31
  - flake8 to zero, CI gates it, tooling backlog closed, docs build set up — PR #32
  - Retired `worklist #N` numbering repointed at issues; the inverted `normalize_coordinates`
    test rewritten — PR #33
  - `CHANGELOG.md` and `v0.2.0` — PR #36
  - Phase 2, mechanical slice: docstrings that contradict their signatures — PR #37
  - #21 closed and #20's function-level half — PR #38
  - `main` protected: 1 review, four required status checks, admins exempt
  - Audit disposition approved and filed 2026-08-22: issues #40–#61 (mapping in
    `AUDIT_FINDINGS.md` § 0.3), #6 closed, folds commented onto #20/#22/#23/#35 —
    landed via PRs #62 (disposition) and #63 (quick wins)
  - Wave 1 — the four decided fixes (#47, #44, #53, #48-partial) plus the
    cyclic-anneal and Constant-schedule ride-alongs — PR #64
  - SCOPE rulings pass: the register's 13 § 4 rulings transcribed into SCOPE.md
    (every claim re-run against `main` first), the § 0.5 dead cluster deleted
    (`symmetric_chammfer`, `sdf_gradients`, `find_object_bounds_random_sampling`,
    `NSM/configs/deep_sdf_config`), and the stale §1 default-config bullet corrected
    post-#64 — PR #65
  - Phase-2 prose pass: all 62 § 3 corrections (two were already moot — sinkhorn left
    with #64's EMD deletion, the modulated-activations print with #63 — and the
    sdf_dataset unused-imports entry was a deliberate skip per its verdict), plus the
    `get_latent_vecs` variational docstring, plus ride-alongs the corrections exposed
    (Makefile `PDOC_ALLOW_EXEC` had lost its rationale to #64 and was removed after
    re-verifying the build; the never-pushed mesh-interp archive branch/tag is now
    recorded in the completed interpolation plan's Diverged) — and **the register
    `docs/AUDIT_FINDINGS.md` is deleted**, PR #66. Draft→issue mapping, preserved
    from its § 0.3: 1→#40, 2→#41, 3→#42, 4→#43, 5→#44, 6→#45, 7→#46, 8→#47, 9→#48,
    10→#49, 11→#50, 12→#51, 14→#52, 15→#53, 16→#54, 17→#55, 18→#56, 19→#57, 20→#58,
    21→#59, 22→#60, 23→#61 (13 withdrawn — the variational behaviour is deliberate)
  - `sdf_dataset.py` file:function fixes, maintainer-reviewed 2026-08-22: #40 (one
    buffered-cube helper, symmetric and unclipped — History §6), #41 (empty pos/neg
    raises by name; unused surfaces handled), #43 (`subsample` validated,
    `joint_scale_buffer` accepted), #61 (integer-reference registration path;
    `combine_meshes` keeps its Mesh contract), #22 (in-memory datasets train; timing
    keys optional), #23 (zero-count combos skipped), #24 (`LOC_SDF_CACHE` at
    construction) — one commit per concern, all closed by PR #68. Also filed and
    pinned strict-xfail: #67 (the None-surface path has never built end to end)
  - `sdf_dataset.py` semantic docstring pass — landed in PR #70, merged 2026-08-22:
    `mean` deleted from both samplers —
    the file's only accepted-and-never-read parameter left, by AST scan (#20 instance;
    CHANGELOG entry); every public function/method documented, silent conventions
    written down ("pts"/"xyz" key asymmetry, npz vs in-memory spelling, batch
    contracts, cache-key omissions), false text corrected; the in-memory
    joint-scaling defect pinned strict-xfail, filed as #69 with approved text and a
    `KNOWN_ISSUES` § Open entry; `_harness.py`'s stale import-time `LOC_SDF_CACHE`
    claim fixed (stale since #24); the scale-erasure fact measured (ICP scale factor
    exactly 1/1.3 on a 1.3-vs-1.0 sphere pair) and recorded as `NSM_TRAINING_IDEAS.md`
    Idea 6 plus docstring clauses — initially misfiled as a KNOWN_ISSUES entry, moved
    on the maintainer's call that it is a design alternative, not a defect
- **Surprises:**
  - **"Fixed seed" was not available.** NSM called no seeding function anywhere, and the
    near-surface sampler could not be seeded by any caller. Closed via pymskt 0.1.21 plus
    `derive_seed`; see `docs/KNOWN_ISSUES.md` § History.
  - **The seed had to be *derived*, per (subject, sampling pass, surface), keyed on mesh
    content.** Keying on list position and then on the cache hash each silently changed a
    subject's data when something unrelated moved.
  - **"Pin current behaviour, bugs included" reported the opposite of the truth.** Written
    that way, ~20 tests passed *because* something was broken. They now assert the
    behaviour NSM should have, marked `xfail(strict=True)`.
  - **Quarantinable code was 882 lines, not ~1,800.** Every module the plan expected to
    fall held a capability nothing else implements.
  - **Both inferred audit claims that were later tested were wrong**, in the direction of
    overstatement. This is why `AUDIT_FINDINGS.md` entries are hypotheses, not findings.
  - **The "obvious fix" has now been worse than the bug twice**, and the second was far
    worse than the first. `normalize_coordinates`' `padding` would have shifted the SDF by
    ~0.06 on a `tanh`-bounded output. `get_pts_center_and_scale`'s `center`/`scale` would
    have switched scaling **off** on every default run, because every caller passes
    `scale=norm_pts` and `norm_pts` defaults to `False` — a different coordinate frame for
    every dataset and checkpoint ever produced. Both were deleted rather than honoured.
    **Assume the next accepted-and-ignored argument is the same shape until measured.**
  - **`norm_pts: false` has never done what it says.** `center_pts` and `norm_pts` decide
    *whether* to normalize, not *which* operation runs, so the shipped config asks for
    centering without scaling and gets both. No NSM run has ever been centered-but-unscaled.
  - **Line numbers in prose do not survive editing the code they describe.** A seven-line
    portability fix invalidated ~15 citations in the same sitting, and a `black` pass would
    have moved them again. Citations are now symbols, checked by a test (#31). A checker for
    line *numbers* was considered and rejected: it converts silent rot into recurring
    transcription work.
  - **The 439 flake8 violations were 302 formatting, 110 prose, 27 real.** The backlog that
    justified `continue-on-error` for years was two commands and an afternoon.
  - **CI's lint step never ran `black` or `isort`** despite being named for them. That is
    how nine files drifted out of compliance unseen.
  - **`TestAFreshlyTrainedDecoder` passes with the training discarded** — verified by
    rebuilding the model after training and rerunning. An untrained decoder still yields a
    surface. Filed as #34.
  - **The `Target` requirement is already in `v0.1.0`**, so no released version predates it.
    A downstream on positional entries cannot dodge the migration by pinning an older tag.
  - **Something in the suite resets the locale to ASCII.** A bare `read_text()` passes in
    isolation and raises `UnicodeDecodeError` under the full suite; cost 27 failures to find.
  - **#41's None-surface trigger cannot reach `sdf_pos_neg_idx` end to end.** A `None`
    surface dies earlier, in `MultiSurfaceSDFSamples.get_sample_data_dict`: the buffer is
    preallocated `sum(n_pts_)` rows but the sampler returns only the non-None surfaces'
    points. The fdfe902 feature has therefore never worked through the dataset class;
    filed as #67, pinned strict-xfail, and the NaN-column handling is tested by direct
    method call until it is fixed.
  - **The shipped default config carries `dataset_uniform_pts_buffer: 0.2`**, so #40's
    asymmetric-cube bug affected real training data, not just a dormant parameter — it
    became History §6 rather than close-by-deletion. And because the buffer is absent
    from the cache key (#19), the fix alone does not resample: pre-fix `.npz` caches
    keep serving the old points until deleted.
  - **The docstring pass found a never-worked configuration the fix pass had missed.**
    Writing `norm_and_scale_all_meshes`' docstring forced running its in-memory branch:
    `KeyError` on both classes, plus a silently-dropped `joint_scale_buffer` behind it.
    Documenting a function honestly means executing it — the branch had survived #22's
    in-memory fixes and #43's `joint_scale_buffer` work untouched because neither had a
    reason to run that exact combination.
  - **#17's blast radius was narrower than the issue feared.** The executed determination
    (§8.0.B) showed only *one* configuration of `include_surf_in_pts` ever returned
    results — centering on, numeric sigmas; the production-shaped configuration
    (`scale_jointly=True` → centering off) always crashed with `UnboundLocalError`, and
    any `None` sigma always crashed with `ValueError`. Silent corruption was real but
    confined to multi-object fits on `scale_jointly=False` models; training data was
    never touched because the dataset classes never pass the flag.
  - **Test module basenames must be unique across `testing/`.** The tree has no
    `__init__.py` files, so pytest imports test modules flat: a second
    `test_import_compat.py` in another directory fails collection with "import file
    mismatch". Cost one amended commit in §8.0.C.
  - **`Regress` cannot take an honest-NaN latent row** — sklearn's
    `LinearRegression.fit` refuses NaN design matrices — so #29's "score the failed
    subject NaN" had to mean *skip the regression and report NaN r²*, not feed NaN
    rows through it. Safe only because degeneracy is decoder-level: every subject
    fails together, so there is no partial-alignment case to handle.
  - **#69's "two halves" were one defect, and the fix was net-negative.** The KeyError
    and the missing buffer both came from the in-memory branch reimplementing what the
    disk branch already did. Unifying on the disk branch's semantics (compute the frame,
    let `__getitem__` apply it) deleted the entire mutate-in-place branch — −48 lines —
    instead of repairing it.

---

> Measurements: run `pytest testing/ --cov=NSM` rather than trusting any figure quoted
> below — the numbers in §1.3, §5 and §7 are as-measured-then and are not maintained.

---

**Motivation:** The LR-schedule bug (see §1) was a silent numerical error that ran
undetected for ~3 years and was found by an external collaborator, not by us. It is a
symptom, not an incident. This plan makes that class of bug findable and preventable.

> **Scope.** Code health only: documentation accuracy, structural mapping, test coverage,
> and decomposition of monoliths. Deliberately **out of scope**: new science
> (`NSM_TRAINING_IDEAS.md`, `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md`) and the ICP
> registration-robustness work, which has no plan file in this repo. Those resume *after*
> Phase 2, on firmer ground.

---

## 1. Why — the evidence that motivated this

### 1.1 The triggering bug

`get_optimizer()` built optimizer param groups in the order `[latent, model...]` and
assigned `lr_schedules[1]` to latents, `lr_schedules[0]` to the model — correct. But
`adjust_learning_rate()` then reassigned **by position** every epoch:

```python
for i, param_group in enumerate(optimizer.param_groups):
    param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
```

**Every Adam/AdamW training run from May 2023 to Aug 2026 trained latents under the model
schedule and vice versa — for 100% of the run.** `get_optimizer` does set the intended
rates at construction, but the epoch loop starts at 1 and `adjust_learning_rate` runs at
the top of `train_epoch`, so the correct values are overwritten before the first
`optimizer.step()`.

`schedule_free_*` runs skipped `adjust_learning_rate` and so were never mis-mapped — but
they were arguably hurt worse, since every config was *tuned* against the Adam path and
running one schedule-free applies the values inverted and undecayed. See
`docs/KNOWN_ISSUES.md` §1.

Reported 2026-07-10 by an external collaborator, credited in the ledger.

### 1.2 Why our tooling could not have caught it

- Nothing crashed. No exception, no NaN, no shape error. Only the numbers were wrong.
- The two functions live in different conceptual layers and were never tested together.
- `NSM/utils.py` — the file containing both — has **zero docstrings** across 22 functions
  and 5 classes, so the `[0]=model, [1]=latent` convention existed only in the author's head.
- No test asserted "what learning rate does each param group have at epoch *k*".

Unit tests on extracted helpers would **not** have caught this. Only an assertion on
observable numerical state across an epoch boundary would. That shapes Phase 2.

### 1.3 Measured state of the codebase (2026-08-14)

| Metric | Value |
|---|---|
| `NSM/` source | 11,565 lines |
| Docstring coverage | 48% (122/247 functions, 8/26 classes) |
| Test coverage | 32% (100 tests + 1 skip, 13s runtime) |

Coverage is inverted relative to risk — the newest code is best tested, the core that
everything depends on is barely tested at all:

| Module | Lines | Docstrings | Coverage | Note |
|---|---|---|---|---|
| `datasets/sdf_dataset.py` | 2,195 | 55% | **7%** | largest file in repo |
| `reconstruct/main.py` | 1,443 | 62% | **24%** | production path |
| `train/train_deep_sdf.py` | 618 | **0%** | **10%** | 618 lines in 2 functions |
| `utils.py` | 280 | **0%** | 25% | most-imported module; held the bug |
| `models/modulated_periodic_activations.py` | 252 | **7%** | 67% | |
| `mesh/correspondence_metrics.py` | 699 | 100% | 94% | recent work — the target state |
| `models/loader.py` | 387 | 100% | 84% | recent work |

### 1.4 The production surface is not yet established

NSM is a training library first. `NSM/train/`, `NSM/datasets/sdf_dataset.py`,
`NSM/losses.py` and the mesh path are the product, not internals — so most of the 11.5k
lines are in active use, not incidental.

Known consumers:

| Consumer | Surface used |
|---|---|
| Training (first-party) | `train/`, `datasets/`, `models/`, `losses`, `mesh/` |
| `kneepipeline` inference | `TriplanarDecoder`, `reconstruct_mesh` |
| `nsosim` | mesh interpolation; extent unverified |
| Published models, shared configs | `model_params_config.json` schema, checkpoint format |
| Downstream forks | unknown; they carry their own trainers |

Checkpoint and config **formats** are part of the contract even though nothing imports
them. The LR fix needed a migration path precisely because a config file is a public
interface.

None of the above is enumerated. Until it is, assume any refactor can break something
outside this repo. Enumerating it is the first Phase 0 task (§3), not an input to this plan.

### 1.5 Open issues are all symptoms of the same disease

| # | Title | Class |
|---|---|---|
| 1 | `norm_and_scale_all_meshes` shouldn't work if not loading in memory | dead/incorrect path |
| 2 | SDFSamples — slow loading | performance |
| 3 | Scale of sigma depends on `scale_jointly` | ambiguous API contract |
| 5 | Wandb not needed if just doing inference | unnecessary coupling |
| 6 | pyvista `polydata._faces` syntax update | upstream API rot |

Four of five are exactly what a systematic audit surfaces. Issue #3 is already written up
as `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` and stalled mid-Phase-1.

---

## 2. Guiding principles

1. **The map precedes the documentation.** Never write a docstring for code that is a
   deletion candidate.
2. **Quarantine, don't delete.** Downstream forks and `nsosim` may reach into anything. Moving
   to `NSM/deprecated/` with a `DeprecationWarning` is reversible; `git rm` is a support
   burden when someone's pipeline breaks silently.
3. **Behavioural tests before structural tests.** A golden-output regression harness that
   catches silent numerical drift is worth more than 500 unit tests on pure helpers.
4. **Every bug found gets a provenance entry.** For science code, "which results are
   affected by which bug" is a first-class deliverable (§7).
5. **Silent behaviour changes are forbidden.** Any change that alters training or
   reconstruction numerics must fail loudly on old inputs and offer an explicit path to
   reproduce historical behaviour. Phase A (§4) is the reference implementation.

---

## 3. Phase 0 — Decide what the library is for  ✅ (0a done; 0b open)

**Why first:** Phase 1's map cannot mark anything "deprecated" until these calls are made.
Blocking all later phases.

- [x] Write a one-page scope statement: what NSM is, what it supports, what it does not.
- [x] Rule on each ambiguous module — **supported**, **deprecated**, or **dead**.
      **All five proposed rulings were refuted.** See `docs/SCOPE.md` §2:
  - `train/train_deep_sdf_multi_head.py` (428 lines) — the "broken" half is confirmed
    (`:85` passes the `for model in models` loop variable to `get_optimizer`). The
    "superseded" half is **false**: it trains N *independent* decoders against one shared
    latent embedding, whereas `train_deep_sdf` with `objects_per_decoder > 1` is one
    network with N output channels. Different architecture, not a replacement. Ruling:
    **supported, fix it** — a two-identifier repair
  - `train/deprecated/` — split. `..._multi_surface_orig.py` (562) is a strict subset →
    dead. `..._orig.py` (318) holds the only live `sample_difficulty_lx` implementation →
    port ~12 lines into `train_deep_sdf.py` first
  - `mesh/refine_mesh.py` — zero importers confirmed, but it is the only cross-mesh
    subdivision with ID preservation, and the completed interpolation plan records in
    writing that it was kept deliberately. **Research, keep**
  - `reconstruct/reconstruct_latent_S3.py` — only Sim(3) joint pose+latent fit in the repo,
    and an active item in the ICP plan §5. **Deferred research, scheduled**
  - `reconstruct/cartilage_func.py` — **production**: wired into `DICT_VALIDATION_FUNCS` in
    the live trainer. `predictive_validation_class.py` — research, live caller
  - `configs/generate_sdf_default_config.py` — supported. Confirmed
- [x] Establish the public API contract — `docs/SCOPE.md` §3. Consumer surface is exactly
      two symbols; tiers proposed as 6 public-stable / 48 public-provisional / rest internal.
- [ ] **`__all__` in code — deferred, and the plan text needs amending.** `NSM/__init__.py`
      imports only `utils`, so a top-level `__all__` would either name unbound symbols or
      force eager subpackage imports — which would pull `wandb`, `vtk` and a root-logger
      reconfiguration into every `import NSM` and destroy the one property that makes the
      consumer's import cheap. Put `__all__` per subpackage instead. `docs/SCOPE.md` §3.3.
- [x] Include the checkpoint and `model_params_config.json` formats in that contract —
      `docs/SCOPE.md` §4. Six on-disk contracts, none versioned except the LR `Target` key.
- [ ] **0b — survey downstream consumers before quarantining anything.** `nsosim` is not
      available locally; this remains open and now gates *only* the physical quarantine
      move, not the map. See `docs/SCOPE.md` §5.

**Deliverable:** `docs/SCOPE.md` ✅ + an `__all__` per subpackage (deferred, see above).

---

## 4. Phase A (done) — LR fix as the migration template

Delivered 2026-08-14/15 across PRs **#9**, **#10** and **#11**, all merged. Recorded here
because it is the **pattern** every subsequent behaviour-changing fix should follow.

### What shipped

- **`Target` on every `LearningRateSchedule` entry** (`"model"` / `"latent"`). Entry order
  is ignored. A config missing it — including a half-annotated one — raises, printing a
  paste-ready annotated copy of the caller's own entries.
- **No positional indexing anywhere in the LR path.** `get_learning_rate_schedules`
  returns a `{target: schedule}` dict; param groups carry a matching `target`;
  `adjust_learning_rate` is one lookup. `name` survives as a human label only.
- **Pre-Aug-2026 checkpoints are refused at load time**, not left to a downstream
  `KeyError` — which is skipped for `schedule_free_*` and would have failed hours in.
- **Migration code isolated** in `NSM/_lr_migration.py`, with a delete-when condition in
  its header.
- `docs/KNOWN_ISSUES.md` seeded, with ShapeMedKnee_2024 as a worked example.

### Approaches tried and rejected

Each of these shipped in a first draft and was removed before the work finished. They are
listed because the next behaviour-changing fix will be tempted by the same three.

- **A config flag declaring the entry order** (`lr_schedule_convention: v2 |
  legacy_swapped`). It made the ambiguity explicit rather than removing it, and left the
  ordering itself in place. Replaced by a per-entry `Target`. A flag that describes how to
  read positions is a sign the positions should not carry meaning.
- **Storing param-group names in the checkpoint** (`optimizer_group_names`).
  `state_dict()` already retains custom group keys and `load_state_dict()` restores them,
  so it never did anything. It was carried over from a downstream fork along with a
  docstring justifying it; the justification was tested and proved false, and the code
  survived anyway on a replacement rationale.
- **Restoring names positionally for old checkpoints** (`rename_` /
  `restore_optimizer_param_group_names`). Both guessed identity from group order, which is
  the assumption the fix existed to remove. Pre-Aug-2026 checkpoints are now refused
  outright.

Final size: `NSM/utils.py` grew by 173 lines, down from 341 in the first draft, with no
loss of function. Everything removed came out in review rather than from any automated
check — which is what the `Making Changes` section of `CLAUDE.md` exists to change.

**The generalizable lessons:**

1. Internal correctness guards and user migration guards are different mechanisms, and a
   behaviour-changing fix needs both. A name-based `KeyError` protects the invariant but
   is invisible to someone holding an old config.
2. Fix the *class* of defect, not the reported instance. This bug was positional coupling;
   the first fix removed one of three instances in the same code path.
3. Migration scaffolding needs a lifespan declared at write time, or it becomes permanent
   API by default.

---

## 5. Phase 1 — Map the codebase  ✅ (quarantine step gated on 0b)

**Order note:** this was step 1 in the original sketch. It runs *after* Phase 0 and
*before* documentation, so we never document code we are about to quarantine.

- [x] Build a module dependency graph; identify cycles. `docs/ARCHITECTURE.md` §2.
      Built with `ast` rather than `pydeps` (not installed), so deferred imports are
      distinguished structurally. **Layering is strictly unidirectional** and there is
      **exactly one cycle**, `utils` ↔ `_lr_migration`, which is deliberate and documented
      at both ends. Not a refactor target.
- [x] For every public function/class: caller count, in-repo and in `kneepipeline`.
      Note for anyone repeating this: exclude `build/lib/`, a stale gitignored copy of the
      whole package that double-counts every naive `grep -r`.
- [x] Mark each module — ledger at `docs/ARCHITECTURE.md` §3, with lines, coverage,
      docstring count, *inaccurate*-docstring count, importers and status.
- [x] Flag duplicated logic. **Six traps, not one.** The two `adjust_learning_rate`
      implementations are worse than described: `reconstruct/utils.py`'s copy is leaked
      into `NSM.reconstruct`'s namespace by `from .main import *`, so
      `from NSM.reconstruct import adjust_learning_rate` silently binds the wrong one.
      `docs/ARCHITECTURE.md` §6.
- [ ] Quarantine everything ruled dead → gated on **0b**, and it is now one file.

**Deliverable:** `docs/ARCHITECTURE.md` ✅ + `docs/AUDIT_FINDINGS.md` (216 findings) ✅.
**Checkpoint (re-measured on `main`):** 11,861 lines, 34% coverage, 153 tests + 1 skip.
**Quarantinable: 882 lines, not ~1,800** — and 12 of them must be ported out first. Every
module the plan expected to fall turned out to hold a capability nothing else implements.
Treat the 1,800 as a prediction that was tested and failed, not a target to hit.

---

## 6. Phase 2 — Documentation accuracy pass

Only over modules that survived Phase 1.

- [ ] Docstrings on every surviving public function/class: purpose, args, returns, raises,
      **and any silent convention** (index orderings, coordinate spaces, units).
      `[0]=model, [1]=latent` is precisely the kind of thing that must be written down.
- [ ] Verify each existing docstring against the implementation — 48% coverage says
      nothing about whether those 48% are *true*.
- [ ] Enforce mechanically so it cannot rot: add `flake8-docstrings` (or `pydocstyle`) to
      `make lint`, failing on missing docstrings in public API.
- [ ] Rewrite `CLAUDE.md` §Architecture from the Phase 1 map; drop the stale
      "EIKONAL LOSS HAS NOT BEEN TESTED" shout-comment into a tracked issue instead.

### 6.1 Warnings, not just docstrings

Phase 1 found capabilities that are real but unready. A docstring is the wrong instrument
for those — the user who needs the warning is the one who did not read the docstring. Each
of these is small and independent of the decomposition work:

- [ ] **Rewrite the `train_deep_sdf_multi_head` deprecation text** (`:30`). It currently
      says "Use `NSM.train.train_deep_sdf` with `'objects_per_decoder' > 1` instead" —
      advice that silently hands the user a different architecture (`docs/SCOPE.md` §2.1).
      Say broken-and-unfixed; name no replacement. Keep it out of the documented surface:
      its hyperparameters have never been tuned.
- [ ] **Warn on the Eikonal loss at the point of use.** Never run by its author, never
      executed by a test. See §3 below — the warning's wording depends on whether it works.
- [ ] **Fix, then warn, then document `mesh/refine_mesh.py`,** in that order. It raises
      `UnboundLocalError` on its own defaults (`:399`), so documentation written today
      describes something nobody can run. `docs/SCOPE.md` §2.3 lists the three conditions.
- [ ] **Guard `sample_difficulty_lx` when porting it** out of `train/deprecated/`. Its
      off-state has never been exercised; a feature whose disabled path is untested turns
      itself on eventually. Document it at the config key, not only in code.
- [ ] **Find the config keys that silently do nothing** because their implementing branch is
      commented out. These read as working features and produce no error — the inverse of
      the hazard above, and harder to notice.

**Deliverable:** docstring coverage ≥90% on surviving public API, lint-enforced.

---

## 7. Phase 3 — Test to a known-good baseline

> **§7.1 is DONE** (PRs #13, #14, then `eaee68c`/`94b48f0`). `testing/NSM/regression/`,
> green in CI via `make test`; what it asserts and how to work on it is in that directory's
> `README.md`. Every checkbox in §7.1 below is ticked. The defects it surfaced are in
> `docs/KNOWN_ISSUES.md` § Open. See the **State** block at the top of this file for
> what to do next and for what §7.1 assumed that turned out to be false.
>
> §7.2 has NOT started. Phase 4 is now unblocked.


**The step that stalled before.** It stalls because "test everything" against ~3,000
uncovered statements has no end condition. Bound it by *risk*, not by coverage percentage.

### 7.1 First and most important: end-to-end numerical regression harness  ✅ DONE

Before any unit tests. A tiny synthetic dataset (2–3 analytic meshes, 8 epochs, CPU,
fixed seed), then assert against stored baselines:

- [x] Training: loss trajectory, final latent norms, **per-param-group LR at each epoch**
- [x] Reconstruction: fitted latent, output mesh vertex positions, surface metrics
- [x] Tolerances tight enough to catch a real change, loose enough to survive
      platform float noise. Store baselines as versioned artifacts.
- [x] Must run in CI in <2 minutes. (~20s; whole suite 33s)

**Scope widened 2026-08-15** after Phase 1. The four items above leave the layer with the
worst findings untouched, so add:

- [x] **The dataset cache round-trip.** `sdf_dataset.py` is 2,195 lines at 7% coverage and
      its worst findings are *silent wrong data*: `mesh_to_scale`, `uniform_pts_buffer` and
      `subsample` all change cached content and none is in `get_hash_params`, so a second
      run silently reuses the first's alignment. A harness that builds its data in memory
      never touches that path. Assert: build cache → reload → samples identical; and
      changing a hashed parameter changes the key.
- [x] **The consumer's actual entry point, not just `reconstruct_latent`.** `kneepipeline`
      calls `reconstruct_mesh` with a *list* of paths (multi-object). That function has
      exactly one executed line in the current suite — its `def`. Assert the returned
      `mesh` list **order** too: index 0 = bone, 1 = cartilage is a load-bearing contract
      that nothing in the signature, docstring or result dict names.
- [ ] **Model save/load round-trip.** `testing/NSM/models/test_loader.py:232` loads a saved
      model and never compares its output to the original's, so a wrong-but-same-shaped
      forward passes every assertion. Train → save → load → assert bitwise-identical.
- [ ] **Name the CPU/GPU gap rather than discovering it later.** A <2-minute CI harness is
      CPU; production is CUDA. Add a separate opt-in GPU test asserting the seed-ordering
      constraint `kneepipeline` depends on (`torch.manual_seed` *after* `.cuda()`, per
      `docs/KNOWN_ISSUES.md`), and state in the harness that CPU baselines do not
      bound GPU divergence.

Roughly 30–40% more than the original four items, still one bounded artifact, still under
two minutes.

This single harness is what protects every refactor in Phase 4, and it is the only thing
that would have caught the LR bug.

**It is also how the audit register gets settled.** `docs/AUDIT_FINDINGS.md` holds 216
findings of which **178 are read-only inference, not executed** — and both inferred claims
that have since been tested were wrong, in the direction of overstatement (the triplanar
"affine map" claim, and the eikonal "forward+backward runs" claim). There is deliberately
**no separate verification pass**: findings get confirmed or killed as a by-product of
building these tests, so the effort leaves permanent tests behind instead of throwaway
scripts. The register's job is to tell this section where to point. The 30 landmines that
are both inferred and on the production path are the highest-value targets.

### 7.2 Contract tests on the production API (§1.4)

- [ ] `TriplanarDecoder` — construction from config, forward shapes, checkpoint round-trip
- [ ] `reconstruct_mesh` — end-to-end on a synthetic model, asserted against baseline
- [ ] Checkpoint backward compatibility: load a real historical checkpoint and assert it
      still reconstructs identically. Non-negotiable — this is the promise to `nsosim`.

### 7.3 Characterization tests, written just-in-time

For each monolith, **only immediately before decomposing it in Phase 4**. Pin observed
behaviour, bugs included — a characterization test documents what the code *does*, and
anything surprising becomes a §8 ledger entry rather than a silent fix.

Priority order (by lines × inverse coverage × production-reachability):
1. `datasets/sdf_dataset.py` (2,195 lines @ 7%)
2. `reconstruct/main.py` (1,443 lines @ 24%)
3. `train/train_deep_sdf.py` (618 lines @ 10%)
4. `utils.py` (280 lines @ 25%) — partly done in Phase A

### 7.4 Targets

- [ ] Overall coverage 32% → **≥70%**, with **≥90%** on the production API
- [ ] Suite stays under 2 minutes so nobody skips it
- [ ] CI runs it on every PR

---

## 8. Phase 4 — Decompose the monoliths

Standard extract-under-test: pull a coherent piece out,
have the monolith call the extracted function, keep §7.1 green, then unit-test the
extracted piece properly.

- [ ] `train_deep_sdf.py`: split the 618-line/2-function structure into setup, epoch loop,
      validation, checkpointing.
- [ ] `sdf_dataset.py`: separate mesh loading, normalization/scaling, registration, SDF
      sampling, caching. `NSM/datasets/utils.py` is a 2-line stub whose only content is a
      TODO to do exactly this — it has been waiting.
- [ ] `reconstruct/main.py`: separate latent optimization, mesh generation, evaluation.
- [ ] Fold in the stalled API-cleanup plans, which are Phase-4-shaped and should not run
      separately: `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` +
      `.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` (issue #3).
- [ ] Close issues #1, #2, #5, #6 as the relevant modules are touched.

### 8.0 `sdf_dataset.py` decomposition — plan statement (2026-08-22)

The file is 2,644 lines: ~900 of module-level functions, ~1,700 of the two Dataset
classes. Slice A below moves the functions out **verbatim**; restructuring anything
comes later, each slice with its own statement.

**Slice A — target layout (all permanent, nothing transitional):**

- `NSM/datasets/utils.py` — the 13 leaf helpers, no NSM-internal imports:
  `derive_seed`, `mesh_content_key`, `get_rand_uniform_pts`,
  `get_pts_center_and_scale`, `is_zipfile`, `meshfix`, `get_cube_mins_maxs`,
  `get_buffered_cube_mins_maxs`, `unpack_pts`, `unpack_numpy_data`,
  `check_probabilities`, `check_probabilities_sum`, `combine_meshes` (~330 lines).
- `NSM/datasets/mesh_sampling.py` — the two subject-level pipelines,
  `read_mesh_get_sampled_pts` and `read_meshes_get_sampled_pts` (~580 lines);
  imports only from `.utils`.
- `NSM/datasets/sdf_dataset.py` — the two Dataset classes plus a **permanent
  re-import block**: `NSM.datasets` *and* `NSM.datasets.sdf_dataset` are both live
  import paths (`reconstruct/main.py` uses both in adjacent lines; downstream forks
  assumed to as well). The re-import is public API, not a shim — no delete-when.
  The file's top import block stays verbatim: flake8 ignores F401 globally, and the
  unused imports (`pcu`, `numpy_to_vtk`, …) leak into `NSM.datasets` via the
  star-import (ARCHITECTURE.md §5), so removing them would change that namespace.
  `__all__` stays out of this slice — it is the deferred Phase 0 item and a separate
  namespace decision.

**Size budget:** moves are verbatim; allowed growth is two module headers + import
blocks + the re-import block, **~70 lines net**. Beyond that is scope creep.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization tests against the *current* file (§7.3);
3. the move, with the ARCHITECTURE.md §3 ledger rows corrected in the same commit;
4. State update.

**Characterization targets (commit 2)** — what moves and is unpinned today:
`unpack_numpy_data` key spellings and raises; `is_zipfile` unreadable-path fallback;
the probability validators; cube-helper `ValueError`s; reader branches — missing
path → `None`, `get_random=False` (the `"pts"`-not-`"xyz"` spelling), the no-norm
branch (`scale=1`, `center=0`), register-without-mean raise (single raises bare
`Exception`, multi raises `ValueError` — pinned as-is), the deprecated-kwarg print
contract, `include_surf_in_pts` (single asserts correct behaviour; multi asserts
the surface's own points and is `xfail(strict=True)` for #17), and the
`mesh_to_scale` / `scale_all_meshes` / `center_all_meshes` frame math on analytic
spheres.

**Verification per claim:**

| Claim | Verification |
|---|---|
| Verbatim move changes no behaviour | full suite + regression harness green before/after; `git diff --color-moved` shows pure moves |
| Both import paths keep every symbol | new test importing each moved name from both `NSM.datasets` and `NSM.datasets.sdf_dataset`, list frozen in the test |
| No import cycle introduced | `utils.py` imports nothing NSM-internal, `mesh_sampling.py` only `.utils`; ARCHITECTURE.md §2 ast pass re-run |
| Cache keys unaffected | existing `TestHashedParametersChangeTheKey` / `TestCacheRoundTrip` green |
| Pool pickling unaffected | classes stay in `sdf_dataset`; existing `test_multiprocessing_does_not_change_the_data` green |

**Deferred out of slice A, deliberately:** splitting the reader internals
(registration / frame computation / per-surface draws) is slice B — that is where
#69 (`norm_and_scale_all_meshes` in-memory) and #3 (sigma coordinate space) ride
along, and it gets its own statement first. Class-side cache/build decomposition
stays grouped with #19/#27 (see State § Deliberately deferred).

### 8.0.B `read_meshes_get_sampled_pts` internals split — plan statement (2026-08-23)

Slice A moved the readers verbatim; this slice restructures the multi reader's
inside. The single reader stays untouched — it is 80 flat, legible lines, and its
`include_surf_in_pts` block is the *correct* one (#17's trap: any "unification"
risks copying the broken variant over it).

**Target shape (all permanent, nothing transitional).** Three module-private
helpers inside `mesh_sampling.py`, named for the State's own decomposition —
private because they are internals, and so that the `test_import_compat` frozen
namespace list does not change:

- `_register_to_mean(orig_meshes, new_meshes, new_pts, paths, mesh_to_scale,
  mean_mesh, icp_transform)` → the transform used (caller-supplied or computed via
  `combine_meshes` for a list `mesh_to_scale`); applies it to every surface in place.
- `_compute_shared_frame(new_pts, mesh_to_scale, scale_all_meshes,
  center_all_meshes, scale_method)` → `(center, scale)`. Pure — the six-branch
  tangle becomes directly testable against the sphere arithmetic the slice-A
  characterization already asserts.
- `_draw_surface_samples(new_meshes, new_pts, sigma, n_pts, rand_function,
  include_surf_in_pts, uniform_pts_buffer, seed)` → `(rand_pts, pts_surface)`.
  The per-surface SDF loop and the `get_random=False` branch stay inline — coherent
  as they are, and the State does not name them.

**Fixes land *before* the split**, so the split commit is purely
behaviour-preserving (no xfail transitions inside it):

- **#17** — the leaked `new_pts_` in the draw loop. The issue's "determine which of
  the three applies" question is now settled by execution (2026-08-23, sphere pair):
  with `center_pts`/`norm_pts` both False and numeric sigmas — the shape
  `reconstruct_mesh` produces for `scale_jointly=True` models — it **always crashed**
  (`UnboundLocalError`); with any sigma `None` it **always crashed** (`ValueError`,
  `new_pts_` is a leaked list); with centering on and numeric sigmas it produced
  **silently wrong data** — 1580 points where 1874 is correct: the *last* surface's
  pre-normalization vertices appended once per surface, wrong surface *and* wrong
  frame. So the `KNOWN_ISSUES.md` § History entry covers exactly one run class:
  multi-surface calls with centering on — via `reconstruct_mesh`, that is
  `get_rand_pts=True` on a `scale_jointly=False` model. Neither shipped config
  reaches it (`get_rand_pts_recon: false`). Fix: append `new_pts[new_pts_idx]`; the
  slice-A strict-xfail passes unmarked. Extraction then removes the *class* of
  defect — a helper's scope cannot see a leaked binding from another section.
- **#69** — fix by unification, not by patching the in-memory branch:
  `norm_and_scale_all_meshes` computes `self.center`/`self.max_radius` in both
  storage modes (the only difference is reading `new_pts_{i}` npz keys vs the
  in-memory `new_pts` list) and the existing per-batch application in `__getitem__`
  — present in both classes (`SDFSamples.__getitem__`,
  `MultiSurfaceSDFSamples.__getitem__`) and conditioned only on the attributes —
  does the scaling. The mutate-in-place branch is deleted; the buffer is applied by
  construction. Disk-path numerics unchanged. Always crashed → no History entry.

**#3 rides along structurally, not as a fix.** After the split there is exactly one
site where sigma is consumed (`_draw_surface_samples`); its docstring states the
coordinate-space fact — draws happen in whatever frame the meshes are in at call
time: normalized when centering ran, original units otherwise. The breaking change
itself stays with `BREAKING_CHANGE_PROPOSAL.md` / `SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md`
and needs its §4-style migration guard; nothing in this slice moves it.

**Re-verified during scoping, already tracked:** `reconstruct_mesh` passes
`n_pts_random=` to readers whose parameter is `n_pts=`, so the kwarg is swallowed
and the 200,000-point default is used (asked 7, got 200,122 = 200,000 + vertices).
That is #16; its fix belongs to `reconstruct/main.py`, not this slice.

**Characterization added before any change** (commit 2), pinning what the split
touches and slice A left unpinned: the `icp_transform` reuse path (the dataset's
cross-combo contract — pass a transform back in, registration is skipped, the same
frame comes out), the joint uniform-cube draw (`None` in sigma, without
`include_surf_in_pts`), reader-level seed determinism (same seed → identical
draws; different seed → different), and a second #17 strict-xfail for the
uniform-cube + `include_surf_in_pts` combination.

**Size budget:** #17 ≤ +5 code lines; #69 net ≤ +10 (unification should land ≤ 0);
the split ≤ +80 net in `mesh_sampling.py` (three signatures + docstrings; bodies
are moves). Characterization tests are additive and outside the budget. Beyond
this is scope creep.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization additions; 3. #17 fix + History entry +
xfail unmarks; 4. #69 fix + xfail unmark + `KNOWN_ISSUES` § Open removal;
5. the split; 6. State update.

**Verification per claim:**

| Claim | Verification |
|---|---|
| #17 fix appends each surface's own points | slice-A strict-xfail passes unmarked; new uniform-cube xfail passes unmarked |
| #17's affected-run classes are as stated | the executed determination above, recorded in the History entry |
| #69 both halves land together | `TestScaleJointlyInMemory` passes unmarked (asserts the buffered domain; its `raises=KeyError` made a half-fix a plain failure) |
| #69 leaves disk-path numerics unchanged | regression harness green (`test_dataset_cache`, training regression) |
| Split changes no behaviour | full suite + harness green before/after; `git diff --color-moved`; no xfail transitions in the split commit |
| No namespace change | `test_import_compat` frozen list untouched |
| No new import or dependency edge | helpers live in `mesh_sampling.py`; module still imports only from `.utils` |

### 8.0.C `reconstruct/main.py` — fixes and first decomposition slice — plan statement (2026-08-23)

The file is 1,509 lines; the three §8 concerns (latent optimization, mesh
generation, evaluation) live in one module. This slice lands the three issues
filed against the file (#15, #16, #29) plus the in-package instances of #16's
class, then moves the latent-optimization stack and the wandb helpers out
verbatim. **Fixes land before the split**, 8.0.B's pattern: the split commit is
purely behaviour-preserving.

**Target shape (permanent unless marked):**

- `NSM/reconstruct/latent_fit.py` — the latent-optimization stack, moved
  verbatim: the four `reconstruct_latent_*` type-check/setup helpers,
  `reconstruct_latent_preprocess_sdf_gt`, `project_latent`,
  `latent_norm_penalty`, `reconstruct_latent` (~690 lines). Imports torch/wandb/
  `NSM.losses`/`.utils` only — leafward, no cycle. The log *format* has no
  `%(name)s`, so the logger renaming to `NSM.reconstruct.latent_fit` changes no
  output byte.
- `NSM/reconstruct/wandb_logging.py` — `_process_meshes_for_wandb`,
  `prepare_results_for_wandb` (~145 lines). Separate module because
  `reconstruct_mesh` calls it: parking it in an evaluation module would make
  `main` import evaluation and evaluation import `main`.
- `NSM/reconstruct/main.py` — keeps `reconstruct_mesh`, `tune_reconstruction`,
  `get_mean_errors`, `compute_correlation_coefficient`, the module-scope
  `logging.basicConfig` (verbatim — its removal is #58's business), and a
  **permanent re-import block**: `NSM.reconstruct` (star-import namespace) *and*
  `NSM.reconstruct.main` are both live import paths
  (`test_predictive_validation.py` monkeypatches `NSM.reconstruct.main`;
  kneepipeline and the trainers use the package path; forks assumed to use
  both). Every name currently importable from either path stays importable from
  both, `_process_meshes_for_wandb` included.

**Deferred out of this slice, deliberately:** moving the evaluation trio
(`get_mean_errors`, `tune_reconstruction`, `compute_correlation_coefficient`)
to its own module. It buys nothing now — `main` still imports wandb for
`reconstruct_mesh`'s own logging, so #5 is not advanced — and costs either an
import cycle (`main` ⇄ evaluation) or breaking `NSM.reconstruct.main`'s
namespace. It becomes worth doing as part of #5 (making wandb optional), which
needs its own statement. Also out: #58 (ungated prints), #56 (the
`sigma_rand_pts` default is 0.001 in `reconstruct_mesh` and 0.01 in
`get_mean_errors` — that divergence is #56's class), `reconstruct_latent_S3.py`
(deferred research, #35).

**The fixes, with the decisions of record:**

- **#15 — unify the readers on `"pts"`.** `read_mesh_get_sampled_pts` writes its
  draw to `results["xyz"]`; everything first-party reads `"pts"`
  (`reconstruct_mesh`, `reconstruct_latent_S3`, and both dataset classes via a
  two-branch probe). Fix in `mesh_sampling.py`: the single reader writes the
  draw to `"pts"` on every path, keeping `"xyz"` as an alias **bound to the same
  array** — *transitional*, delete when the 0b fork survey confirms no external
  `["xyz"]` readers, or at v0.3.0, whichever first. The two probe workarounds in
  `sdf_dataset.py` (`:608`, `:1331`) collapse to `result_["pts"]` — removing the
  existing workarounds, not adding a third. Single-object `get_rand_pts=True`
  always crashed → **no History entry**; cached content is unchanged (same
  array, new key) → cache tests prove it.
- **#16 — honour `n_pts_random`.** `reconstruct_mesh` forwards
  `n_pts_random=` to readers whose parameter is `n_pts=`; both readers'
  `**kwargs` swallow it and the 200,000-point default runs. Fix: pass
  `n_pts=n_pts_random` (the multi path already listifies it). This is the
  accepted-and-ignored shape the memory says to *delete* — but the two deleted
  precedents (`padding`, `center`/`scale`) would have changed every default run
  if honoured, and honouring this one is measured to touch only
  multi-object `get_rand_pts=True` runs (never shipped; single always crashed,
  #15), while the parameter's intent is unambiguous and config-plumbed
  (`n_pts_random_recon` → three layers). Honouring changes those runs' draw
  from 200,000/surface to the requested value → **History entry**. Ride-along:
  the regression-harness class docstring that documents the swallowed argument
  (and the measured 400,688) is rewritten, and the seeded tests get ~1000×
  smaller draws, which is the harness cost §7.1 blamed on this bug.
- **#16's class, enumerated** (AST scan, every function, parameter name absent
  from body — run 2026-08-23, both repos' consumers checked): in-package,
  `compute_recon_loss(n_samples_assd)` — its implementing call is commented
  out, no caller passes it → **deleted**, and the `batch_size_latent_recon`
  plumbing — `get_mean_errors` forwards it into `reconstruct_mesh`'s `**kwargs`,
  so **every validation pass prints the deprecation warning at itself** →
  parameter deleted from `get_mean_errors`/`tune_reconstruction` and the
  `train_deep_sdf.py:238` call site; the shim in `reconstruct_mesh` stays (it is
  the migration surface for external callers). Out of package, recorded not
  fixed: `models/` instances are #20's annotated parameter half;
  `train_epoch(return_loss, verbose)` ×4 belongs to the trainer slice;
  `interpolate.update_positions(verbose)` and S3's `epsilon` noted for their
  modules' passes; `get_learning_rate(epoch)` on Constant and `__exit__(args)`
  are interface conformity, not defects.
- **#29 — raise by name instead of the fake-success dict.** New
  `NoZeroLevelSetError(RuntimeError)` in `main.py`, raised where the mean mesh
  comes back `None`; message names the state (zero-latent SDF has no sign
  change at `n_pts_per_axis_mean_mesh` resolution) and the two causes (model
  not trained far enough; grid too coarse). `get_mean_errors` catches it
  per-path and records what it always recorded — nan metrics — plus a **NaN
  latent row** to `Regress` (today it hands over the zero vector, so the r²
  is computed against fabricated latents; NaN rows make it an honest nan and
  keep path↔latent alignment). Training validation therefore still survives a
  model that has not learned a sign change; the direct caller (kneepipeline)
  gets a named error instead of a `KeyError` at `result["center"]`. Pre-fix
  runs recorded zero latents and nan metrics as if fitted → **History entry**
  (detection: latent exactly all-zero *and* metrics nan).
  `TestDecoderWithNoZeroLevelSet` is rewritten around the raise; its strict
  xfail retires with this commit. House precedent: #48's barrier, #41.

**Size budget:** #15 ≤ +5 in `mesh_sampling.py`, net negative in
`sdf_dataset.py`; #16 is a 2-line call change; the class sweep is net negative;
#29 ≤ +45 (exception, raise, catch-and-record); the split ≤ +80 net (module
headers, import blocks, re-import block). Characterization tests are additive
and outside the budget.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization — freeze both namespaces
(`NSM.reconstruct`, `NSM.reconstruct.main`) as a list-pinned import test, pin
the unpinned movers (`prepare_results_for_wandb`'s filtering,
`project_latent`, `latent_norm_penalty`'s quadratic/huber branches, the
type-check raises, `reconstruct_latent_get_lr_update_freq` arithmetic), pin the
self-inflicted deprecation print (capsys), add strict-xfails for #15 (single
sampled path returns) and #16 (the request is honoured); 3. #15 fix + xfail
unmark; 4. #16 fix + History + harness-doc rewrite; 5. the in-package class
sweep; 6. #29 fix + History + test rewrite (namespace list gains the exception
— deliberate, in the same commit); 7. the split + ARCHITECTURE.md §3 ledger
rows; 8. State update.

**Verification per claim:**

| Claim | Verification |
|---|---|
| #15 unification changes no cached data | `test_dataset_cache` green untouched; alias asserted `result["pts"] is result["xyz"]` |
| #15 single sampled path works | the strict-xfail passes unmarked |
| #16 honoured | strict-xfail passes unmarked: request 200, receive 200 + appended surface points, not 200,000 |
| #16 touches only never-shipped runs | both shipped configs `get_rand_pts_recon: false` (re-checked); single path always crashed (#15's evidence) |
| deprecation print was self-inflicted | capsys pin in commit 2, deleted in commit 5 when the print stops |
| #29 keeps validation alive | new test: `get_mean_errors` over a no-zero-level-set decoder returns nan metrics and nan r², raises nothing |
| split changes no behaviour | full suite + harness green; `git diff --color-moved` pure moves; no xfail transitions in the split commit |
| both import paths keep every name | frozen namespace lists, asserted on both paths, written in commit 2 *before* anything moves |
| no new import cycle | `latent_fit`/`wandb_logging` import nothing from `.main`; ARCHITECTURE.md §2 ast pass re-run |

### 8.0.D `train/train_deep_sdf.py` — the five filed issues and the orchestrator split — plan statement (2026-08-23)

The file is 649 lines in two functions: `train_deep_sdf` (~223, the orchestrator) and
`train_epoch` (~373, the epoch loop). The five issues filed against it (#28, #42, #49,
#52, #59) land first, then the trainer-side instances of #16's class, then the
orchestrator's concerns split into module-private helpers. **Fixes land before the
split** (8.0.B's pattern): the split commit is purely behaviour-preserving.

**Target shape (all permanent, nothing transitional).** No new modules — module-private
helpers inside `train_deep_sdf.py`, 8.0.B's pattern, so the `NSM.train` namespace does
not change:

- `_resume_from_checkpoint(config, model, latent_vecs, optimizer)` — the resume block
  (both checkpoint loads plus the two refusal guards).
- `_schedule_free_eval_warmup(model, latent_vecs, data_loader, config, epoch)` — the
  eval-mode warm-up. #42's fix site; extraction gives its test a callable seam.
- `_run_validation(config, model)` — the `get_mean_errors` kwarg block.
- `_save_checkpoint(config, epoch, model, latent_vecs, optimizer, sdf_dataset)` — the
  three save calls.
- `train_deep_sdf` keeps setup (defaults, validation, wandb init, dataloader, latents,
  optimizer) and the epoch loop, becoming a legible orchestrator. **`train_epoch` stays
  whole** — it *is* the epoch loop §8 names; its internal loss-pipeline decomposition
  would need its own statement and is not this slice.

**The fixes, with the decisions of record:**

- **#42 — the schedule_free warm-up forwards a raw dataloader item.** `model(batch)`
  where `batch` is `(sdf_data, indices)`; every schedule_free run dies at its first
  checkpoint or validation epoch (verified 2026-08-21 with real schedulefree 1.4.1).
  Fix: unpack the way `train_epoch` does — xyz reshaped, indices repeated per sample,
  latents looked up (variational sampling included), chunked by `batch_split` (the
  knob exists because full batches do not fit; a warm-up that skips it OOMs on exactly
  the configs that need it), `model(inputs, epoch=epoch)` under the existing
  `no_grad`. Always crashed → **no History entry**. Test strategy: `schedulefree` is
  not installed in `nsm-dev`, and the crash is in the trainer's forward, not in
  schedulefree — the pin monkeypatches `NSM.utils.schedulefree` with a stub
  `AdamWScheduleFree` (AdamW plus `train()`/`eval()` no-ops) and trains through a
  checkpoint epoch. The stub exercises the trainer's code path, not schedulefree's
  numerics — KNOWN_ISSUES §1 already records those need retuning.
- **#49 — `resume_epoch == 1` silently skips epoch 1 without resuming.** The resume
  guard reads `> 1` while the loop starts at `resume_epoch + 1`. Fix: one boundary —
  the resume block loads for `resume_epoch >= 1`, so `1` means "epoch 1's checkpoint
  is complete; continue at 2" (the issue's load option, not the raise option:
  `resume_epoch` uniformly names the last completed epoch). `resume_epoch=0`
  unchanged. Pre-fix `resume_epoch=1` runs completed, training from scratch for
  `n_epochs − 1` epochs with nothing loaded → **History entry** (detection: the run
  was launched with `resume_epoch=1`).
- **#59 — logged latent-norm stats are the last batch's; the positional back door.**
  `step_mean_vec_length`/`step_std_vec_length` use `=` where every surrounding
  accumulator uses `+=`, then divide by `len(data_loader)` — logged values wrong by
  ~×n_batches on every run since inception (issue's verification: true mean 0.0107,
  logged 0.0053, ×n_batches matches). Fix: `+=`. Gradients and weights untouched;
  wandb-only → **History entry** scoped to logged metrics.
  `add_plain_lr_to_config` loses `idx_model`/`idx_latent` — the positional override
  the Aug-2026 fix exists to forbid, whose only caller is a test asserting
  deliberately swapped labels; parameters and test deleted together. CHANGELOG
  Breaking (the parameters).
- **#16's class, trainer instances.** `train_epoch(return_loss, verbose)` — both
  AST-confirmed unread (the body reads `config["verbose"]`; `log_dict` is returned
  unconditionally). Deleted from the live trainer's signature and its one call site.
  multi_head's and the deprecated trainers' copies **stay**: multi_head belongs to
  #51's repair, `deprecated/` to the 0b quarantine — deleting parameters from a
  module ruled dead is churn. Also dead: the `if "resume_epoch" not in config`
  re-default at `:86` (the `setdefault` at `:57` precedes it). CHANGELOG Breaking
  (`train_epoch` signature).
- **#28 — `train_deep_sdf` returns nothing.** Returns the per-epoch history: one
  entry per epoch, `{**log_dict` (the exact wandb payload, validation metrics
  included on val epochs)`, "epoch", "lrs": {group_name: lr}, "targets":
  {group_name: target}, "latent_norms": [per-subject]}`. LRs read *after*
  `train_epoch` (the rates the epoch actually ran with), keyed by group `name` with
  the name→target map alongside — several groups can share a target, so target alone
  cannot key the dict. The wandb payload stays byte-identical (extras go on the
  history entry, not into `log_dict`). The harness's `recording_train_epoch` wrapper
  is **deleted**; `run_training` maps the return value into the record shape the
  baselines already pin, so the strict-xfail passes unmarked and the regression suite
  becomes a consumer of the public contract. CHANGELOG (non-breaking: `None` → list).
  Ordered after #59 so the history's `mean_vec_length` is honest from birth.
- **#52 — `mesh_names` is persisted as ground truth and can be silently wrong.** The
  names move to where the ordering is defined: `MultiSurfaceSDFSamples(...,
  mesh_names=None)`, validated at construction against the per-subject surface count
  (`self.n_meshes`), stored on the instance. The trainer, at its existing
  `mesh_names` validation site — entry, not save time: the same check, hours
  earlier, and `save_model_params` then writes what was checked — **adopts** the
  dataset's names when the config has none and **raises** when both exist and
  disagree. Deliberately *not* in the cache key: names do not change sampled data,
  so unlike #19's omissions this one is correct. Single-surface `SDFSamples` is
  skipped: one surface has no ordering to mis-declare. Config-only names with a
  dataset that carries none keep today's behaviour — identity has to come from the
  user; the fix moves the declaration next to the ordering rather than inventing
  one. Root CLAUDE.md § Multi-Surface Config gains the dataset-side recommendation.
  CHANGELOG Added.

**Deferred out of this slice, deliberately:** `train_epoch`'s internal decomposition
(needs its own statement); everything multi_head (#51) and `train/deprecated/` (0b
quarantine, including the §6.1 `sample_difficulty_lx` port); wandb-optional (#5 —
`train_deep_sdf` still imports and calls wandb, and `log_latent` still builds
`wandb.Histogram`s); the two documented-not-fixed Open entries (`enforce_minmax`
prediction-clamp — SCOPE §2.2 config work; `grad_clip` latent-clipping experiment).

**Size budget:** #42 ≤ +15; #49 ≤ +5; #59 net negative in `NSM/`; the sweep net
negative; #28 ≤ +25 in the trainer and net negative across the repo (the harness
wrapper goes); #52 ≤ +45 across both files; the split ≤ +80 net (four signatures +
docstrings; bodies are moves — and the file's two public functions gain its first
docstrings). Characterization tests are additive and outside the budget.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization — freeze the `NSM.train.train_deep_sdf` public
namespace (frozen-list import test), pin `resume_epoch=0` and `>1` behaviour, add
strict-xfails for #42 (a stubbed schedule_free run survives its first checkpoint
epoch), #49 (`resume_epoch=1` resumes from epoch 1's checkpoint), #59 (logged
`mean_vec_length` matches the true epoch mean), #52 (dataset-carried names are
adopted/validated) — #28 is already pinned
(`TestTrainerContract::test_train_deep_sdf_returns_its_history`); 3. #42 fix +
unmark; 4. #49 fix + History + unmark; 5. #59 fix + History + CHANGELOG + the
override-test deletion; 6. the trainer sweep + CHANGELOG; 7. #28 fix + harness
wrapper deletion + unmark + CHANGELOG; 8. #52 fix + CLAUDE.md + CHANGELOG + unmark;
9. the split; 10. State update.

**Verification per claim:**

| Claim | Verification |
|---|---|
| #42's crash is the trainer's, not schedulefree's | the strict-xfail reproduces the `TypeError` with the stub optimizer — no schedulefree import involved |
| #42 fixed means the warm-up runs the real forward | the test trains a stubbed schedule_free run through a checkpoint epoch to completion |
| #49's two boundaries agree | resume test: train, resume with `resume_epoch=1`, assert the epoch-1 checkpoint is loaded (latents match its file) and epochs [2..n] run |
| #59's `+=` restores the true mean | the logged value is asserted against a mean computed over `latent_vecs` directly — the issue's ×n_batches check, as a test |
| the sweep deletes only unread parameters | the AST scan (parameter name absent from body) re-run in-commit; full suite green |
| #28's history is what wandb would have seen | harness baselines (loss trajectory, per-group LR, latent norms) stay green with the wrapper deleted — the same numbers now flow through the return value |
| #52 catches a wrong declaration | dataset names ≠ config names raises at entry; dataset names + no config names adopts |
| split changes no behaviour | full suite + harness green; `git diff --color-moved` pure moves; no xfail transitions in the split commit |
| namespace unchanged | the frozen-list import test written in commit 2, untouched by the split |

### 8.1 Make the library plural — added 2026-08-15

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

## 9. Deliverable: the bug provenance ledger

New file `docs/KNOWN_ISSUES.md`. For science code this is a first-class artifact —
it answers "which of my results are affected?", which a code comment cannot.

Each entry: what was wrong, exact date range affected, which configs/optimizers/code paths,
observable consequence, how to detect it in an existing run, how to reproduce old behaviour.

- [ ] Seed with the LR-schedule bug (May 2023 → Jul 2026, Adam/AdamW only, `schedule_free_*`
      unaffected). Prior to this it existed only as a docstring in a downstream fork — it
      must live somewhere durable and citable.
- [ ] Add the sigma coordinate-space ambiguity (issue #3).
- [ ] Add every subsequent finding from Phases 1–4.
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

- [ ] **Tag `v0.1.0` — "the state before the refactor."** Have `kneepipeline` and `nsosim`
      pin it. This is the rollback point and it decouples their release cadence from this
      work.
- [ ] **Not `1.0.0`.** That is a stability promise, and Phase 1 found 71 landmines with 30
      of them unverified on the production path. Claiming 1.0 and then breaking things in
      Phase 4 makes the number meaningless. `0.x` is honest and gives the same rollback
      guarantee.
- [ ] **Bump on release, not on commit.** Under `0.x`: breaking changes bump the minor
      (`0.2.0` after Phase 4), additive changes bump the patch.
- [ ] **Move to `1.0.0` when there is something to promise** — when `__all__` exists (§3)
      and the §7.1 harness is green. That ties the version to a milestone rather than a date.
- [ ] **Derive the version from git tags.** `pyproject.toml` already lists `setuptools-scm`
      in `build-requires` with `[tool.setuptools_scm]` commented out. Uncommenting it makes
      the tag the single source of truth. A hand-edited literal is exactly why the version
      sat at `0.0.1` for years, and re-deciding the scheme without fixing that mechanism
      leaves it free to go stale again.

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
| `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` | Phase 1 partial | Fold into Phase 4 |
| `.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` | Not started | Fold into Phase 4 |
| `.claude/plans/HYBRID_OPTIMIZER_REPORT.md` | Findings, Aug 2025 | Reference for `reconstruct/main.py` |
| `.claude/plans/NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` | Proposed | Blocked on stable interpolation API |
| `.claude/plans/NSM_TRAINING_IDEAS.md` | Open master list | Idea 3 (test Eikonal loss) belongs in Phase 3 |
| `.claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md` | Complete 2026-05-22 | Target-state example |
| `docs/MULTI_SURFACE_REGISTRATION.md` | Current | Feature doc, verify in Phase 2 |
