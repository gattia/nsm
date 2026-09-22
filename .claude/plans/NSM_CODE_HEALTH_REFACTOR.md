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

**Updated:** 2026-09-21 · **Status:** open

- **Next:** step 0 below — restart the celery worker, then dispose of PRs #108 and #109.
  No code. Then slice **P**.
- **Blocked on:** nothing. Two maintainer decisions are open (§ Decisions) and neither
  blocks step 0 or slice P.
- **Done:** Phases 0–3 complete. Eighteen slices executed, A through R, each with its own
  PR. v0.2.0 (PR #36) and v0.3.0 (§8.0.O) shipped. Both post-v0.3.0 validation runs
  passed — §7.5a on the production box (PR #102), §7.5b on the maintainer's cluster
  (PR #105) — so nothing gates the remaining slices. **§8.0.R merged 2026-09-21 as
  `d9ae062` (PR #107):** suite 1180 → 1213 passed, net +61 lines in `NSM/`. Per-slice
  detail is in the history file.
- **Surprises:**
  - **A slice can close a finding in the same breath as deferring it.** All five
    `reconstruct_latent` sites §8.0.K deferred to R were already fixed by §8.0.K's own
    review round and §8.0.J's kwargs refusal. The row recorded the deferral and lost the
    fix, so R was scheduled to re-fix a "200× step size" that had already moved. **Build a
    slice row by sweeping what names it, and re-run it before scheduling it.**
  - **The "accepted and never read" class is empty on the function surface.** Every
    candidate was a polymorphic hook whose *sibling* implementation reads the parameter, so
    #20's delete-the-parameter remedy needs judging across every implementation. Every live
    instance is one frame up, at the config layer, in `models/loader.py`. That finding is
    what started `NSM_CONFIG_SECTIONS_AND_MODEL_REGISTRY.md`.
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
- [ ] **Restart the celery worker (PID 653457).** The website computes each job's `nsm`
      version stamp with `git describe` but caches it with `@lru_cache` for the life of the
      worker process (`backend/services/version.py`). The worker has been up since
      2026-09-01, so all twenty of those manifests record `nsm: v0.3.0-46-gcd83ccc` while
      the tree was at `ee396a2`. **The plan's own "merging is not deploying; the pull is
      the deploy" note is true for the code and false for the record.** Restarting is safe
      now: the worker is idle, holding only its 102 MiB bare CUDA context with no step
      children. The durable fix belongs in the website repo — drop the cache, or key it on
      the tree's HEAD — and is worth an issue there.
- [ ] **Validate §8.0.R against production, once.** R shipped without a §7.5a-style check,
      and those twenty jobs were its only production exposure. Re-run one archived job on
      `main` and compare both BScore variants against the archived values; the measured
      run-to-run band on that box is ~1e-04. This is cheap and it is the last unvalidated
      slice.
- [ ] **Commit the `CLAUDE.md` attribution rule**, uncommitted on this tree since
      2026-09-17.
- [ ] **Merge PR #109** (`idea-13-conv-activation-results`). Docs only, green, independent.
- [ ] **Close PR #108** (`drop-implicit-two-stage`) **without merging**, and fold its second
      commit into slice P. Its first commit `9d2224a` deletes
      `NSM/models/modulated_periodic_activations.py` entire, which the maintainer ruled on
      2026-09-04 **stays** as the ShapeMed-Knee paper's MPA baseline. The PR was last
      updated the day before that ruling and still carries it, so merging as-is deletes a
      published benchmark model. Its second commit `3d0775e` (delete `two_stage`, −414
      lines) is approved and survives. Closing rather than rebasing also avoids a conflict
      with #107 in `loader.py` and `SCOPE.md`.
- [ ] **Delete `.claude/handoff/`.** Its three files are superseded by this plan and by the
      posted #107 description. The directory is gitignored, so nothing is lost. `CLAUDE.md`
      names handoff files as something this repo does not keep.

### Step P — slice §8.0.P: quarantine and delete

Ungated 2026-08-30: the 0b consumer survey is answered and measured, and `nsosim` is
inference-only with a five-symbol NSM surface, none of it in `train/` (`docs/SCOPE.md` §5).

- Delete `NSM/train/deprecated/` — 876 lines, two files.
- **Rule on #18 first, because it decides the shape of that deletion.**
  `train_deep_sdf_orig.py` holds the only live `sample_difficulty_lx` implementation, ~12
  lines. Port it into `train_deep_sdf.py`, or delete it and close #18 as won't-fix. Either
  way §8.0.N's finding is the first task: the four `sample_difficulty_lx` config keys are
  dead on every supported path today (`docs/KNOWN_ISSUES.md` § Open), so nothing regresses
  by deleting them with it.
- Cherry-pick `3d0775e` from the closed #108 — delete the `two_stage` model type. **Strip
  its `Co-Authored-By: Claude Fable 5` trailer.** It needs reconciling with #107, which
  added two-stage tests that the deletion makes dead: remove
  `TestTwoStageTranslatesWhatItsSiblingsRead` and the two-stage evidence test from
  `testing/NSM/test_parameter_surface.py`, and note in `docs/KNOWN_ISSUES.md` § History 30
  that the model type it describes was removed in the same release.
- **Correct #108's `SCOPE.md` §1 line as you carry it.** It says `load_model` advertises two
  model types and one of them, `deepsdf`, cannot be reconstructed. That was measured false
  on 2026-09-04: since #105's `_decode` signature dispatch, both `deep_sdf.Decoder` and
  `ImplicitDecoder` reconstruct end to end.
- #51 rides here: `docs/SCOPE.md` §2.1 already carries the downgrade to
  unsupported-until-needed, and the issue stays open as the repair checklist. Confirm, do
  not redo.

Net effect is roughly 1,300 lines deleted and nothing added.

### Step S — slice §8.0.S: v0.4.0, the public signatures

Six Breaking items earlier slices deferred to §8.0.O by name. v0.4.0 is already scheduled
by something else — `NSM/_verbose_deprecation.py` says *delete at v0.4.0* and v0.3.0
existing is what makes that due — so these do not need a boundary invented for them.

**Re-measured 2026-09-21**, because the row's numbers were a year of slices old:

| Item | State today |
|---|---|
| (1) the four public signatures as one set | `reconstruct_mesh` **59** named + `**kwargs`, `reconstruct_latent` **39** + `**kwargs`, `create_mesh_adaptive` **26**, `create_mesh` **17**. The row said 58 and 38. |
| (2) `reconstruct_latent(pts_surface=None)` | still present — a default that declares optional a parameter the type check has always rejected |
| (3) the `lbfgs_*` prefix decision | read them on both paths or delete the non-tested non-hybrid path; a signature call either way. `docs/KNOWN_ISSUES.md` § Open records the hybrid path as unvalidated |
| (4) refuse unknown `**kwargs` on `Decoder` / `TriplanarDecoder` | still open, and #107 asserted the config path to it: a misspelled key in a `two_stage` config's `triplanar_params` block builds at the default `padding=0.1` and says nothing |
| (5) delete `max_batch_size` | still accepted-and-warned, `NSM/reconstruct/latent_fit.py:684` |
| (6) submodule imports in `NSM/mesh/__init__.py` | **derive the rationale before doing it.** The row itself says this looks additive rather than Breaking and that §8.0.I never said why it needs the boundary. If it is additive, it does not belong in S. |
| (7) the `verbose` bridge | delete `NSM/_verbose_deprecation.py` and the 13 sites it decorates. Its own header names v0.4.0 as the delete-when condition. |

**One item on the old row is already done and should not be re-attempted.** It listed
§8.0.N's keyword-only `roundtrip_distance` / `directed_distance_percentiles` pair as
joining S. Verified 2026-09-21: `roundtrip_distance` and `forward_backward_disagreement`
are keyword-only on `main` today, and `directed_distance_percentiles` was deliberately
excluded because it is asymmetric, so a swap there changes the number rather than hiding.
That landed in §8.0.N and the CHANGELOG already carries it.

Then **tag v0.4.0**.

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

| | Scheduled (2026-08-29) | Now (2026-09-21) |
|---|---|---|
| `testing/` vs `NSM/` lines | 14,770 / 14,460 | **16,685 / 15,104** |
| ratio | 1.02 | **1.10** |
| suite wall clock | 109 s | **180 s** |

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

Neither blocks step 0 or slice P.

1. **Does slice S shrink the four public signatures, or does the config initiative?**
   Recommendation: **hand item (1) to the config initiative and ship the rest of S now.**
   The 59-parameter `reconstruct_mesh` is exactly where the wanted "set the reference mesh"
   option lands, and the config plan's `recon` section is what that signature should
   mirror. Shrinking it now and regrouping it later is two breaking changes for one result.
   This is narrower than the config plan's own §5 proposal to absorb **all** of S, which
   should be declined — that would tie the refactor's ending to a new feature being
   designed and built.
2. **Close the `grad_clip` entry in `docs/KNOWN_ISSUES.md` § Open as working by design?**
   Recommendation: **yes.** `train_epoch` passes `model.parameters()` to `clip_grad_norm_`
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
