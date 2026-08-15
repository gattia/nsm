# Plan: NSM code-health audit and refactor

**Status:** **Open.** Phase A (LR fix) delivered 2026-08-14/15 — PRs #9, #10, #11 merged.
Phases 0–4 not started. This plan is *not* complete: Phase A was the motivating example
and the migration template, not the body of work.
**Created:** 2026-08-14. **Last updated:** 2026-08-15.
**Repo:** `gattia/nsm` (NSM).
**Motivation:** The LR-schedule bug (see §1) was a silent numerical error that ran
undetected for ~3 years and was found by an external collaborator, not by us. It is a
symptom, not an incident. This plan makes that class of bug findable and preventable.

> **Scope.** Code health only: documentation accuracy, structural mapping, test coverage,
> and decomposition of monoliths. Deliberately **out of scope**: new science
> (`NSM_TRAINING_IDEAS.md`, `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md`) and the ICP work
> (`NSM_ICP_REGISTRATION_ROBUSTNESS.md`). Those resume *after* Phase 2, on firmer ground.

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
`docs/KNOWN_ISSUES_HISTORY.md` §1.

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
as `planning/BREAKING_CHANGE_PROPOSAL.md` and stalled mid-Phase-1.

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

## 3. Phase 0 — Decide what the library is for

**Why first:** Phase 1's map cannot mark anything "deprecated" until these calls are made.
Blocking all later phases.

- [ ] Write a one-page scope statement: what NSM is, what it supports, what it does not.
- [ ] Rule on each ambiguous module — **supported**, **deprecated**, or **dead**:
  - `train/train_deep_sdf_multi_head.py` (420 lines) — verified broken: the optimizer is
    built from a leaked loop variable, so only the last decoder receives gradients.
    Deprecated Aug 2026; `train_deep_sdf` with `objects_per_decoder > 1` supersedes it
  - `train/deprecated/` (880 lines, 2 files, zero importers)
  - `mesh/refine_mesh.py` (480 lines, zero importers, 0% coverage, fully docstringed)
  - `reconstruct/reconstruct_latent_S3.py` (350 lines, 4% coverage)
  - `reconstruct/cartilage_func.py`, `reconstruct/predictive_validation_class.py`
  - `configs/generate_sdf_default_config.py` — it generates the shipped
    `default_config.json`, now pinned by `testing/NSM/configs/test_default_config_sync.py`;
    supported, not dead
- [ ] Establish the public API contract. Work through each consumer in §1.4 and record
      what it actually uses, starting with training — it is the largest surface and the
      easiest to overlook, since it is first-party and does not show up as an import in
      any other repo. Write the result into `NSM/__init__.py` as `__all__`; only then is
      everything outside it refactorable at will.
- [ ] Include the checkpoint and `model_params_config.json` formats in that contract. They
      are public interfaces even though nothing imports them.
- [ ] Survey downstream consumers for module usage before quarantining anything.

**Deliverable:** `docs/SCOPE.md` + an `__all__` in `NSM/__init__.py`.

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
- `docs/KNOWN_ISSUES_HISTORY.md` seeded, with ShapeMedKnee_2024 as a worked example.

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

## 5. Phase 1 — Map the codebase

**Order note:** this was step 1 in the original sketch. It runs *after* Phase 0 and
*before* documentation, so we never document code we are about to quarantine.

- [ ] Build a module dependency graph (`pydeps` or equivalent); identify cycles.
- [ ] For every public function/class: caller count, in-repo and in `kneepipeline`.
- [ ] Mark each module: **production** (reachable from §1.4 API) / **research** /
      **dead** / **deprecated**.
- [ ] Flag duplicated logic — notably the two `adjust_learning_rate` implementations
      (`NSM/utils.py` and `NSM/reconstruct/utils.py`) with unrelated signatures and the
      same name, a genuine trap.
- [ ] Quarantine everything ruled dead in Phase 0 → `NSM/deprecated/` + `DeprecationWarning`.

**Deliverable:** `docs/ARCHITECTURE.md` with the graph and the module ledger.
**Checkpoint:** re-measure line count. Expect ~1,800 lines quarantined.

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

**Deliverable:** docstring coverage ≥90% on surviving public API, lint-enforced.

---

## 7. Phase 3 — Test to a known-good baseline

**The step that stalled before.** It stalls because "test everything" against ~3,000
uncovered statements has no end condition. Bound it by *risk*, not by coverage percentage.

### 7.1 First and most important: end-to-end numerical regression harness

Before any unit tests. A tiny synthetic dataset (2–3 analytic meshes, 8 epochs, CPU,
fixed seed), then assert against stored baselines:

- [ ] Training: loss trajectory, final latent norms, **per-param-group LR at each epoch**
- [ ] Reconstruction: fitted latent, output mesh vertex positions, surface metrics
- [ ] Tolerances tight enough to catch a real change, loose enough to survive
      platform float noise. Store baselines as versioned artifacts.
- [ ] Must run in CI in <2 minutes.

This single harness is what protects every refactor in Phase 4, and it is the only thing
that would have caught the LR bug.

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
      separately: `planning/BREAKING_CHANGE_PROPOSAL.md` +
      `planning/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` (issue #3).
- [ ] Close issues #1, #2, #5, #6 as the relevant modules are touched.

**Rule:** every commit keeps the §7.1 harness green. Any intended numerical change gets a
§9 ledger entry and a §4-style migration guard.

---

## 9. Deliverable: the bug provenance ledger

New file `docs/KNOWN_ISSUES_HISTORY.md`. For science code this is a first-class artifact —
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

**Before starting Phase 1:** tag a release (`v2.x`) and have `kneepipeline` and `nsosim`
pin it. Gives an unambiguous rollback point and decouples their release cadence from this work.

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
| `planning/BREAKING_CHANGE_PROPOSAL.md` | Phase 1 partial | Fold into Phase 4 |
| `planning/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` | Not started | Fold into Phase 4 |
| `planning/HYBRID_OPTIMIZER_REPORT.md` | Findings, Aug 2025 | Reference for `reconstruct/main.py` |
| `.claude/plans/NSM_ICP_REGISTRATION_ROBUSTNESS.md` | Open, Phase 0 done | Resume after Phase 2 |
| `.claude/plans/NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` | Proposed | Blocked on stable interpolation API |
| `.claude/plans/NSM_TRAINING_IDEAS.md` | Open master list | Idea 3 (test Eikonal loss) belongs in Phase 3 |
| `.claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md` | Complete 2026-05-22 | Target-state example |
| `docs/MULTI_SURFACE_REGISTRATION.md` | Current | Feature doc, verify in Phase 2 |
