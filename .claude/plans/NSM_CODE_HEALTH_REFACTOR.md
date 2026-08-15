# Plan: NSM code-health audit and refactor

**Status:** **Open.** Phase A (LR fix) delivered 2026-08-14/15 — PRs #9, #10, #11 merged.
Phase 0 (scope) and Phase 1 (map) delivered 2026-08-15 — `docs/SCOPE.md`,
`docs/ARCHITECTURE.md`, `docs/AUDIT_FINDINGS.md`. Phases 2–4 not started.
**Created:** 2026-08-14. **Last updated:** 2026-08-15.

> `main` merged in at `458e6e6`, so everything §4 and §9 cite now exists in this tree.
> Baseline for all measurements below: 11,861 lines, 34% coverage, 153 tests + 1 skip, 13.1s.
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

**Scope widened 2026-08-15** after Phase 1. The four items above leave the layer with the
worst findings untouched, so add:

- [ ] **The dataset cache round-trip.** `sdf_dataset.py` is 2,195 lines at 7% coverage and
      its worst findings are *silent wrong data*: `mesh_to_scale`, `uniform_pts_buffer` and
      `subsample` all change cached content and none is in `get_hash_params`, so a second
      run silently reuses the first's alignment. A harness that builds its data in memory
      never touches that path. Assert: build cache → reload → samples identical; and
      changing a hashed parameter changes the key.
- [ ] **The consumer's actual entry point, not just `reconstruct_latent`.** `kneepipeline`
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
      `docs/KNOWN_ISSUES_HISTORY.md`), and state in the harness that CPU baselines do not
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
      separately: `planning/BREAKING_CHANGE_PROPOSAL.md` +
      `planning/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` (issue #3).
- [ ] Close issues #1, #2, #5, #6 as the relevant modules are touched.

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
`KNOWN_ISSUES_HISTORY.md` entry. Neither ShapeMedKnee config contains the key and
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
| `planning/BREAKING_CHANGE_PROPOSAL.md` | Phase 1 partial | Fold into Phase 4 |
| `planning/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` | Not started | Fold into Phase 4 |
| `planning/HYBRID_OPTIMIZER_REPORT.md` | Findings, Aug 2025 | Reference for `reconstruct/main.py` |
| `.claude/plans/NSM_ICP_REGISTRATION_ROBUSTNESS.md` | Open, Phase 0 done | Resume after Phase 2 |
| `.claude/plans/NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` | Proposed | Blocked on stable interpolation API |
| `.claude/plans/NSM_TRAINING_IDEAS.md` | Open master list | Idea 3 (test Eikonal loss) belongs in Phase 3 |
| `.claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md` | Complete 2026-05-22 | Target-state example |
| `docs/MULTI_SURFACE_REGISTRATION.md` | Current | Feature doc, verify in Phase 2 |
