# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NSM (Neural Shape Models) is a deep learning library for creating generative neural models of human anatomy, with focus on musculoskeletal tissues (particularly knee structures). It uses implicit neural representations (signed distance functions) to model 3D anatomical surfaces.

## Development Commands

```bash
# Install for development (editable mode with all deps)
make install-dev

# Run all tests
make test

# Run specific test file
pytest testing/NSM/models/test_loader.py -v

# Apply isort and black (100 char line length)
make autoformat

# Check formatting and lint without modifying: isort, black, flake8
make lint

# Run tests with coverage report
make test-coverage

# Quick dev cycle (format + run loader tests)
make quick-test
```

## Code Style

- Black formatting with 100 character line length
- isort with black profile for imports
- Tests live in `testing/` directory (not `tests/`)
- Pytest is configured in pyproject.toml

## Working on this repo

Operational facts that are not derivable from the code, and that cost time to rediscover.

- **Environment:** `/mnt/data/conda-envs/nsm-dev`. `make` targets need it on `PATH`
  (`export PATH=/mnt/data/conda-envs/nsm-dev/bin:$PATH`); invoking `python -m pytest`
  directly works without it.
- **`make lint` checks isort, black and flake8; `make autoformat` applies the first two.**
  Same names as `gattia/pymskt`. `flake8` is at zero and CI gates it, so a lint failure
  blocks the test job.
- **`main` is protected** — no direct pushes. One approving review and four passing status
  checks. Admins are exempt, deliberately: GitHub forbids approving your own pull request,
  and in a single-maintainer repo that would otherwise be a permanent lock. **An admin merge
  is the normal path here, not a bypass of last resort.**
- **Fold related commits into one PR.** A docs-only change or a State-block update rides
  along with the work that caused it.
- **Reading files in tests needs `encoding="utf-8"` explicitly.** Something in the suite
  resets the locale to ASCII, so a bare `read_text()` passes in isolation and raises
  `UnicodeDecodeError` under the full suite.

## Making Changes

This is a research library with one maintainer. Every line added is a line someone
maintains alone, and dead code here is worse than a gap — it looks load-bearing.

The rules below were paid for by the Aug 2026 LR-schedule fix, which took four rounds of
review to get from **+341 lines to +173** in `NSM/utils.py` with no loss of function.
Each one names the specific mistake it prevents.

**Run the claim before you write it.** Assertions about what a change does were wrong more
often than right until executed: "`state_dict()` drops custom group keys" (false),
"schedule_free runs were unaffected" (true in the narrow sense, backwards in practice),
"regenerating the default config changes it" (behaviourally inert), "the `"None"` sentinel
is cosmetic" (it is truthy, so it is a trap). Every one of these was settled by a
five-line script. Write the script first; the claim is the output, not the input.

**Do a deletion pass before presenting.** For each symbol you added, ask what actually
breaks if it is removed, and answer by deleting it and running the tests. Roughly 100
lines survived the first draft of that fix purely because nobody asked. `/simplify` does
this if you would rather not do it by hand.

**Never inherit a rationale along with the code.** If you port something and its stated
justification turns out to be false, the code goes too unless you can independently
justify it. Disproving the premise and keeping the conclusion — with a freshly invented
reason — is how the 40 lines of `optimizer_group_names` plumbing survived a full review.

**Separate permanent from transitional at write time, not later.** Migration helpers,
deprecation shims and one-time explainers go in their own module with a delete-when
condition in the header (see `NSM/_lr_migration.py`). Inline, unmarked, they become
indistinguishable from permanent API within a year.

**Fix the class of defect, not the reported instance.** The LR bug was positional
coupling. Fixing only the reported site left two more instances of the same coupling in
the same code path, each found in a later round. When a bug has a shape, enumerate every
place that shape occurs before proposing a fix.

**Size docs to the reader's need, not your uncertainty.** Long docstrings on functions you
found hard to reason about are self-soothing. Error text that restates a document it
already links to is padding. Keep what the reader cannot look up.

### Plans

A plan for a non-trivial change should state, before any code:

- which parts are **permanent API** and which are **transitional**, with the condition
  under which the transitional parts get deleted
- roughly how much code the permanent part justifies, so growth past it is visible
- the **verification** for each behavioural claim the plan rests on — the script or test
  that settles it, not the reasoning that suggests it

### Numerical-behaviour changes

Any fix that silently changes training or reconstruction output for inputs that
previously ran without error needs an entry in `docs/KNOWN_ISSUES.md` § History. The test
is whether a reader can determine, years later, if a run they have on disk is affected and
what to do about it. Bugs that always crashed need no entry — nobody has results from them.

## Documents and work

**Two homes for knowledge, one queue for work.**

| Where | Holds | Deleting it loses |
|---|---|---|
| `docs/` | What is true of the library | Facts about the code |
| `.claude/plans/` | Intent and state of in-flight work | Why you are mid-refactor |
| GitHub issues | What we intend to fix | The queue, not any fact |

Nothing else. No `planning/`, no notes files, no findings registers, no handoffs. When
something has no obvious home it goes to an issue or `docs/KNOWN_ISSUES.md` — starting a
new file is how a repo ends up with eleven of them.

### `docs/` — three files, plus one per user-facing feature

- **`SCOPE.md`** — supported, deprecated, dead, or unsupported by design
- **`KNOWN_ISSUES.md`** — **Open**: reproduced, user-visible, not fixed yet.
  **History**: was wrong, silently changed results, now fixed.
- **`ARCHITECTURE.md`** — invariants and traps. No table a command regenerates.

Durable facts live here, never in an issue. Issues live on GitHub; the repo is what
survives. "Which of my runs are affected by this" must be readable in 2031 by someone who
has the repo and not necessarily the tracker.

### `.claude/plans/` — one per initiative, ≤5 active

Each plan opens with its own state. There is no separate handoff file: a global
"where we are" cannot survive two concurrent plans. For what a plan must state before any
code is written, see [Plans](#plans) above.

```
## State
**Updated:** YYYY-MM-DD · **Status:** open | blocked | done

- **Next:** the single next action
- **Blocked on:** nothing
- **Done:** what landed, each line naming the PR that landed it
- **Surprises:** an assumption that turned out false, and what replaced it
```

A completed plan **keeps its body** and moves to `completed/`, gaining two sections:

```
## Delivered   — what actually shipped, with PR links
## Diverged    — where reality differed from the plan, and why
```

`Diverged` is the most valuable thing in the file and it exists nowhere else: the code
shows what was built, git shows when, and only this shows what we believed beforehand and
why it was wrong. Do not compress it.

What a plan is *not* is a running log. Keep the intent as written, mark what changed
against it, and let the PR links carry the detail. Ideas go in the single ideas file,
never one file per idea.

### Issues — the only work queue

The bar is **evidence plus a fixable statement**: you reproduced it, and you can say what
"fixed" means. You cannot file one without having run something. That bar is the whole
point — it is what stops a tracker filling with speculation.

- **Closes** by a PR, or by a decision not to fix — which moves it to `KNOWN_ISSUES.md`
  § Open or `SCOPE.md`. A won't-fix that closes nowhere becomes the eleventh document.
- **Order by `file:function`**, so issues surface as you open each file. That is the only
  moment anyone has the context to act on one.
- Tests name the issue number in their `xfail` reason.

### Four rules

1. **A number is computed or it is not committed.** Worth keeping → a test computes it (a
   tolerance's headroom over its measured break; an assertion goes red rather than stale)
   or CI publishes it (coverage, suite timing). What is forbidden is the hand-transcribed
   number: stale the day after it is written, and nothing says so.
2. **Inference is not a finding.** If you did not run it, it is a hypothesis: it goes in a
   plan's **Next**, not in `docs/` and not in an issue.
3. **Evidence lives in the test and its docstring.** A measurement that cannot be asserted
   — why a deliberate break is 20 vertices and not 1 — belongs in the docstring of the test
   it constrains. Delete it and the next reader simplifies the test back into uselessness.
4. **A PR that closes work deletes the notes that closing it made obsolete, and leaves a
   pointer to the PR that closed it.** Not "net-negative lines" — a permanent, non-obvious
   fact should cost lines. What must not survive is scaffolding answering a question nobody
   is asking.

### Correcting

When you touch code a `docs/` file describes, verify it in the same commit or put
`> ⚠️ Unverified since <date>` at its top. A stale doc that says so is honest.

## Architecture

### Core Modules

**`NSM/models/`** - Neural representation architectures:
- `deep_sdf.py`: Standard DeepSDF implicit representation using MLPs
- `triplanar.py`: Three-plane feature grid decomposition decoder
- `modulated_periodic_activations.py`: SIREN-style networks with modulated activations
- `two_stage.py`: Hybrid triplanar + MLP approach
- `loader.py`: Unified model loading interface with config templates (supports: triplanar, deepsdf, two_stage, implicit)

**`NSM/datasets/sdf_dataset.py`** (~2200 lines) - Core dataset class:
- Generates signed distance function samples from 3D meshes
- Supports multi-surface rigid registration (aligning multiple anatomical structures)
- Handles mesh preprocessing, scaling, centering, ICP registration
- Caches processed data in H5 or numpy formats

**`NSM/train/`** - Training pipelines:
- `train_deep_sdf.py`: Main training loop
- `train_deep_sdf_multi_head.py`: Multi-head training for multiple surfaces
- `utils.py`: Weight scheduling (linear, exponential, exponential_plateau, constant), KLD loss

**`NSM/reconstruct/`** - Reconstruction and evaluation:
- `main.py` (~1400 lines): Core reconstruction pipeline - latent code optimization, surface reconstruction, mean shape computation
- `recon_evaluation.py`: Evaluation metrics with logging
- `cartilage_func.py`: Cartilage-specific analysis

**`NSM/mesh/`** - Mesh processing:
- `main.py`: Marching cubes with adaptive refinement, deterministic coarse grid bounds
- `refine_mesh.py`: Mesh refinement techniques
- `interpolate.py`: Mesh interpolation utilities

**`NSM/losses.py`** - Eikonal loss for enforcing ||∇SDF|| = 1 constraint
 NOTE - EIKONAL LOSS HAS NOT BEEN TESTED. WE SHOULD TEST THIS
 TO MAKE SURE IT WORKS, DOESNT ERROR, AND TO SEE HOW IT 
 CHANGES THINGS. NAMELY - DOES IT CHANGE INTERPOLATION?

### Key Concepts

- **SDF (Signed Distance Function)**: Core representation - positive values outside surface, negative inside, zero at surface
- **Latent codes**: Learned embeddings that encode shape variation across anatomical samples
- **Multi-surface registration**: Aligns multiple anatomical structures (e.g., medial+lateral menisci) in a common reference frame

### Multi-Surface Config: `objects_per_decoder` and `mesh_names`

When training models that decode multiple surfaces (e.g., bone + cartilage + menisci), two config parameters control the output:

- **`objects_per_decoder`** (int): Number of surfaces the decoder outputs. Set via `config.setdefault()` in `train_deep_sdf.py`. Saved to `model_params_config.json` automatically.

- **`mesh_names`** (list of str, optional): Human-readable names for each decoder output, in order. For example: `["bone", "cart", "med_men", "lat_men"]` for a 4-surface femur model. Must have the same length as `objects_per_decoder`. Saved to `model_params_config.json` alongside other config fields.

**Always include `mesh_names` in new training configs.** Without it, downstream consumers (e.g., nsosim) must infer mesh identity from the output count, which is fragile. A warning is emitted during training if `mesh_names` is not provided.

Example config snippet:
```json
{
    "objects_per_decoder": 4,
    "mesh_names": ["bone", "cart", "med_men", "lat_men"]
}
```

### Learning Rate Schedules: the `Target` key

Every `LearningRateSchedule` entry **must** declare `"Target"`, either `"model"` or
`"latent"`. Exactly two entries, one per target. **Entry order is ignored.**

```json
"LearningRateSchedule": [
    {"Target": "model",  "Type": "Step", "Initial": 0.001,  "Interval": 500, "Factor": 0.5},
    {"Target": "latent", "Type": "Step", "Initial": 0.0005, "Interval": 500, "Factor": 0.5}
]
```

A config missing `Target` on any entry — including only one of the two — **raises**, with
a message that prints the paste-ready annotation reproducing that run's historical
behaviour. This applies to every optimizer, `schedule_free_*` included.

Note the two optimizer families migrate to **opposite** annotations. Adam/AdamW ran
through `adjust_learning_rate()` (entry 0 drove the latents); `schedule_free_*` skipped it
and kept `get_optimizer()`'s own assignment (entry 0 drove the model). The error message
picks the right one from `config["optimizer"]`.

**There is no positional indexing anywhere in the LR path**, and `target` is the single
vocabulary spanning config and optimizer:

- **Schedule entries** carry `Target`. `get_learning_rate_schedules()` returns a
  `{target: schedule}` **dict**, not a list — there is no index to get wrong.
- **Param groups** carry `target` too, so `adjust_learning_rate()` is one lookup:
  `group["lr"] = lr_schedules[group["target"]].get_learning_rate(epoch)`.
- Param groups also carry `name` (`latent`, `model_0`, …), but that is a **human label
  only** — nothing dispatches on it. Renaming a group changes nothing.

Several groups may share a target: every decoder and the classification heads all take
the `model` schedule. That many-to-one relation is why group `name` and schedule `Target`
are separate fields rather than one.

Background: `docs/KNOWN_ISSUES.md` §1 — a positional-mapping bug swapped these two
schedules on every Adam/AdamW run from May 2023 to Aug 2026.

### Dependencies

Key libraries: PyTorch (core ML), mskt (pip name for pymskt - musculoskeletal toolkit), pymeshfix, einops, wandb (experiment tracking)