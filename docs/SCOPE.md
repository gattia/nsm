# NSM scope

**Phase 0 deliverable of `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md`.**
**Verified:** 2026-08-15, against `main` at commit `73a0326`.
**§2.8 and the 2026-08-22 amendments to §1, §2.6 and §3.1:** verified 2026-08-22, against
`main` at `986fded` (post-PR #64) — every claim in them was re-run, not transcribed.

> ⚠️ **Line references predate the Aug 2026 seeding work.** That work moved
> `sdf_dataset.py` by over 100 lines, so a `file:line` below may not land where it did when
> this was written. The rulings and the module inventory are unaffected; only the line
> numbers are. Re-locate by symbol name rather than trusting a number.

This document makes the calls that Phase 1 needs before it can mark anything for
quarantine: what NSM is for, what each module's status is, and what the public API is.

---

## 1. What NSM is

NSM is a **training library** for implicit neural shape models of anatomy. Its product is
a trained decoder — a network mapping (latent code, xyz) to a signed distance — plus the
machinery to fit a latent code to a new mesh and turn it back into a surface.

Most of the 11.9k lines are the product, not internals. Training, dataset construction,
meshing and reconstruction are all things a user calls directly.

**It supports:**

- Training a decoder from a directory of meshes, single- or multi-surface, from a JSON
  config (`NSM.train.train_deep_sdf`).
- Fitting a latent to an unseen mesh and reconstructing surfaces from it
  (`NSM.reconstruct.reconstruct_mesh`) — this is the inference path shipped downstream.
- Loading a trained model from `model_params_config.json` + a checkpoint
  (`NSM.models.load_model`).
- Latent-space interpolation with point correspondence (`NSM.mesh.interpolate`).
- Scoring correspondence quality (`NSM.mesh.correspondence_metrics`).

**What it is meant to be, and is not yet.** The name is plural — *neural shape model**s***.
The library is not a wrapper around one hybrid model; adding a new architecture and having
it work end-to-end is the point. Two things currently prevent that, and both are defects to
fix rather than limitations to document:

- **Only `TriplanarDecoder` survives the reconstruction path.** `reconstruct_latent` calls
  decoders with a keyword-only `(latent=, xyz=)` interface that only `TriplanarDecoder`
  implements (`reconstruct.reconstruct_latent`), with no fallback — while `mesh.decode_sdf`
  inspects the signature and *does* fall back. Two conventions in one pipeline; `load_model`
  advertises four model types and three of them cannot be reconstructed.
  → **Phase 4 work item: a common decoder interface plus a registration pathway**, so a
  third party can add a model and have it work in train, reconstruct, mesh and interpolate
  without editing NSM internals. One calling convention has to win.

  The `implicit` type is the furthest gone of the three, in two independent ways
  (audit rulings, re-verified 2026-08-22): `loader._get_implicit_params` requires
  `latent_dim`/`hidden_dim`/`num_layers` — a vocabulary no real training config uses
  (the shipped configs carry `latent_size`/`layer_dimensions`) — and both that loader
  and `ImplicitDecoder` default the output through a sigmoid, whose (0, 1) range cannot
  represent a signed distance. Unreachable from real configs, and non-SDF by default
  even when reached. Fold any fix into the registration-pathway work; neither half is
  worth patching in isolation.
- **The shipped `default_config.json` describes only the triplanar production model.**
  PR #64 (issue #48) replaced the old 61-key DeepSDF-shaped default — which could not
  drive `train_deep_sdf` at all — with a sanitized snapshot of the ShapeMedKnee
  `647_nsm_femur_v0.0.1` triplanar config, pinned by the generator-sync test and a test
  that instantiates the trainer from the shipped file. That delivers the first of the
  per-model-type defaults.
  → **Remaining Phase 4 work item: a default config for each *other* model type**
  (deepsdf, two_stage; `implicit` first needs the vocabulary reconciliation above).

**Genuinely experimental — needs a warning, not a fix:**

- **The Eikonal loss.** Wired into both live loss paths, never executed by any test
  (`losses.py` at 10% coverage), and never run by its author. It may help considerably;
  nobody knows, including whether it works at all. It needs a loud warning at the point of
  use, and a minimum test that answers "does this do anything" before any claim is made
  about whether it helps.
- **`train_deep_sdf_multi_head`.** Kept (see §2.1), but its training parameters have never
  been tuned and it has never been used in anger. **Do not advertise it as a supported way
  to train models.** Its capability is real; its readiness is not.
- **`multi_object_overlap`.** A config key both live trainers read and neither implements:
  enabling it raises `Exception("Not implemented yet")` mid-epoch, after data loading and
  the first forward pass (`train_deep_sdf.train_epoch` and its multi_head counterpart;
  re-verified 2026-08-22). Accepted by config, not implemented, crashes any run that sets
  it — the same shape as the eikonal ruling above, and it gets the same treatment: it is
  not a defect to patch in isolation, it is an unbuilt feature whose key must not read as
  a working option.

---

## 2. Module rulings

Five modules were named in the plan as needing a status call. **All five proposed rulings
were refuted**, each by a dedicated skeptic that searched for importers across the repo,
the downstream consumer, all branches, and git history before ruling.

The plan's Phase 1 checkpoint expects "~1,800 lines quarantined." The defensible number is
**564** — and 12 of those lines must be ported out first.

### 2.1 `train/train_deep_sdf_multi_head.py` (428 lines) — **supported, broken, fix it**

Proposed: *deprecate; superseded by `train_deep_sdf` with `objects_per_decoder > 1`.*

**Not superseded.** The two trainers are different architectures, not two spellings of one:

- `train_deep_sdf(config, model, ...)` takes **one** decoder emitting N channels from
  shared hidden layers (`train_deep_sdf.train_epoch`).
- `train_deep_sdf(config, models: tuple, ...)` in multi_head takes **N independent
  decoder networks** against a single shared latent embedding, with per-surface loss
  weighting (`train_deep_sdf_multi_head.train_epoch`).

Deleting it removes the multi-network-per-latent capability from the library entirely.
The defect is real — the optimizer is built from a leaked loop variable in `train_deep_sdf`,
so only
the last decoder is trained — but it is a two-identifier repair (`model` → `models`)
against a `get_optimizer` that already normalizes list input and emits one `model_{idx}`
group per decoder (`utils.get_optimizer`).

**Ruling: supported. Fix the optimizer, keep the `DeprecationWarning` until it is fixed.**
The plan's §3 text on this module is wrong and should be corrected.

Two qualifications from the maintainer:

- **Its current warning text is actively wrong and must be rewritten.** It says "Use
  `NSM.train.train_deep_sdf` with `'objects_per_decoder' > 1` instead"
  (its `DeprecationWarning`) — advice that silently hands the user a different
  architecture. It should say broken-and-unfixed, and name no replacement.
- **Do not advertise it as a supported training path.** Its hyperparameters have never been
  tuned and it has effectively never been used. Keep the capability, keep it out of the
  documented surface until someone runs it.

### 2.2 `train/deprecated/` (880 lines) — **split; one is dead, one is not yet**

- `train_deep_sdf_multi_surface_orig.py` (562) — strict subset of `train_deep_sdf.py`.
  Nothing unique. **Dead. Quarantine.**
- `train_deep_sdf_orig.py` (318) — contains the only *live* `sample_difficulty_lx`
  inverse-Lx loss-weighting branch (in its `train_epoch`). `train_deep_sdf.py` stops at
  `sample_difficulty_weight` and has that algorithm only as a commented-out
  block. **Port those ~12 lines into `train_deep_sdf.py` first** — the helpers are already
  imported there — then it is dead.

  Two conditions on the port. **It must be impossible to enable by accident**: nobody has
  used it, so its off-state has never been exercised, and a feature whose disabled path is
  untested is a feature that turns itself on eventually. **And it must be documented at the
  config key**, not just in code. This is the specific instance of a general problem the
  maintainer raised — the config options are poorly documented and many names do not
  describe what the code does with them. That is tracked as the config overhaul in §1.

  The inverse hazard is equally live and easy to miss: config keys a user can set that
  silently do nothing because the branch implementing them is commented out. Those read as
  working features and produce no error.

Neither has an importer; `train/deprecated/` has no `__init__.py`, so neither appears in
the coverage denominator either. 880 untested lines are currently invisible to `make
test-coverage`.

### 2.3 `mesh/refine_mesh.py` (480 lines) — **research, keep**

Proposed: *zero importers, 0% coverage, therefore dead.*

Zero importers is confirmed. "Therefore dead" is not. `subdivide_triangles_on_base_mesh`
selects cells by metrics computed on **one** mesh and splits them on a
**different** mesh, preserving original point IDs. `pyvista.subdivide_adaptive` is present
and cannot express that base/warped split — the completed interpolation plan records that
both were tested for exactly this reason, and states in writing that the hand-built code
was kept deliberately.

Dead code is code nobody decided about. This is code someone decided to keep, in writing.

**Ruling: research. Keep, documented as such** — with three conditions, in order:

1. **Make it work.** `get_target_cells` reads `np.zeros_like(max_length_binary)` where it
   means
   `max_lengths`, so both public entry points raise `UnboundLocalError` on their own
   defaults. One-word fix, and it comes first — documenting a module that raises describes
   something nobody can run.
2. **Warn at the entry points.** It is research code with a precondition nobody states:
   `subdivide_triangles_on_base_mesh:465` computes cell indices on one mesh and applies them
   to a different one, which is valid only if the two share connectivity and cell ordering.
   Violating it produces a wrong mesh, not an error.
3. **Document what it is for and what not to do with it,** in the module docstring so it
   travels with the code: the cross-mesh/ID-preserving capability it uniquely provides, why
   `pyvista.subdivide_adaptive` was tested and rejected, its preconditions, and what is
   known broken. Including that `area_threshold` is compared against a *relative deviation*,
   not an area, despite three docstrings calling it "the maximum area of a triangle."

### 2.4 `reconstruct/reconstruct_latent_S3.py` (350 lines) — **deferred research, scheduled**

Proposed: *near-zero coverage, no first-party caller, therefore dead.*

It is the only implementation of joint differentiable Sim(3) pose + latent optimization
(arXiv:2004.09048) in the repo. `reconstruct_mesh(register_similarity=True)` is
categorically different — non-differentiable ICP as preprocessing, pose then held fixed.

More decisively, it is an active work item: branch `icp-registration-robustness` carries a
plan whose §5 states the module has a gradient-flow bug that alone explains the earlier
negative result, so the method has not had a fair test.

**Ruling: deferred research, scheduled for repair.** Keep the `reconstruct_latent_S3`
re-export from `reconstruct/__init__.py` — removing it is a public-surface break.

### 2.5 `reconstruct/cartilage_func.py` (149) and `predictive_validation_class.py` (97)

Proposed: *research-only, no production caller.* Wrong on the caller half for both.

- `cartilage_func.py` is imported by the **live** trainer and wired into its
  `DICT_VALIDATION_FUNCS`, dispatched by config key
  `recon_val_func_name`. It also owns the only region-index maps in the repo
  (`CART_REGIONS`, `CART_REGIONS_DICT`). **Production.**
- `predictive_validation_class.py` is called from `reconstruct/main.py`. It is the only
  latent-to-factor regression validator. **Research.** Its seam defect —
  `reconstruct.get_mean_errors` passed the whole result dict to `Regress.add_latent`
  instead of the fitted latent — was repaired Aug 2026 (#48).

**Maintainer confirmation:** both were used for training and validation in the ShapeMedKnee
paper, where they were critical. They are clunky and lightly used now, which is what made
them look disposable from the call graph alone. They are not. This is the clearest case in
the audit for why importance and recent-usage are different measurements — and the seam
defect above is pinned down (2026-08-23): `reg.add_latent(result_)` was born in that form
in `2811d27` (Jul 2023, pre-rename `GenerativeAnatomy`), so the validator never worked
through `get_mean_errors` at any point in this repo's history. How the paper's validation
actually ran is not answerable from this repo.

### 2.6 Rulings not yet adjudicated

| Module | Lines | Status | What decides it |
|---|---|---|---|
| `models/loader.py` | 387 | **production — fix, under investigation** | Not a status question. It is the documented entry point (README, `examples/`) *and* the natural home of the extensibility work in §1, since `load_model` is what a registration pathway would hang off. But three of its four advertised model types cannot be reconstructed, and the consumer does not use it — `steps/run_nsm.py:94-112` hand-rolls the config→constructor mapping instead and drops `padding`. **Open question being investigated: could the consumer switch to `load_model` today, and if not, what exactly is missing?** That answer sets the size of the fix. |
| `mesh/triangle_metrics.py` | 97 | **keep — scope under investigation** | Both importers (`correspondence_metrics`, `refine_mesh`) are themselves unreached from production, so it cannot be ruled on independently of §2.3. Two open questions: is all five of its public symbols live, or only the part `correspondence_metrics` uses; and its `areas(norm=True)` default returns a relative deviation rather than areas, which is what makes `refine_mesh`'s `area_threshold` misleading. **Keep either way** — the question is whether it stays a separate file or the live part merges into `correspondence_metrics`. Input to that merge decision, from the audit (re-verified 2026-08-22): the two modules implement the edge-ratio statistic with deliberately opposite failure behaviour — `TriangleProperties.edge_ratio` raises on a zero-length edge, `correspondence_metrics.triangle_health` degrades gracefully and reports a `degenerate_count`. A merge must reconcile that split or keep it, deliberately. |
| `datasets/utils.py` | 2 | **dead** | A two-line TODO proposing the Phase 4 `sdf_dataset` split. Zero importers. Delete when Phase 4 does the split it describes. |
| `configs/generate_sdf_default_config.py` | 112 | **supported** | Confirmed — it owns the shipped `default_config.json` and is pinned by `test_default_config_sync.py`. The plan already ruled this correctly. |

### 2.7 Net effect on Phase 1's checkpoint

**"Quarantine" defined,** since the plan uses the word without introducing it. It is the
middle rung of three:

| | What happens | Reversible by |
|---|---|---|
| **Deprecate** | Code stays put and still works; calling it emits a `DeprecationWarning`. | Deleting the warning |
| **Quarantine** | Code *moves* to a `deprecated/` directory. Still importable, still works, visibly not part of the live library. | `git mv` back |
| **Delete** | `git rm`. Gone from the working tree. | Git history only |

Principle 2 prefers quarantine over delete because downstream forks may reach into
anything, and `git rm` converts "someone's pipeline broke" into a support burden with no
visible cause. Moving a file at least leaves it findable.

For this repo the distinction is nearly moot: both files below are *already* in
`NSM/train/deprecated/`, quarantined in Aug 2025. So Phase 1's quarantine step is close to
a no-op, and the real open decision is the one after it — see the note below the table.

| | Lines |
|---|---|
| Quarantine now: `train/deprecated/train_deep_sdf_multi_surface_orig.py` | 562 |
| Quarantine after porting 12 lines: `train/deprecated/train_deep_sdf_orig.py` | 318 |
| Delete when Phase 4 lands: `datasets/utils.py` | 2 |
| **Total** | **882** |
| Plan's expectation | ~1,800 |

They are already in a `deprecated/` directory, so the quarantine step is close to a no-op —
what is missing is an `__init__.py` so coverage counts them, or removal so they stop being
counted as library code at all.

**No module ruled dead had zero cost to remove.** That is the finding, and it argues for
keeping Principle 2 ("quarantine, don't delete") rather than relaxing it.

### 2.8 Function-level rulings — audit round, ruled 2026-08-22

The Aug 2026 audit (register since deleted; disposition approved by the maintainer
2026-08-22) surfaced symbols whose status no module-level ruling covers. Each claim below
was re-verified by execution in the commit that wrote it.

**Ruled dead and deleted** (the maintainer-approved cluster — the exception to
Principle 2, because every one was unreachable or content-free, so there is no downstream
use to break):

- `symmetric_chammfer` (was in `NSM/utils.py`) — a `pass` stub with a whitespace-only
  docstring, returning `None` to any caller. Zero callers.
- `sdf_gradients` (was in `NSM/mesh/interpolate.py`) — zero callers, including inside its
  own module (the interpolation path computes gradients through its own private helpers).
  Its return prepended latent-width columns of fabricated zeros presented as gradient —
  98.8% zero padding at the production latent size.
- `find_object_bounds_random_sampling` (was in `NSM/mesh/main.py`) — zero callers,
  non-deterministic by construction, and superseded by the deterministic
  `main.coarse_bounds_from_sign_change`. A stale gitignored `build/` tree is the only
  thing that still referenced it; do not let a grep over `build/` resurrect it.
- `NSM/configs/deep_sdf_config` — a 404-byte scratch-notes file, untouched since the
  initial commit, read by nothing, excluded from wheels (`NSM/configs` has no
  `__init__.py`), and preserving the obsolete two-positional-entry LR shape as if it were
  documentation.

**Ruled dead, deletion deferred to the review that owns the file** — each was left in
place so its removal happens in one reviewed pass over its module, not as a drive-by:

| Symbol | Evidence (re-run 2026-08-22) | Delete with |
|---|---|---|
| `utils.compute_assd` (reconstruct) | Its only import is commented out (`recon_evaluation` imports `compute_chamfer  # , compute_assd`); the live ASSD path is pymskt's `get_assd_mesh` in `recon_evaluation.compute_recon_loss` | the #20 cleanup of `reconstruct/utils.py` |
| `main.tune_reconstruction` (reconstruct) | Zero callers; reads 27 config keys of which 22 are absent from the shipped default, so no shipped config can drive it | Phase 4 decomposition of `reconstruct/main.py` |
| `main.compute_correlation_coefficient` (reconstruct) | A four-line `np.corrcoef` wrapper, zero callers | Phase 4 decomposition of `reconstruct/main.py` |
| `losses.l1_loss`, `losses.l2_loss` | One-line re-exports of torch's functional l1/mse losses, labelled "legacy aliases"; zero callers | the eikonal repair's pass over `losses.py` (plan §8.2) |

**Ruled kept despite zero callers:**

- `losses.compute_sdf_gradients` and `losses.combined_sdf_loss` — uncalled today, but
  they are the eikonal helper surface: `compute_sdf_gradients` carries the same
  `retain_graph` defect the eikonal repair must fix, and both stand or fall with that
  repair (plan §8.2), not with caller count. Experimental, same ruling as the eikonal
  loss itself (§1).

Two audit rulings needed no new text, verified rather than assumed: `refine_mesh`'s
cross-mesh cell-indexing precondition is already condition 2 of §2.3, and the
only-TriplanarDecoder reconstruction limit is already §1's first bullet.

---

## 3. The public API contract

### 3.1 What the downstream consumer actually uses

`kneepipeline` imports exactly **two** symbols:

| Symbol | Import site | Contract |
|---|---|---|
| `NSM.models.TriplanarDecoder` | `steps/run_nsm.py:85` | Constructed with 15 named kwargs read out of `model_params_config.json`; then `load_state_dict(...)`, `.cuda()`, `.eval()`. |
| `NSM.reconstruct.reconstruct_mesh` | `steps/run_nsm.py:170` | Called with 27 kwargs, all by name. Result keys read: `mesh[0]`, `mesh[1]`, `latent`, `icp_transform`, `center`, `scale`, `assd_0`, `assd_1`. |

`steps/compute_bscore.py` imports nothing from NSM. Its coupling is the on-disk
`NSM_recon_params.json` and one key, `latent`.

Two things about that surface are load-bearing and undocumented:

1. **`reconstruct_mesh`'s result `mesh` list is ordered, and the order is the contract.**
   The consumer hardcodes index 0 = bone, index 1 = cartilage
   (`steps/run_nsm.py:216,220,232,235`). Nothing in the signature, docstring, or returned
   dict names the surfaces — and the repo already has a `mesh_names` config field for
   exactly this, which `NSM/models/` never reads. This is the same undocumented-positional
   -ordering shape as the LR bug.

   The same assumption is admitted in code one layer down (audit ruling, re-verified
   2026-08-22): when a fit has fewer ground-truth surfaces than the decoder has outputs,
   `main.reconstruct_latent` silently `break`s out of the surface loop under an in-code
   TODO that says outright "it assumes the first surface is the bone / only of interest".
   A deliberate, written-down design compromise, not a defect to file — it is recorded
   here because it is one more instance of the positional-surface-identity contract this
   section owns, and any surface-naming fix must cover it.
2. **The consumer hand-rolls the config→constructor mapping and omits `padding`.** It
   passes 15 of `TriplanarDecoder`'s 16 meaningful arguments. `padding` is not a learned
   parameter, so a checkpoint trained at a different value loads cleanly under strict
   `load_state_dict` and then samples the feature planes at the wrong scale, silently.
   The duplicated mapping exists because NSM offers no supported "build the model this
   config describes" call that the consumer can use — `load_model` exists but is not what
   the consumer uses. **Closing that gap is the single highest-value API change available.**

`reconstruct_mesh` has **one executed line** in the entire test suite: its `def`.

**Deprecated, with a delete-when (audit ruling, re-verified 2026-08-22):**
`batch_size_latent_recon`. `reconstruct_mesh` dropped the parameter, absorbs it via
`**kwargs`, and prints a deprecation warning on every call — while the consumer still
passes it (`steps/run_nsm.py`) and `main.get_mean_errors` still takes it as a real
parameter. The shim behaves correctly; what the audit flagged is that it is inline and
undated, indistinguishable from permanent API (the failure shape CLAUDE.md § "Separate
permanent from transitional" names). **Delete the shim when kneepipeline stops passing
the argument**; the kneepipeline-side change is a consumer cleanup, not an NSM defect.

### 3.2 Proposed `__all__` tiers

**What `__all__` is,** since NSM has never had one — `grep -rn __all__ NSM/` returns nothing
anywhere in the package. It is a module-level list of strings naming what is public:

```python
# NSM/models/__init__.py
__all__ = ["TriplanarDecoder", "load_model", "list_supported_models"]
```

It does two things, and changes no behaviour beyond the first:

1. **It controls `from X import *`.** Without it, a star-import takes every name not
   starting with `_` — *including modules the file itself imported*. That is why
   `NSM.reconstruct` currently exposes `os`, `sys`, `torch`, `np`, `wandb`, `logging` and
   `mskt` as if they were NSM API, and why `from NSM.reconstruct import
   adjust_learning_rate` silently binds the wrong one of the two functions by that name
   (§6 of `ARCHITECTURE.md`).
2. **It states intent in writing** — "these names I will try not to break; everything else
   is mine to change." That is the part the plan actually wants from it. Without that line
   there is no difference between refactoring an internal helper and breaking a consumer.

Adding it breaks nothing. The tiers below are the proposed content.

**public-stable — 6.** Breaking any of these breaks a known consumer or the documented
example. These are the only names that should carry a compatibility promise.

```
TriplanarDecoder   reconstruct_mesh   load_model
list_supported_models   get_model_config_template   __version__
```

**public-provisional — 48.** The first-party training and mesh surface. Wanted, used,
documented in places, but not frozen. Includes `train_deep_sdf`, `SDFSamples`,
`MultiSurfaceSDFSamples`, `reconstruct_latent`, `create_mesh_adaptive`, the LR target
vocabulary (`LR_TARGET_KEY`, `LR_TARGET_MODEL`, `LR_TARGET_LATENT`, `LR_TARGETS`,
`PARAM_GROUP_TARGET_KEY`), the checkpoint writers (`save_model`, `save_latent_vectors`,
`save_model_params`), the nine correspondence metrics, the five interpolation functions,
and the five cartilage-comparison validators. Full list in the workflow output; it should
be transcribed into code when §3.3 is resolved.

**internal — everything else,** including all 24 names currently leaked into
`NSM.reconstruct` by `from .main import *` (among them `os`, `sys`, `torch`, `np`,
`wandb`, `logging`, `mskt`, and `adjust_learning_rate`).

### 3.3 Why `__all__` is not being written into `NSM/__init__.py` yet

The plan's Phase 0 deliverable says "an `__all__` in `NSM/__init__.py`". As specified this
cannot be done, and the reason is worth recording rather than working around.

`NSM/__init__.py` imports **only** `utils`. After a bare `import NSM`, `NSM.models`,
`NSM.reconstruct`, `NSM.mesh`, `NSM.datasets` and `NSM.train` do not exist — every
consumer reaches them by writing `from NSM.models import ...`, which triggers the
submodule import as a side effect. So a top-level `__all__` naming `TriplanarDecoder` or
`reconstruct_mesh` would either name unbound symbols, or force `NSM/__init__.py` to import
every subpackage eagerly.

Eager import is not cheap or neutral here. `NSM.models` is fully isolated — importing it
does not pull `wandb` — which is precisely why the consumer's
`from NSM.models import TriplanarDecoder` is fast. Importing `NSM.reconstruct` pulls
`wandb`, `pymskt`, `vtk`, `point_cloud_utils`, and reconfigures the **root logger** for the
host process (at `reconstruct/main.py` module scope). Making that unavoidable for anyone who
types
`import NSM` is a regression, not a cleanup.

**Recommendation:** put `__all__` in each subpackage `__init__.py`, which is where the
leakage actually is, and leave the top level lazy (or use PEP 562 `__getattr__` if
`NSM.models` should resolve from a bare `import NSM`). Either way it is a **code change
with import-semantics consequences**, so it belongs in a reviewed commit, not in a mapping
pass. Amend the plan's Phase 0 deliverable to "a written API contract + `__all__` per
subpackage."

---

## 4. Format contracts

These are public interfaces even though nothing imports them. Changing any of them breaks
consumers silently.

| Artifact | Written by | Read by | Versioned? |
|---|---|---|---|
| `model_params_config.json` | `utils.save_model_params` | `load_model`, `examples/load_trained_model.py`, **both consumer scripts (hand-rolled)** | No |
| checkpoint `{epoch, model, optimizer}` | `utils.save_model` | `loader.py` (4 possible key layouts), consumer `load_state_dict` | Pre-Aug-2026 refused at load |
| `latent_codes/{epoch}.pth` | `utils.save_latent_vectors` | `train_deep_sdf` on resume | No |
| `LearningRateSchedule[].Target` | config author | `utils.resolve_schedule_targets` | Yes — missing key raises with a migration message |
| SDF cache `.npz` / `.h5` | `sdf_dataset` | `sdf_dataset` | **No, and the hash is wrong** — see below |
| `NSM_recon_params.json` → `latent` | consumer, from `reconstruct_mesh` | `steps/compute_bscore.py:72` | No |

The dataset cache hash is the one to act on. `get_hash_params` omits `mesh_to_scale`,
`uniform_pts_buffer` and `subsample`, all of which change cached content — so two runs
differing only in `mesh_to_scale` produce the same key and the second silently reuses the
first's alignment and normalization. Separately, a `Mesh` object passed as `reference_mesh`
hashes via its memory address, so the cache never hits. Tracked as issue #19 (whose
fix is deliberately bundled with #27 — see the plan's State block).

---

## 5. Open items

**`nsosim` could not be surveyed.** It is not present on this machine. The plan makes fork
and `nsosim` usage an *input* to the dead-code call, which means Phase 0 cannot be closed
solo. Two of the rulings above (`refine_mesh`, `interpolate`) are the ones a mesh-oriented
consumer is most likely to reach into.

**Recommendation — split the gate.** Nothing above requires the survey except the physical
move of `train/deprecated/`. Mapping, documenting and testing a module that might later be
quarantined costs nothing; moving it costs a broken downstream. So:

- **0a (done, this document):** rulings from evidence available here → unblocks Phase 1.
- **0b (blocked on the survey):** the quarantine move only.

**The release tag still needs settling.** The plan's §10 prerequisite — tag a version, have
consumers pin it — is not yet met. `pyproject.toml` derives the version from
`NSM.__version__`, which the code-health branch bumps from `"0.0.1"` to `"0.1.0"`. A
`v0.1.0` tag exists but points at a commit on that branch rather than on `main`, so it is
not the pre-refactor rollback point it is sometimes described as. Settle which commit the
tag names before asking anyone to pin it.

**`NSM.configs` will not ship in a built distribution.** It has no `__init__.py`, so
`find_packages(include=['NSM','NSM.*'])` excludes it; there is no `package-data` or
`MANIFEST.in`. A wheel contains neither the config generator nor `default_config.json`. It
works today only because installs are editable.
