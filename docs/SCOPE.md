# NSM scope

**Phase 0 deliverable of `.claude/plans/completed/NSM_CODE_HEALTH_REFACTOR.md`.**
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

- **Adding a model type means editing NSM internals.** ~~Only `TriplanarDecoder` survives
  the reconstruction path.~~ That half is **fixed**: `reconstruct_latent` called decoders
  through a keyword-only `(latent=, xyz=)` interface with no fallback, which only
  `TriplanarDecoder` implements, while `mesh.decode_sdf` had inspected the signature and
  fallen back since before the refactor. `_decode` is that dispatch at the site that was
  missing it (§8.0.K, memoized in §8.0.R), and **measured 2026-09-22, all three advertised
  types — `triplanar`, `deepsdf` and `implicit` — fit a latent through `reconstruct_latent`
  end to end.** What remains is the part a fallback does not buy: `load_model` is a
  hardcoded `if/elif` over type names, so a model NSM does not already know about cannot be
  reached from a config at all.
  → **Phase 4 work item: a common decoder interface plus a registration pathway**, so a
  third party can add a model and have it work in train, reconstruct, mesh and interpolate
  without editing NSM internals. One calling convention has to win, and `_decode` has made
  it the permissive one.

  The `implicit` type is the furthest from usable, in two independent ways
  (audit rulings, re-verified 2026-08-22): `loader._get_implicit_params` requires
  `latent_dim`/`hidden_dim`/`num_layers` — a vocabulary no real training config uses
  (the shipped configs carry `latent_size`/`layer_dimensions`) — and both that loader
  and `ImplicitDecoder` default the output through a sigmoid, whose (0, 1) range cannot
  represent a signed distance. Unreachable from real configs, and non-SDF by default
  even when reached. **A third, measured 2026-09-03 (plan §8.0.R):**
  `_get_implicit_params` is the only one of the four translators that does not read
  `activation`, so `block_type: "linear"` builds `LinearBlockFactory()` at its `nn.ReLU`
  default whatever the config asks for — the same key both sibling translators honour.
  Fold any fix into the registration-pathway work; no half is worth patching in
  isolation. Pinned by
  `test_model_options.TestTheEvidenceForSlicesThatOwnTheFix`.
- **The shipped `default_config.json` describes only the triplanar production model.**
  PR #64 (issue #48) replaced the old 61-key DeepSDF-shaped default — which could not
  drive `train_deep_sdf` at all — with a sanitized snapshot of the ShapeMedKnee
  `647_nsm_femur_v0.0.1` triplanar config, pinned by the generator-sync test and a test
  that instantiates the trainer from the shipped file. That delivers the first of the
  per-model-type defaults.
  → **Remaining Phase 4 work item: a default config for each *other* model type**
  (deepsdf; `implicit` first needs the vocabulary reconciliation above).

**Working by design, ruled 2026-09-22:**

- **`grad_clip` clips the model's parameters and never the latent codes.** The name reads
  global and is not, so this sat in `KNOWN_ISSUES.md` § Open from Aug 2026 with an
  experiment proposed against it. Closed as correct rather than run, on three checks:
  `train_epoch` passes `model.parameters()` to `clip_grad_norm_` and nothing else, which
  **matches the upstream reference**: in `facebookresearch/DeepSDF/train_deep_sdf.py`,
  `lat_vecs` is its own optimizer param group and the only `clip_grad_norm_` call takes
  `decoder.parameters()` (re-fetched and re-read 2026-09-22 — it was at line 521, and a
  line number in someone else's repo is the part of this that will go stale first). The
  latents also already carry their own L2 regularization with warmup and their own LR
  schedule, so they are not unregularized; and the knob is `null` in the shipped default
  and in both production model configs, so nothing has ever set it. Pinned by
  `test_train_epoch.TestGradClipReachesTheModelOnly`. Clipping the latents too would
  change the numerics of every run that sets `grad_clip`, which makes it a training
  experiment rather than a defect.

**Unsupported by design, since Aug 2026 (§8.0.H):**

- **Per-layer LayerNorm in `deep_sdf.Decoder` (`norm_layers` / the `layers_with_norm`
  config key).** Introduced Jun 2023 (`01d774a`) and reachable **only** with
  `weight_norm=False`: the branch that built the norm layers was an `elif` under weight
  norm, so with weight norm on — every shipped config, and every config known to this repo
  — nothing was ever built and the key was inert. With weight norm off it worked, and then
  indexed the norm list by absolute layer index, so any set not starting at layer 0 raised
  `IndexError` on the first forward.

  The argument is deleted. **What that costs, stated plainly:** a checkpoint from a
  `weight_norm=False` run with a non-empty `layers_with_norm` carries `bn.*` keys and its
  architecture can no longer be built here. There is no shim, and the error says so —
  **pin NSM < 0.3.0 to load one.** No model this repo knows of is in that case (both
  ShapeMedKnee models and the `361_nsm_femur_cartilage` training config all set
  `weight_norm: true`), which is why deletion was preferred to carrying a half-working
  option. This is not a `KNOWN_ISSUES` § History entry: no result was ever silently wrong,
  and nothing changed numerically — a build stopped being possible, which is a scope
  question.

  The *intent* behind that commit — weight norm **and** LayerNorm together, which the
  `elif` is exactly what prevented — was never delivered and is not delivered here. It is
  new capability with an unmeasured benefit, queued as `NSM_TRAINING_IDEAS.md` Idea 14.

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

### 2.1 `train/train_deep_sdf_multi_head.py` (443 lines) — **deleted, Sep 2026 (plan §8.0.P)**

`v0.3.0` holds the last copy. [#51](https://github.com/gattia/nsm/issues/51) is closed,
and its repair checklist stays readable on the issue — the module is one `git show` away,
so copying the checklist here would duplicate a thing rather than preserve it.

The ruling moved twice before this. Aug 2026: *supported, broken, fix it*. 2026-08-29:
*unsupported until someone needs it* — the repair was never scheduled, so the old ruling
was a promise the plan could not keep, and what changed then was the promise rather than
the code. 2026-09-22: **delete**, on four facts rather than on taste.

- **It is not a model type.** It defines no class. It is a training loop taking
  `models: tuple` — N ordinary decoders against one shared latent embedding. Nothing else
  in the library knows it exists.
- **It is substantially a stale copy.** 156 of its 335 non-comment lines (**47%**) appear
  verbatim in `train_deep_sdf.py`. That is the failure mode `87c5e88` named when it merged
  the single- and multi-surface trainers in Dec 2024: *"maintaining two separate ones led
  to the one not being used falling behind"*. This is the third one, and it fell behind.
- **It has been broken since 2023 and nobody fixed it.** `train_deep_sdf` builds its
  optimizer from a leaked loop variable, so only the last decoder ever trains
  (`KNOWN_ISSUES.md` § History 2). Two identifiers would repair it. Three years passed.
- **Nobody is working on it.** Last feature work `408e78b`, 27 Jan 2025 (device
  compatibility). Every 2026 commit touching it is a refactor sweep that touched every
  file in the package.

**What is lost, stated plainly:** the multi-network-per-latent capability. Since only the
last network ever trained, no run has ever exercised it, so nothing on disk depends on it.
Reviving it means `git show v0.3.0:NSM/train/train_deep_sdf_multi_head.py` plus #51's
two-identifier fix.

**One thing went with it that is not about multi-head at all.** Its `train_epoch` was the
only function under `NSM/train/` that took `verbose=`, and carried the package's only
`@honour_verbose`, so **the deprecation bridge no longer reaches `NSM/train/`**.
`train_deep_sdf` and its `train_epoch` never took the parameter — both raise `TypeError`
on it and did so before this change, so nothing that worked stops working. Plan Step S
item (7) deletes the bridge at v0.4.0 and should count its remaining sites then rather
than trust a number written here.

### 2.2 `train/deprecated/` (880 lines) — **deleted, Sep 2026 (plan §8.0.P)**

Both files are gone, and so is the one feature that kept them alive. `v0.3.0` holds the
last copy of each.

- `train_deep_sdf_multi_surface_orig.py` (562) — strict subset of `train_deep_sdf.py`,
  nothing unique, no importer. Dead on every reading it ever got.
- `train_deep_sdf_orig.py` (318) — held the only *live* `sample_difficulty_lx` inverse-Lx
  loss-weighting branch, which is why this directory outlived its sibling rulings.
  [#18](https://github.com/gattia/nsm/issues/18) proposed porting those ~12 lines into
  `train_deep_sdf.py`; it is **closed as won't-fix** and the four config keys go with the
  file. The ruling, taken 2026-09-22 once the branch was actually read:

  - **The two curriculum components NSM advertises are both already implemented and both
    are the paper's.** Curriculum DeepSDF ([arXiv:2003.08593](https://arxiv.org/abs/2003.08593))
    has equation 5, surface accuracy (`surface_accuracy_e`), and equation 6, sample
    difficulty (`sample_difficulty_weight`). Both are live in `_surface_l1_loss`.
  - **The inverse-Lx form is not in the paper.** It has no inverse-power weighting of any
    kind. `sample_difficulty_lx` was NSM's own experiment.
  - **It was tried, and turned off.** Live in both trainers from `5188417` (Aug 2023) until
    `e173adc` (14 Feb 2024, "Remove some hard sample difficulty weight"), which commented
    it out along with `hard_sample_difficulty_power` and kept equation 6. No shipped model
    was trained with it.
  - **It did not work, and the reason is structural.** Its weight is built from the loss it
    multiplies and was never detached, so the gradient pointed the wrong way above error
    `(epsilon / (lx - 1)) ** (1 / lx)` — 0.01 at `lx=2, epsilon=1e-4`. Equation 6 is immune
    because its weight is built from `sgn`, which passes no gradient. The two branches
    Feb 2024 removed are exactly the two that lacked that property
    (`KNOWN_ISSUES.md` § History 31).

  Porting it would have meant carrying ~150 lines of code, tests and documentation for an
  untuned experiment with no users that its author had already rejected. Reviving it means
  `git show v0.3.0:NSM/train/deprecated/train_deep_sdf_orig.py`, and a `.detach()`.

What the directory demonstrated, and what survives it: **a config key can be settable, do
nothing, and say nothing, because the branch implementing it lives in a module the
supported path does not import.** The four `sample_difficulty_lx` keys were exactly that
from Feb 2024 until this release deleted them. Found by plan §8.0.N sweeping every shipped
key against every live module, and missed on that sweep's first pass because it counted the
unsupported trainer as a reader — which is now pinned by `test_default_config_sync`, whose
sweep excludes the modules `SCOPE` puts outside the documented surface.

### 2.3 `mesh/refine_mesh.py` (480 lines) — **research, keep**

Proposed: *zero importers, 0% coverage, therefore dead.*

Zero importers is confirmed. "Therefore dead" is not. `subdivide_triangles_on_base_mesh`
selects cells by metrics computed on **one** mesh and splits them on a
**different** mesh, preserving original point IDs. `pyvista.subdivide_adaptive` is present
and cannot express that base/warped split — the completed interpolation plan records that
both were tested for exactly this reason, and states in writing that the hand-built code
was kept deliberately.

Dead code is code nobody decided about. This is code someone decided to keep, in writing.

**Ruling: research. Keep, documented as such** — with three conditions, in order.
**All three executed** in plan §8.0.I (Aug 2026); the module's status line in
`ARCHITECTURE.md` §3 changed with them.

1. **Make it work.** `get_target_cells` read `np.zeros_like(max_length_binary)` where it
   meant `max_lengths`, so both public entry points raised `UnboundLocalError` on their own
   defaults. One-word fix, and it came first — documenting a module that raises describes
   something nobody can run. A test per criterion now covers the three thresholds
   separately, because it was the two *unused* ones that had never run.
2. **Warn at the entry points.** It is research code with a precondition nobody stated:
   `subdivide_triangles_on_base_mesh` computes cell indices on one mesh and applies them
   to a different one, which is valid only if the two share connectivity and cell ordering.
   Violating it produces a wrong mesh, not an error. It now warns where that is *provably*
   false — differing cell counts, or equal counts with differing face arrays. Two meshes
   that share connectivity in a different cell order are still silent, and the helper's
   docstring says so: this narrows the case rather than closing it.
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

### 2.5 `reconstruct/cartilage_func.py` (283) and `predictive_validation_class.py` (97)

Proposed: *research-only, no production caller.* Wrong on the caller half for both.

- `cartilage_func.py` is imported by the **live** trainer and wired into its
  `DICT_VALIDATION_FUNCS`, dispatched by config key
  `recon_val_func_name`. It also owns the only region-index maps in the repo
  (`CART_REGIONS`, `CART_REGIONS_DICT`). **Production.** Opened by plan §8.0.N′ in
  Aug 2026 — it was at 19% coverage with no docstring on any of its five public
  functions, and the eight defects that found are in `CHANGELOG.md` and § History 27
  below. `CART_REGIONS` is the **femur's** subregions, which is what the bare
  `compare_cart_thickness` scores when a config names it for any other joint.
- **The validators are fixed-layout by design** (maintainer ruling, 2026-08-30). They
  were built to monitor biomarkers while training the ShapeMedKnee femur bone+cartilage
  model, which is why each names its joint and requires exactly two meshes (six for
  `_whole_joint`). A model with another surface layout — e.g. the four-surface
  `["bone", "cart", "med_men", "lat_men"]` femur model the `mesh_names` docs use as
  their example — has **no** validation function until a case needs one, and gets a new
  named `DICT_VALIDATION_FUNCS` entry then. Before §8.0.N′'s length check,
  `compare_cart_thickness_femur` took the first two meshes of any list: right numbers on
  a femur-first layout by position, the femur's pair against the tibial indices on a
  whole-joint list (§ History 27). The refusal keeps the second from being silent; this
  ruling is why the first goes with it rather than being special-cased back in. Pinned
  by `test_cartilage_func.TestTheMeshListLength`.
- **The original cartilage mesh is required but not read** (ruled 2026-09-22).
  `compare_cart_thickness` takes `orig_meshes = [bone, cartilage]` but only uses the bone:
  the original's thickness comes from arrays already on the bone mesh. The cartilage slot
  stays because both lists must have the same layout. `reconstruct_mesh` calls every
  validator as `func(sampled["orig_mesh"], meshes)`, and removing the slot from one side
  would change the slicing in `_whole_joint`. Pinned by
  `test_cartilage_func.TestTheOriginalCartilageIsNeverRead`.
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

### 2.5b `None` surfaces — **supported at reconstruction, not at dataset build**

Ruled 2026-08-29. `fdfe902` (May 2025) added `None`-surface support in two files and the
two halves have opposite outcomes. Knowing which half you are in is the whole of it, and
the issue title names only the broken one.

**Supported: fitting a latent from a subset of surfaces.** Pass `sdf_gt=[bone, None, None]`
to `reconstruct_latent` and it fits against the surfaces present and decodes all of them.
This is deliberate and was carried through the §8.0.K decomposition intact:
`latent_fit.py` documents that `sdf_gt_` keeps `None` for a surface that has none, skips
it in the per-surface loss, and slices around it when chunking. This is the half in use.

> **Corrected 2026-08-29, one day after this ruling, by running it end to end.** The
> paragraph above was written from `latent_fit.py` and is right about the fit. It was
> wrong about the capability: one frame up, `compute_recon_loss` read the *original* mesh
> unguarded, so `reconstruct_mesh(path=[bone, None])` raised `AttributeError` under
> `calc_symmetric_chamfer` or `calc_assd` — both `true` in the shipped config, and both
> passed by `get_mean_errors`, the only production caller. The supported half was
> unreachable from the production entry point until plan §8.0.N′ fixed it. It is now
> pinned end to end by
> `test_reconstruct_mesh.TestASubjectMissingASurface`, which is the test this
> ruling should have had.

**Not supported: building a dataset from subjects that are missing a surface.**
`MultiSurfaceSDFSamples` cannot build one — `get_sample_data_dict` preallocates
`sum(n_pts_)` rows per combination while the sampler returns points only for the
non-`None` surfaces, so the first write raises. It has never worked. The downstream
handling that would consume such data (NaN columns in `remove_overlapping_points`, empty
index lists in `sdf_pos_neg_idx`) is ready for data the class cannot currently produce.

The two are different capabilities, not two states of one: *fit against what you have*
versus *train on subjects with holes*. The second is a plausible future need — a cohort
with unsegmented menisci is the obvious first case — so it is kept as a feature rather
than deleted: **issue #67**, with the strict `xfail` in
`test_dataset_cache.TestEmptySignedSamples::test_a_none_surface_subject_must_build`
naming it. That test is the mechanism that will report the day the build starts working.

### 2.6 Rulings not yet adjudicated

| Module | Lines | Status | What decides it |
|---|---|---|---|
| `models/loader.py` | 411 | **production — keep; the extensibility question moves to §8.1** | It is the documented entry point (README, `examples/`) *and* the natural home of the extensibility work in §1, since `load_model` is what a registration pathway would hang off. All three of its advertised model types reconstruct as of Sep 2026 (§1); what is left for §8.1 is that the type list is hardcoded here. **The open question is answered, by execution (2026-08-26): see below.** |

**Could the consumer switch to `load_model` today?** **Yes, after one edit to two files
it does not own.** Answered by running it, not by reading: both shipped
`model_params_config.json` files were loaded through `load_model` on CPU, and both models
built and forwarded (`647`: 20,801,924 parameters, output width 2; `551`: 20,801,410,
width 1).

- The consumer's 15-key mapping (`steps/run_nsm.py:94-112`) is `_get_triplanar_params`'
  dict minus `padding`, written with `[...]` where the loader uses `.get(...)`. Nothing
  else differs.
- `torch.load`: the consumer passes `weights_only=True`, `load_model` leaves torch's
  default. Checked on a `{epoch, model, optimizer}` checkpoint under torch 2.8 — all three
  settings load it identically.
- Device: `.cuda()` against `.to(device)` with device defaulting to cuda-if-available.
  Equivalent where the consumer runs, and `load_model` additionally works on CPU.
- `load_model` returns only the model, and the consumer also needs `model_config` — which
  it already reads from the JSON itself, so nothing is lost.

**The one blocker is the edit:** as of §8.0.H, `load_model` refuses a triplanar config that
omits `padding` (#26, § History 16), and **both shipped configs omit it**. Both models were
trained at the constructor default, so adding `"padding": 0.1` to each is the whole
migration — and that omission is exactly what the hand-rolled mapping was hiding. What
switching would buy: one mapping instead of two, and the refusal reaching the consumer.
What it would not fix: the surface-order contract in §3.1, which is unrelated.
| `mesh/triangle_metrics.py` | 97 | **keep — scope under investigation** | **New input, §8.0.I (Aug 2026):** it now also holds `get_faces`, the one validated face accessor all three sibling modules route through (#57) — including `interpolate`, which did not import it before, and `refine_mesh`, which used to define it and now imports the name (so `NSM.mesh.refine_mesh.get_faces` still resolves, to the same object). That makes it the `mesh/` package's leaf rather than a metrics helper, and argues for keeping it a separate file; the open question below is now only about the edge-ratio pair, not about the file. Both original importers (`correspondence_metrics`, `refine_mesh`) are themselves unreached from production, so it cannot be ruled on independently of §2.3. Two open questions: is all five of its public symbols live, or only the part `correspondence_metrics` uses; and its `areas(norm=True)` default returns a relative deviation rather than areas, which is what makes `refine_mesh`'s `area_threshold` misleading. **Keep either way** — the question is whether it stays a separate file or the live part merges into `correspondence_metrics`. Input to that merge decision, from the audit (re-verified 2026-08-22): the two modules implement the edge-ratio statistic with deliberately opposite failure behaviour — `TriangleProperties.edge_ratio` raises on a zero-length edge, `correspondence_metrics.triangle_health` degrades gracefully and reports a `degenerate_count`. A merge must reconcile that split or keep it, deliberately. |
| `datasets/utils.py` | 360 | **prod** — *ruling executed, 2026-08-22* | Was a two-line TODO proposing the Phase 4 `sdf_dataset` split, ruled dead pending that split. The split happened (§8.0 slice A, PR #71): the file now holds the 13 leaf helpers, is imported by `sdf_dataset.py` and `mesh_sampling.py`, and is one of the best-covered modules in the package. The row is kept rather than deleted because "delete when Phase 4 does the split" was a correct ruling that a reader will otherwise go looking for. |
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

For this repo the distinction turned out to be moot: both files below were *already* in
`NSM/train/deprecated/`, quarantined in Aug 2025, so Phase 1's quarantine step was a no-op
and the real decision was the one after it. **It was taken in Sep 2026 and it was delete**
(§2.2): a second quarantine adds nothing over the first, and the directory's real cost was
that it was neither live nor gone — 880 lines with no `__init__.py`, invisible to
`make test-coverage` and indistinguishable from library code to anyone reading the tree.

| | Lines |
|---|---|
| Deleted Sep 2026: `train/deprecated/train_deep_sdf_multi_surface_orig.py` | 562 |
| Deleted Sep 2026, after porting 12 lines: `train/deprecated/train_deep_sdf_orig.py` | 318 |
| ~~Delete when Phase 4 lands: `datasets/utils.py`~~ — became live code instead (§2.6) | 0 |
| **Total** | **882** |
| Plan's expectation | ~1,800 |

**No module ruled dead had zero cost to remove.** That is the finding, and it is what
made Principle 2 ("quarantine, don't delete") worth keeping as the *first* move. It does
not argue for a second one: the Sep 2026 ruling above deletes code that had already been
quarantined for a year, which is the point at which quarantine has bought everything it
can and the remaining cost is all on the reader.

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
| `losses.l1_loss`, `losses.l2_loss` | One-line re-exports of torch's functional l1/mse losses, labelled "legacy aliases"; zero callers | the eikonal repair's pass over `losses.py` (plan §8.2) |

Two rows retired 2026-08-24: `tune_reconstruction` and
`compute_correlation_coefficient` were deleted by their deferred-to pass — §8.0.E's
work over `reconstruct/main.py` (branch `wandb-optional`; CHANGELOG v0.3.0
§ Breaking), zero callers re-verified at deletion.

**Ruled kept despite zero callers:**

- `losses.compute_sdf_gradients` and `losses.combined_sdf_loss` — uncalled today, but
  they are the eikonal helper surface: `compute_sdf_gradients` carries the same
  `retain_graph` defect the eikonal repair must fix, and both stand or fall with that
  repair (plan §8.2), not with caller count. Experimental, same ruling as the eikonal
  loss itself (§1).

Two audit rulings needed no new text, verified rather than assumed: `refine_mesh`'s
cross-mesh cell-indexing precondition is already condition 2 of §2.3, and the
only-TriplanarDecoder reconstruction limit is already §1's first bullet.

### 2.9 Removed model types — **last shipped in v0.3.0, resurrectable from there**

- **`two_stage` / `TwoStageDecoder`** (`models/two_stage.py`): triplanar + MLP summed.
  Removed Sep 2026 (plan §8.0.P) on measured non-use: **zero training runs, ever** — no
  launcher script and no saved run config in the maintainer's training project
  (`nsm_femur/training_run_files/python_calls`, ~120 scripts), neither measured consumer
  imports it (kneepipeline: `TriplanarDecoder` + `reconstruct_mesh`; nsosim: five symbols,
  §5), and until [#46](https://github.com/gattia/nsm/issues/46) (Aug 2026) the class was
  not even constructible — `[latent_size + 3] + dims` on a tuple — so nothing outside this
  repo can hold a checkpoint of it. Its §8.0.O padding and norm-type repairs were correct
  and survive as prose in the tests that pin the same behaviour on the triplanar path.

  **`implicit` / `ImplicitDecoder` is not in this section, deliberately.** The same survey
  proposed removing it and the maintainer ruled on 2026-09-04 that it **stays**: it is the
  ShapeMed-Knee paper's modulated-periodic-activations baseline, so the gaps §1 records
  against it are defects against a published result rather than reasons to delete it.

**Resurrection:** `git show v0.3.0:NSM/models/two_stage.py` is the complete module as last
shipped; its loader branch (`_get_two_stage_params`), config template and tests live in the
same tag under `NSM/models/loader.py` and `testing/NSM/`. Reviving it means re-adding those plus the Phase-4
registration pathway (§1) that removal pre-empted.

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
   2026-08-27): when a fit has fewer ground-truth surfaces than the decoder has outputs,
   `latent_fit._recon_loss` `break`s out of the surface loop under an in-code TODO that says
   outright "it assumes the first surface is the bone / only of interest". Since §8.0.K it
   is no longer *silent*: that break and the `None`-ground-truth `continue` beside it log at
   `warning` to whatever the host configured. Before, they were hidden unless `verbose`
   was set (a flag removed in v0.4.0).
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
   *Half-closed 2026-08-26 (§8.0.H, #26): `load_model` now refuses a triplanar config that
   omits `padding`, so the value can no longer be silently defaulted on **that** path. The
   consumer's own path is untouched, because it never calls `load_model` — §2.6 above
   establishes by execution that it could, and what the one prerequisite is.*

*`reconstruct_mesh` used to have **one executed line** in the entire test suite: its
`def`. Stale since §8.0.C ran the single-object sampled branch end to end and §8.0.J
(2026-08-27) added the stage contracts —* `python -m coverage run --source=NSM -m pytest
testing/NSM/reconstruct` *puts `reconstruct/main.py` at 85%. The reason it was worth
recording still stands, though: none of that is a fit against a real trained decoder, so
what the tests cover is the plumbing, not the reconstruction.*

**Deprecated, with a delete-when (audit ruling, re-verified 2026-08-22):**
`batch_size_latent_recon`. `reconstruct_mesh` dropped the parameter, absorbs it via
`**kwargs` — the only key it still accepts there, since §8.0.J refuses the rest — and
logs a deprecation warning on every call, while the consumer still passes it
(`steps/run_nsm.py`) and `recon_evaluation.get_mean_errors` still takes it as a real
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

**internal — everything else.** The leak this describes was re-measured 2026-08-29, after
the §8.0.C/E/G splits moved the surface the original count was taken on: `NSM.reconstruct`
binds **43** public names of which 11 are modules (`wandb`, `torch`, `time`, `logging`…),
`NSM.datasets` 43 of which **17** are, `NSM.models` 26 of which 11 are, `NSM.mesh` 24 of
which 9 are. `NSM.train` was already clean. Since v0.3.0 each subpackage declares `__all__`,
so a star-import binds only NSM's own names — see §3.3. Attribute access is unchanged:
`NSM.datasets.torch` still resolves, and removing that would mean replacing the star
re-exports themselves.

### 3.3 Why `__all__` is per subpackage and not in `NSM/__init__.py`

**Shipped in v0.3.0** (plan §8.0.O): `NSM.datasets`, `NSM.mesh`, `NSM.models`,
`NSM.reconstruct` and `NSM.train` each declare `__all__`, naming every name they bind that
is defined in them, submodules included. The rule is mechanical, so nothing is on a list by
opinion; the stability tiering of §3.2 is a separate ruling and has not been applied to
them. Pinned by `testing/NSM/test_packaging.py::TestPublicApiDeclaration`, which asserts
that every declared name resolves and that none of them is foreign. Python then makes
`from <pkg> import *` bind exactly the declaration.

The plan's Phase 0 deliverable said "an `__all__` in `NSM/__init__.py`". As specified that
could not be done, and the reason is why it went per-subpackage instead.

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

**What was done:** `__all__` in each subpackage `__init__.py`, which is where the leakage
actually is, and the top level left lazy. PEP 562 `__getattr__`, so that `NSM.models`
resolves from a bare `import NSM`, was **not** done — it would reintroduce the eager import
above the first time anyone touched the attribute, and no consumer has asked for it.

**What `__all__` does not do, stated once so nobody concludes otherwise:** it controls
`from X import *` and states intent. It does not unbind anything — `NSM.datasets.torch`
still resolves. Removing those bindings means replacing the star re-exports themselves,
which is a larger change with a real chance of breaking a fork, and no evidence yet that
anyone is hurt by them.

---

## 4. Format contracts

These are public interfaces even though nothing imports them. Changing any of them breaks
consumers silently.

| Artifact | Written by | Read by | Versioned? |
|---|---|---|---|
| `model_params_config.json` | `utils.save_model_params` — **first write wins**, and a later checkpoint whose config disagrees warns rather than rewriting (#50, §8.0.M) | `load_model`, `examples/load_trained_model.py`, **both consumer scripts (hand-rolled)** | No |
| checkpoint `{epoch, model, optimizer}` | `utils.save_model` | `loader.py` (4 possible key layouts), consumer `load_state_dict` | Pre-Aug-2026 refused at load |
| `latent_codes/{epoch}.pth` | `utils.save_latent_vectors` | `train_deep_sdf` on resume | No |
| `LearningRateSchedule[].Target` | config author | `utils.resolve_schedule_targets` | Yes — missing key raises with a migration message |
| SDF cache `.npz` | `sdf_dataset` | `sdf_dataset` | **Yes** — `cache_format` entry inside the key (Aug 2026) |
| `NSM_recon_params.json` → `latent` | consumer, from `reconstruct_mesh` | `steps/compute_bscore.py:72` | No |

The dataset cache key was rewritten Aug 2026 (#19; `KNOWN_ISSUES.md` § History 13): a
named canonical mapping with content-stable mesh identities, versioned by a
`cache_format` entry — the next change to what gets cached is one integer bump, not a
new hashing scheme, and no pre-rewrite key can hit again. (This row previously also
listed `.h5`: stale — no h5 path exists in `datasets/`; the cache is one `.npz` per
subject.)

---

## 5. Open items

**~~`nsosim` could not be surveyed.~~ Surveyed 2026-08-30, from a fresh clone of
`gattia/nsosim`.** Its entire NSM surface is five symbols in three subpackages:
`NSM.mesh.create_mesh`, `NSM.mesh.interpolate.interpolate_points` (plus
`interpolate_mesh` in one notebook), `NSM.models.Decoder` / `TriplanarDecoder`, and
`NSM.reconstruct.reconstruct_mesh`. Zero references to `NSM.train`, `train_deep_sdf` or
`sample_difficulty` anywhere — package, notebooks, scripts, tests — and its
`requirements-lock.txt` pins an editable `nsm@b7cfd49`. Inference-only, which the
maintainer stated and the sweep confirms. This paragraph's original guess — that a
mesh-oriented consumer most likely reaches into `refine_mesh` and `interpolate` — was
half right: `interpolate` yes, at both entry points; `refine_mesh` no. **The move this
gated is `NSM_CODE_HEALTH_REFACTOR.md` §8.0.P, ungated in Aug 2026 and executed in
Sep 2026 as a delete rather than a second quarantine (§2.2).**

**Recommendation — split the gate.** Nothing above requires the survey except the physical
move of `train/deprecated/`. Mapping, documenting and testing a module that might later be
quarantined costs nothing; moving it costs a broken downstream. So:

- **0a (done, this document):** rulings from evidence available here → unblocks Phase 1.
- **0b (was blocked on the survey, now done):** the removal of `train/deprecated/` only.

**The release tag no longer needs settling, and the mechanism it depended on is gone.**
From v0.3.0 `pyproject.toml` derives the version from the git tag via setuptools-scm, and
`NSM.__version__` reads the installed distribution's metadata — there is no literal for
anyone to forget. What remains true and worth knowing: the `v0.1.0` tag points at a commit
on the code-health branch rather than on `main`, so it is not the pre-refactor rollback
point it is sometimes described as. `v0.2.0` and `v0.3.0` are both on `main`.

**~~`NSM.configs` will not ship in a built distribution.~~ Half wrong, and fixed at
v0.3.0.** Measured by building a wheel from a clean `git archive`: it contained
`NSM/configs/generate_sdf_default_config.py` and **not** `default_config.json`.
`[tool.setuptools.packages.find]` takes `namespaces = true` by default in `pyproject.toml`,
so `NSM.configs` is found as a namespace package despite having no `__init__.py` — the
generator has been shipping all along. What was missing was package *data*, there being no
`package-data` and no `MANIFEST.in`, so the JSON worked only because installs were
editable. `pyproject.toml` now declares it, and
`testing/NSM/test_packaging.py::TestWhatShips` builds a wheel and byte-compares the shipped
copy. **The claim above was inferred from `find_packages` semantics and never run** — and
the half that would have mattered, whether the generator survives, is the half that was
fine.
