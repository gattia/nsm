# Changelog

Notable changes to NSM, newest first.

**What belongs here:** anything that changes what a caller must write, what an existing
call does, or what an existing result would be. Internal refactors, test coverage and
documentation do not, unless they change one of those three.

**Breaking changes lead each release**, because that is the only section a reader upgrading
across versions has to act on. NSM is pre-1.0, so a breaking change bumps the **minor**.

**When this is updated:** as part of the release, in the PR that renames § Unreleased to
the version being cut, immediately before the tag. Nothing in that PR bumps a version
number: from v0.3.0 the version *is* the tag, derived by setuptools-scm, so cutting the tag
is what publishes it. A release with no changelog entry is a release nobody downstream can
evaluate.

**Related, and deliberately separate:** `docs/KNOWN_ISSUES.md` § History answers *"is a run
I already have on disk affected?"* — that is about results. This file answers *"does my
code still work?"* — that is about API. A change can warrant an entry in both.

**Installing a specific version.** NSM is not on PyPI (the `nsm` name there is an unrelated
package), so pin the tag directly:

```
nsm @ git+https://github.com/gattia/nsm@v0.3.0
```

---

## Unreleased

### Breaking

- **`roundtrip_distance` and `forward_backward_disagreement` are keyword-only**
  ([#56](https://github.com/gattia/nsm/issues/56), plan §8.0.N). The two take *the same two arrays in opposite order*, forty
  lines apart in `mesh/correspondence_metrics.py`, and neither swap was visible in what a
  caller reads: `roundtrip_distance` is `norm(rt - orig)`, symmetric, so swapping it is a
  no-op; `forward_backward_disagreement` sign-flips `field` and leaves
  `magnitude_percentiles` identical. A positional call is now a `TypeError`. The values
  are unchanged. `directed_distance_percentiles` is **not** included and keeps its
  positional signature — it is asymmetric and documented as directional, so a swap there
  changes the number rather than hiding.

- **`SDFSamples.load_mesh_step` no longer takes `verbose`** (plan §8.0.N). Its last reader
  was a `logger.debug` gated on the flag; ungating that left the parameter accepted and
  never read, which is the trap this project deletes rather than keeps. The internal
  `_process_meshes_for_wandb` lost the same parameter for the same reason. Both are called
  only from inside NSM, under public entry points that still accept and honour `verbose`.

- **The cartilage validation functions check their mesh list's length** (plan §8.0.N′).
  `compare_cart_thickness_tibia`, `_patella` and `_femur` each took `orig_meshes[:2]` and
  nothing checked what they sliced from: measured, a six-mesh whole-joint list into
  `compare_cart_thickness_tibia` scored the **femur's** pair against the tibial region
  indices and returned NaN for every one, exit 0. The three now require exactly two meshes
  and `compare_cart_thickness_whole_joint` exactly six, with the layout each assumes in
  the message. A two-surface run — both shipped ShapeMedKnee configs — is unaffected, and
  a whole-joint list into the wrong single-joint wrapper was already producing NaN
  (`docs/KNOWN_ISSUES.md` § History 27). One previously *working* shape is removed with
  them: `compare_cart_thickness_femur` on a femur-first list of more than two surfaces
  sliced the correct pair and scored it. Ruled fixed-layout by design (maintainer,
  2026-08-30): these validators were built to monitor the ShapeMedKnee femur
  bone+cartilage training runs, and a model with another surface layout gets its own
  named `DICT_VALIDATION_FUNCS` entry when a case needs one — `docs/SCOPE.md` §2.5.

- **`regions_label` is refused unless it is `"labels"`** (plan §8.0.N′). It was honoured at
  the scalar transfer and ignored at the read — pymskt's `get_cart_thickness_mean` and
  `_std` both open `get_scalar("labels")` with the name hardcoded — so every other value
  already raised `KeyError: 'labels'`, from the original's side or the reconstruction's
  depending on which arrays the original carried. This only decides which exception and
  whether it says why. Deleting the parameter is scheduled for v0.4.0 (plan §8.0.S).

### Changed

- **`chamfer_norm` and `sigma_rand_pts` have one default each** ([#56](https://github.com/gattia/nsm/issues/56), plan
  §8.0.N). `compute_recon_loss`'s `chamfer_norm` goes 1 → **2**, matching `get_mean_errors`
  and `reconstruct_mesh`; `reconstruct_mesh`'s `sigma_rand_pts` goes 0.001 → **0.01**,
  matching `get_mean_errors` and the shipped config's `sigma_rand_pts_recon`. Both take the
  value the ShapeMedKnee configuration already uses.

  `chamfer_norm` changes no result on any NSM path: `train_deep_sdf.py` passes the argument
  commented out and no shipped config carries the key, so every run has always taken
  `get_mean_errors`' 2 and forwarded it explicitly. `sigma_rand_pts` **can** change a
  result — see `docs/KNOWN_ISSUES.md` § History 29. `compute_chamfer(power=1)` is the same
  knob under a fourth name and deliberately does not move: it is a generic distance helper,
  1 is the textbook Chamfer definition, and its one NSM caller always passes the value
  explicitly.

- **Eight keys in the shipped `default_config.json`** (plan §8.0.N, §6.1). NSM never builds
  a dataset from a config — `train_deep_sdf(config, model, sdf_dataset)` takes one already
  built — so the dataset half of that file is a specification the caller translates into
  `MultiSurfaceSDFSamples` arguments by hand, and six keys spelled that parameter
  differently from the constructor while eighteen others did not. Renamed to the
  constructor's spelling, values unchanged: `n_pts_per_object` → `n_pts`,
  `percent_near_surface` → `p_near_surface`, `percent_further_from_surface` →
  `p_further_from_surface`, `random_function` → `rand_function`, `normalize_pts` →
  `norm_pts`, `dataset_uniform_pts_buffer` → `uniform_pts_buffer`. `decoder_type` and
  `sdf_skip_connection` named nothing on either `model_type` path and are deleted. Nothing
  in NSM reads either spelling, so no run changes; a config on disk keeps working, and
  § History 6 says which spelling it will carry.

- **The `verbose` flag no longer suppresses log records** (plan §8.0.N, completing §8.0.G).
  Fifty-eight `logger.*` calls across eight modules sat inside `if verbose:`, so they were
  gated twice — once by the level the host configured and once by a parameter the host does
  not know exists. A host that configured `DEBUG` and asked for NSM's output now gets it.
  Nothing new is emitted; what changes is that it arrives — **and not only at `DEBUG`**:
  9 of the 58 are `WARNING` and 2 are `INFO`, so a host at default levels sees new output
  too. The loud case is `mesh/main._finish_meshes`, which logs four warnings per surface
  whose SDF does not cross zero — the ordinary state of a decoder early in training, which
  is when validation runs. Those warnings were always *deserved* at their level (a missing
  surface is a finding, not a trace); what was wrong before is that a host had no way to
  receive them without a flag it could not see.

### Fixed

- **`compute_recon_loss` no longer downcasts the caller's meshes — a mixed-dtype pair is
  aligned on copies instead** ([#55](https://github.com/gattia/nsm/issues/55), plan §8.0.N;
  corrected by plan §7.5a's compatibility check). Under `calc_assd=True` it set
  `point_coords` to `float32` in place on both the reconstruction and the caller's ground
  truth, under a comment reading "make sure the points for the meshes are the same types".
  The mutation is gone, but the comment's reason was real: mskt 0.1.19 (the knee-pipeline
  production environment) hands both dtypes straight to `point_cloud_utils`, which raises
  `ValueError` on a mixed pair — and the production path always produces one, because
  the knee pipeline's input meshes carry float64 points on disk while the reconstruction
  arrives float32 from marching cubes. Deleting the cast outright (the first form of this change) broke every
  production `calc_assd` call for the one day `main` carried it. A mixed pair is now
  upcast to float64 on copies: the caller's meshes stay untouched, the upcast is exact,
  and the value matches what mskt 0.1.21 (which casts inside `pcu_sdf`) computes. See
  `docs/KNOWN_ISSUES.md` § History 28 for who is affected.

- **The `sdf_gt` preprocess no longer rewrites the caller's list** ([#55](https://github.com/gattia/nsm/issues/55), plan
  §8.0.N). `reconstruct_latent_preprocess_sdf_gt` assigned the clamped, device-moved
  tensors back into the list it was handed and returned that list;
  `reconstruct_latent_sdf_gt_type_check` passed the caller's object straight through. Both
  build a new list now. The values are unchanged. This also makes the `tuple` the type
  check has always advertised usable — the index assignment raised `TypeError: 'tuple'
  object does not support item assignment` at every call before it.

- **`interpolate_mesh` and `interpolate_points` state whether they mutate**
  ([#55](https://github.com/gattia/nsm/issues/55), plan §8.0.N). `interpolate_mesh` advances the caller's mesh in place and
  returns that object, by design; `interpolate_points` does not touch its input and returns
  a new array. The contract was written only on the private engine both call. No behaviour
  change.

- **A surface the decoder did not produce no longer kills the process** (plan §8.0.N′).
  `create_mesh` leaves a surface's slot `None` when its SDF does not cross zero — the
  ordinary state of a decoder early in training, which is when validation runs — and
  `compare_cart_thickness` passed that `None` to `CartilageMesh`, which builds a 0-point
  mesh that `vtkOBBTree` dies on: measured, **exit 139, `SIGSEGV`**, uncatchable, taking
  the training run with it. It is now scored `np.nan` across the whole key set, which is
  what `compute_recon_loss` has always done with the same input.

- **A subject missing a structure can be scored** (plan §8.0.N′, `docs/SCOPE.md` §2.5b).
  `compute_recon_loss` guarded the reconstructed mesh against `None` and read the original
  unguarded on the next line, so `calc_symmetric_chamfer` or `calc_assd` raised
  `AttributeError: 'NoneType' object has no attribute 'point_coords'`. Both are `true` in
  the shipped `default_config.json`, so the capability §2.5b rules supported was
  unreachable through `get_mean_errors`, the only production caller. The missing surface
  now scores `nan` and the others score normally.

- **A degenerate first validation subject no longer loses the validation function's
  metrics** (plan §8.0.N′). `get_mean_errors` created each `func_` key's list only on
  subject 0, which contributes none when its reconstruction has no zero level set, so the
  next subject with results raised `KeyError`. It fired on the order of the validation
  set.

---

## v0.3.0

The first release since the Phase 4 decomposition began, and the largest break in NSM's
history: 37 breaking changes among 83 entries, against 278 commits. Almost all of them are
the same kind of change — a configuration NSM used to accept and quietly reinterpret now
either works or refuses at the point it is stated.

Three things a reader upgrading should know before the list:

- **A config written before Aug 2026 will be refused, and one message says everything it
  needs.** The three triplanar architecture keys (`padding`, `conv_activation`,
  `conv_norm_type`) are required and cannot be recovered from a checkpoint. Both shipped
  ShapeMedKnee configs omit two of them; `docs/KNOWN_ISSUES.md` § Packaging says what to
  add.
- **The version is now the git tag.** `NSM.__version__` reads the installed
  distribution's metadata instead of a literal that said `0.2.0` for 269 commits.
- **`docs/KNOWN_ISSUES.md` § History is the other half of this file.** This one answers
  "does my code still work"; that one answers "is a result I already have on disk wrong".
  Its 26 entries are what to read before trusting a result produced before Aug 2026.

### Breaking

- **`TriplanarDecoder` and `VAEDecoder` default their normalization to `"layer"`**
  (plan §8.0.O), where the signature said `"batch"`. `"layer"` is what every ShapeMedKnee
  config, `default_config.json` and `two_stage`'s own defaults train with. Since §8.0.H
  made `load_model` require `conv_norm_type`, the old default was unreachable from a
  config and reachable only by direct construction — which is what the downstream consumer
  does, and it passes the key explicitly, so nothing it builds moves. What the old default
  cost was a model built by hand without stating the key: it trained nonlinear and
  evaluated affine (`docs/ARCHITECTURE.md` §7.1). A checkpoint mismatch stays loud in
  either direction, `BatchNorm2d`'s running statistics being a strict superset of
  `LayerNorm`'s keys.

- **A `two_stage` config must state `padding`, and a `padding` it states now reaches the
  decoder** (plan §8.0.O, #26). `_get_two_stage_params` built its `triplanar_params`
  without a `padding` key at all, so `TriplanarDecoder`'s constructor default won:
  measured, `padding: 0.35` in the config produced a decoder at `0.1`. That is #26's
  defect — a coordinate scale that is not a learned parameter, so `load_state_dict`
  cannot contradict it — in the branch §8.0.H did not open, and in its worse form, the key
  being present and discarded rather than absent. The nested `triplanar_params` form is
  held to the same three keys, `two_stage.py`'s module-level default states `padding`, and
  so does the `two_stage` config template. **Nothing existing is affected:** both shipped
  models are `model_type: "triplanar"`, the shipped `default_config.json` is triplanar with
  `padding` stated, and no `two_stage` config exists in this repo or in either consumer.

- **Every subpackage declares `__all__`** (plan §8.0.O, `docs/SCOPE.md` §3.3), so
  `from NSM.models import *` — and the same for `datasets`, `mesh`, `reconstruct`, `train` —
  binds only NSM's own names. It used to bind everything the package's source had imported:
  `torch`, `os`, `wandb`, `logging`, `numpy`, `Pool`, `datetime`. Attribute access is
  unchanged, `NSM.datasets.torch` included; only the star-import surface narrows. This is
  not yet the stability tiering of `SCOPE.md` §3.2, which is a separate ruling.


- **`utils.print_gpu_memory` is deleted** (plan §8.0.M). §8.0.G converted its four `print`
  calls to `logger.info` and left the name saying otherwise. Swept across `NSM/`,
  `testing/`, `examples/` and `docs/`, its only occurrence was its own definition, so a
  rename would have kept a zero-caller helper alive under a truer name. `clear_gpu_cache`
  beside it has two callers and is unchanged.

- **`LearningRateSchedule.get_learning_rate` raises `NotImplementedError`** (plan §8.0.M)
  instead of returning `None`. Only a subclass that never overrides it is affected, and
  such a subclass could not work: the `None` was assigned into `param_group["lr"]` and
  surfaced two calls later at `optimizer.step()` as `unsupported operand type(s) for /:
  'NoneType' and 'float'`, naming neither the schedule, the group, nor the config entry.


- **`reconstruct_latent` refuses a keyword it does not recognise, and refuses a value it
  cannot use** (plan §8.0.K). It takes 38 named parameters and a `**kwargs` that read
  exactly one key, `max_batch_size`; seven of seven misspellings of real parameters ran a
  whole fit with the intended parameter's default and no signal. It now raises `TypeError`
  naming the unknown key — the same refusal §8.0.J gave `reconstruct_mesh`, from one shared
  helper, `reconstruct.utils.refuse_unknown_kwargs`. `max_batch_size` is unchanged, still
  accepted and still warned about. See `docs/KNOWN_ISSUES.md` § History 20.

  **Migrating a config that now raises:**
  `NSM.reconstruct._config_migration.migrate_reconstruct_config(config)` returns a
  corrected copy and a list of what it removed and why. Every key it drops was inert
  before the refusal as well as after — three that named nothing in NSM at any point
  (`min_rel_improve`, `grad_tol`, `param_change_tol`), one that `reconstruct_mesh` never
  forwarded (`log_wandb_step`), and `latent_optimizer_name` when `hybrid_optimizer` is
  set, which the loop does not read — so migrating cannot move a number. The `TypeError`
  names the helper when it recognises one of those keys, and stays quiet for a plain typo.

  Three string parameters are **case-folded and then refused**, where all three had an
  `if`/`elif` chain with no `else`: `optimizer_name` (`{"adam", "lbfgs"}`), `loss_type`
  (`{"l1", "l1_log", "l2"}`) and `convergence` (`{"num_iterations", "overall_loss",
  "recon_loss"}`). `"Adam"`, `"L1"` and `"Recon_Loss"` now work — NSM's training path
  spells its own optimizers `"Adam"` / `"AdamW"`, so refusing that spelling here would be
  the library disagreeing with itself — and anything else raises `ValueError` naming the
  parameter and its accepted values. Folding is not the same as accepting two spellings:
  every branch downstream reads the normalised value.

  `convergence` is the one that changes a result rather than an error. Its missing `else`
  was the *default branch*, so any unrecognised value — `None` and `""` included — was
  accepted and silently meant `"num_iterations"`, turning convergence checking off.
  § History 23.

  And `hybrid_optimizer=True` with a non-default `optimizer_name` raises rather than
  silently discarding the second: hybrid mode derives its optimizer from the step number,
  so `optimizer_name` was accepted and never read. § History 22.

- **`reconstruct_mesh` refuses a keyword it does not recognise** (plan §8.0.J). It takes 58
  named parameters and a `**kwargs` that read exactly one key, `batch_size_latent_recon`;
  every other key was swallowed, so a misspelling ran with the intended parameter's default
  and reported nothing. It now raises `TypeError` naming the unknown key.
  `batch_size_latent_recon` is unchanged — still accepted, still warned about.
  A caller passing anything else the signature does not name was already being ignored;
  see `docs/KNOWN_ISSUES.md` § History 20 for whether a result you have is affected.

- **`mesh/` refuses a face array it cannot read, where it used to reshape past it**
  ([#57](https://github.com/gattia/nsm/issues/57)). Five sites — `self_intersection_count`,
  `foldover_count`, `refine_mesh.get_faces`, `build_mesh_laplacian`, `compute_feature_mask`
  — go through `triangle_metrics.get_faces`, which takes a triangle mesh or an
  (M, 3) array and raises a `ValueError` naming what to pass otherwise. That function
  moved out of `refine_mesh`, which now imports the name, so
  `NSM.mesh.refine_mesh.get_faces` is unchanged as an import path. For an all-triangle
  mesh nothing changes: `regular_faces` is element-equal to the `faces.reshape(-1, 4)[:, 1:]`
  it replaces. What changes is the input that used to *succeed* wrongly — a quad mesh whose
  cell count divides by 4, or a VTK-style flat array handed to `interpolate_points(faces=)`.
  See `docs/KNOWN_ISSUES.md` § History 17 for whether a result you have is affected.

- **`score_correspondence` skips the round-trip metrics when `source_mesh` is absent**
  ([#54](https://github.com/gattia/nsm/issues/54)). `roundtrip_distance` and
  `forward_backward_disagreement` used to measure against the *warped* mesh instead, which
  is a different quantity: 0.2500 where the true answer was 0.0017. Both keys now return
  `{"skipped": True, "reason": "source_mesh not provided"}`, which is what every other
  optional-input metric in the same dict already did. Callers reading those keys must
  handle the skip shape they already had to handle for `foldover_count`. § History 18.

- **`sdf_grid_to_mesh` now accepts numpy and defaults `narrow_band=True`**
  ([#60](https://github.com/gattia/nsm/issues/60)), matching its `sdf_grid_to_mesh_vtk`
  twin — `use_vtk` at the callers selects an extraction backend and no longer also selects
  an accepted input type and a cropping policy. Geometry is unchanged: cropping to the band
  moves vertices by ~7e-08 against a 0.065 voxel.

- **`create_mesh_adaptive(voxel_origin=)` defaults to `None`, meaning "take it from
  `search_bounds`"** ([#60](https://github.com/gattia/nsm/issues/60)). It used to default to
  `(-1, -1, -1)` independently of `search_bounds`, so the no-surface fallback built a grid
  that did not cover the region the caller asked to search. At the default `search_bounds`
  the derived value *is* `(-1, -1, -1)`, so no run at NSM's defaults changes; an explicit
  origin still wins. § History 19.

- **`subdivide_triangles_on_base_mesh` warns when `base_mesh` and `mesh` are shown not to
  share connectivity.** Cell indices selected on one are applied to the other, which is
  only meaningful if the two tessellations match; violating it produced a wrong mesh and no
  error. `docs/SCOPE.md` §2.3.

- **There is one `Sine`, and `NSM.models.Sine` is unambiguously it.** `deep_sdf` defined a
  second one with `w0` hardcoded to 30 and its initializer misspelled `__init` (so
  name-mangled, and never run), and `NSM/models/__init__.py`'s `from .deep_sdf import *`
  runs before the explicit imports, so `NSM.models.Sine` silently meant that one rather
  than `modulated_periodic_activations.Sine`, which takes `w0` and defaults it to 1.0.
  `deep_sdf` now imports the parameterized class and `get_activation("sin")` returns
  `Sine(w0=30)`. **No arithmetic changes** — both computed `sin(30 * x)` — and any code
  that constructed `NSM.models.Sine()` positionally now gets `w0=1.0` instead of 30, so
  pass `w0=30` explicitly if that was the intent.

- **Four arguments that were accepted and never read are deleted from `models/`**
  ([#20](https://github.com/gattia/nsm/issues/20)):
  `TriplanarDecoder.normalize_coordinates(padding=)` (the body reads `self.padding`, and
  its sole caller depends on that — honouring the argument instead would have handed it the
  0.1 default in place of a model's trained value), `Decoder(xyz_in_all=)` and
  `Decoder(latent_noise_sigma=)` (documented, never read — no run has ever had either), and
  `VAEDecoder(activation=)` (the module it selected was built and never appended, so the
  argument chose between two things neither of which ran). The dead function
  `deep_sdf.weight_norm_all` goes with them. **`Decoder` keeps `**kwargs`**, so
  `xyz_in_all` and `latent_noise_sigma` set to something truthy now raise a `TypeError`
  rather than being silently ignored a second time; falsy values — what every NSM-owned
  config carries — are still accepted. `get_model_config_template` no longer advertises
  either key; `default_config.json` keeps `xyz_in_all: false`, which is inert.

- **`conv_activation` exists, and is a required key for `load_model` on the `triplanar` and
  `two_stage` branches.** `VAEDecoder` built a pointwise activation and never appended it,
  from the first triplanar commit (Aug 2023) onwards, so the conv stack has never had one and
  the only nonlinearity in the feature-plane generator is the final `Tanh`
  (`ARCHITECTURE.md` §7.1). `conv_activation` now selects it, through the same vocabulary as
  `get_activation` — and **`null` is the default and the historical architecture**, because
  `nn.Sequential` names children by position, so inserting a parameterless activation
  renumbers every later state-dict key. `null` builds a byte-identical module list and loads
  every existing checkpoint; any other value builds a layout no existing checkpoint fits, and
  says so with `Missing key(s)`. The key is required rather than defaulted so a config
  describes exactly one architecture. **Migration: add `"conv_activation": null`** — every
  model trained before Aug 2026 is that one, including both shipped ShapeMedKnee models.
  *Placement is `conv → norm → activation` and is provisional until the retrain in
  `NSM_TRAINING_IDEAS.md` Idea 13; a naive drop-in measured worse on the synthetic harness.*

- **`conv_norm_type` is a required key for `load_model` on the `triplanar` and `two_stage`
  branches, and the templates now say `"layer"`.** Four places defaulted it and disagreed:
  `"batch"` in the `VAEDecoder`/`TriplanarDecoder` signatures, `_get_triplanar_params` and
  the triplanar template, against `"layer"` in `_get_two_stage_params`,
  `two_stage`'s default triplanar params and `NSM/configs/default_config.json`. **`"layer"` is
  the only value ever trained** — 647, 551 and `ShapeMedKnee_2024_config.json` all use it —
  and it is not cosmetic: it is the only thing making the VAE nonlinear at all, since the
  pointwise activation was never wired in (`ARCHITECTURE.md` §7.1). Under `"batch"` the
  stack trains nonlinear and evaluates affine. Every config on disk already states the key,
  so nothing that worked stops working; what stops is a **fresh** run started from
  `get_model_config_template("triplanar")` silently inheriting a configuration nobody has
  trained. A mismatch against an existing checkpoint was never silent — `BatchNorm2d` and
  `LayerNorm` differ in key set and shape, so torch already refused it. The constructor
  signatures keep `"batch"`; changing them is breaking for a public-stable class.

- **`padding` is a required key for `load_model(..., model_type="triplanar")`**
  ([#26](https://github.com/gattia/nsm/issues/26)). It is not a learned parameter, so a
  checkpoint trained at one value loaded cleanly at `load_model`'s 0.1 default and sampled
  the feature planes at the wrong scale — measured at 0.063 max SDF difference on a
  `tanh`-bounded output. A config that omits it now raises `KeyError` naming the value to
  write. **Configs written before Aug 2026 omit the key**, including the shipped
  `647_nsm_femur_v0.0.1` one; those models ran at the constructor default, so adding
  `"padding": 0.1` reproduces them exactly and is the whole migration.

- **`sum_conv_output_features: false` now uses all three feature planes**
  ([#45](https://github.com/gattia/nsm/issues/45)). It sliced `sdf_latent_size` per plane
  while sizing the VAE for the concatenation, so yz and xy got zero-channel slices and the
  output equalled the xz plane alone — silently, with every VAE parameter still receiving
  gradient. Each plane now takes `sdf_latent_size // 3`. **The VAE's output width is
  unchanged, so pre-fix checkpoints still load** and then compute something different;
  `KNOWN_ISSUES.md` § History 15 says how to tell whether a run of yours is affected.
  `conv_pred_sdf: true` with `sum_conv_output_features: false` now refuses at construction
  — the combination always died on the first forward — and the divisibility guard is a
  `ValueError` rather than an `assert`, so `python -O` cannot strip it.

- **`Decoder` no longer takes `norm_layers`** ([#46](https://github.com/gattia/nsm/issues/46)).
  The branch that built the LayerNorms was an `elif` under `weight_norm`, so the option was
  reachable **only with weight norm off** — and it then indexed the norm list by absolute
  layer index, raising `IndexError` for any set not starting at layer 0.
  **`weight_norm: true` configs are unaffected and still load**: nothing was ever built in
  that branch, so the key was provably a no-op, and a config carrying it gets a logged
  warning rather than an error. **`weight_norm: false` with a non-empty `layers_with_norm`
  is refused**: LayerNorm really was applied there and the checkpoint carries `bn.*` keys,
  so that architecture can no longer be built — pin NSM < 0.3.0 to load one. No shipped
  model is in that case; both ShapeMedKnee configs set `weight_norm: true`.
  `default_config.json` no longer ships the key, and `load_model` keeps mapping it for the
  sole purpose of reaching those two responses.

  *Not fixed here, deliberately:* commit `01d774a` (Jun 2023) introduced this branch with
  the message "separate wieght norm and batch norm **so can use both**", which the `elif`
  is precisely what prevents. Delivering that is new capability — it would add LayerNorm to
  every model built from a config setting both, `default_config.json` included — so it
  needs an explicit opt-in and a version boundary rather than a refactor commit.

- **`layer_split: false` now means no layer split** ([#46](https://github.com/gattia/nsm/issues/46)).
  `Decoder` tested `self.layer_split is not None`, and `False is not None`, so the value
  `default_config.json` ships selected a split at layer 0: every state-dict key moved from
  `layers.N.weight` to `layers.N.0.weight`, and with `objects_per_decoder > 1` the output
  head changed shape as well. A `deepsdf` checkpoint built from such a config no longer
  loads into a model built from the same config — loudly, with `Missing key(s)` — and
  **passing `layer_split=0` explicitly rebuilds the original architecture**. `0` still
  means split at layer 0; only `False` is reinterpreted, since `False == 0` leaves nothing
  but an identity check to tell them apart. No triplanar model is affected:
  `TriplanarDecoder` builds its inner `Decoder` with `layer_split=None`.

- **`Decoder(activation='linear')` refuses at construction** ([#46](https://github.com/gattia/nsm/issues/46)).
  `get_activation` returns `None` for `'linear'`, which is correct for the final position
  and fatal for the hidden one, where `forward` then called `None`. It now raises a
  `ValueError` naming `final_activation='linear'` as the position where `'linear'` is
  supported. Nothing that worked stops working: the configuration always died on the first
  forward pass.

- **The checkpoint format changes: `VAEDecoder` tensors appear once, not twice**
  ([#27](https://github.com/gattia/nsm/issues/27)). Until now every VAE layer was
  registered in both `self.layers` and `self.decoder`, so `state_dict()` emitted each
  tensor under two aliased names and checkpoints were 1.92× their parameter count
  (the shipped 275 MB models would be ~143 MB). `self.decoder` is now the single
  registration. **Old checkpoints keep loading**, through `load_model` and bare
  `model.load_state_dict(strict=True)` alike — a permanent load-time hook on
  `VAEDecoder` drops the `layers.*` aliases, and where the two disagree `decoder.*`
  wins, the same winner as before. **The reverse is not shimmable:** a checkpoint
  saved by this version fails in older NSM with `Missing key(s)`. Results are
  unaffected — the aliases shared one storage, so this costs disk, not accuracy
  (no `KNOWN_ISSUES` History entry for the same reason). Tooling that edits
  checkpoints by key no longer needs to write both names. Re-exporting the shipped
  model checkpoints at the halved size is follow-on coordination with the model
  releases, not part of this change.

- **Every SDF dataset cache key changes**
  ([#19](https://github.com/gattia/nsm/issues/19)). The key is now a named canonical
  mapping: `mesh_to_scale` and `uniform_pts_buffer` are finally in it (two runs
  differing only in one of them used to share a key and silently reuse each other's
  cached data), every mesh path contributes a content-stable `(path, size, mtime)`
  identity so an in-place mesh edit is noticed, and a `Mesh`-valued `reference_mesh`
  hashes by geometry instead of memory address — its cache can hit for the first time.
  No cached `.npz` from before this change is ever served again: the first run per
  configuration rebuilds its cache once (identical data when `random_seed` is set —
  `docs/KNOWN_ISSUES.md` § History 3), and old cache directories are reclaimable disk.
  A `cache_format` entry in the key versions future changes. Cached files now store
  the raw (unpadded) per-sign index sets — the equal-share padding happens at draw
  time, sized by the `subsample` in force — so a cache reloaded under a different
  `subsample` keeps `equal_pos_neg` exact instead of quietly unbalancing batches;
  for an unchanged `subsample`, batches are bit-identical to before. Details:
  § History 13.

- **`tune_reconstruction` and `compute_correlation_coefficient` are deleted**
  (`docs/SCOPE.md` §2's dead ruling, disposition 2026-08-22, executed by the §8.0.E
  pass over `reconstruct/main.py`). Zero callers, re-verified at deletion:
  `tune_reconstruction` read 27 config keys of which 22 are absent from the shipped
  default, so no shipped config could ever drive it; `compute_correlation_coefficient`
  was a four-line `np.corrcoef` wrapper. Both were importable from `NSM.reconstruct`
  and `NSM.reconstruct.main`; an import now fails loudly. Call `get_mean_errors` /
  `np.corrcoef` directly. **No numerical output changes.**

- **`read_mesh_get_sampled_pts` returns its random draw under `"pts"` — the `"xyz"` key
  is gone** (#15). The two readers disagreed on the key, so every consumer that read
  `"pts"` unconditionally — `reconstruct_mesh`'s single-object branch included — crashed
  with `KeyError` the moment `get_rand_pts=True` was set; both now use `"pts"` on every
  path. A reader of the old `"xyz"` key gets a loud `KeyError`: read `"pts"` instead. (A
  transitional alias existed briefly on this branch and was deleted before any release
  carried it — maintainer's call, 2026-08-23.) **No numerical output changes** — same
  array, one name, and the path that read `"pts"` never ran.

- **`reconstruct_mesh` raises `NoZeroLevelSetError` when the decoder's mean shape has no
  surface** (#29), instead of returning a result that looked successful — `mesh` of
  Nones, NaN metrics, and the untouched *zero* `mean_latent` under `"latent"`, with
  every other requested key dropped. `get_mean_errors` catches it and scores the subject
  NaN (`val_prediction_*` included — it used to regress on the zero vectors), so
  training-time validation still survives an under-trained model. Direct callers that
  relied on the soft dict must catch the error; `docs/KNOWN_ISSUES.md` § History 10.

- **`get_mean_errors` no longer takes `batch_size_latent_recon`, and `compute_recon_loss`
  no longer takes `n_samples_assd`** (#16's class — parameters accepted and never used).
  `batch_size_latent_recon` fed a `reconstruct_mesh` parameter that was removed when
  batching was; the only thing forwarding it did was print the deprecation warning at
  every validation pass. `n_samples_assd`'s implementing call has been commented out
  since ASSD moved to `get_assd_mesh`. Both now raise `TypeError` if passed; the trainers
  and the default config no longer carry the key. The `batch_size_latent_recon`
  deprecation shim in `reconstruct_mesh` itself stays — that is the migration surface.
  **No numerical output changes.**

- **`read_mesh_get_sampled_pts` and `read_meshes_get_sampled_pts` no longer take
  `mean`.** No code path ever read it — verified by AST scan, and it is the only
  never-read parameter in `sdf_dataset.py` — so at every value it did nothing. Removed
  rather than honoured, the same call as `get_pts_center_and_scale`'s `center`/`scale`
  below: an offset that suddenly worked would move every caller's samples. An old call
  passing `mean` still runs (both functions swallow unknown kwargs) and now prints a
  deprecation line. **No numerical output changes.**

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

- **`add_plain_lr_to_config` no longer takes `idx_model` / `idx_latent`** (#59). The
  two parameters let a caller override the `Target`-based lookup by position — the
  exact back door the Aug 2026 LR fix exists to forbid — and their only caller was a
  test asserting deliberately swapped labels. Passing them now raises `TypeError`.
  **No numerical output changes.**

- **`train_epoch` no longer takes `return_loss` or `verbose`** (#16's class — parameters
  accepted and never read). The body returns its `log_dict` unconditionally and reads
  `config["verbose"]`, so neither parameter did anything at any value. Passing them now
  raises `TypeError`; `config["verbose"]` remains the way to get verbose output. The
  copies in `train_deep_sdf_multi_head` and `train/deprecated/` are untouched — the
  former belongs to #51's repair, the latter to the quarantine decision (`SCOPE.md` §2).
  **No numerical output changes.**

- **`train_epoch` refuses three config values it used to accept** (plan §8.0.L).
  `multi_object_overlap: true` used to raise a bare `Exception("Not implemented yet")`
  from inside the innermost loop, after a full forward and backward and after every
  earlier batch had already stepped; it now raises `NotImplementedError` before the first
  batch is fetched. `eikonal_weight > 0` used to *run* — `train_deep_sdf` and
  `reconstruct_latent` gate it and `train_epoch` is the code they guard, so a caller
  reaching it directly (it is public) or through `train_deep_sdf_multi_head` got the loss
  the rest of the library refuses; it now raises the same `NotImplementedError`, and
  the plan's §8.2 repair order is unaffected because the code is untouched. A
  `surface_weighting` whose length is not the surface count used to be caught by a bare
  `assert` — stripped by `python -O`, and what was behind it is not a crash: one entry too
  long rescaled every weight that *was* used (measured on two surfaces: `[1, 1, 1]` gave
  2/3 of the unweighted loss, `[3, 1, 99]` gave 1/25 of it), one entry too short raised
  `IndexError`. It is now a `ValueError` naming both counts, under `-O` as well.
  **No working configuration changes.**

- **`train_epoch` refuses a `samples_per_object_per_batch` that disagrees with its batch**
  (plan §8.0.L). The key restates the dataset's `subsample` and the two are declared on
  different objects, so nothing held them together; a disagreement surfaced as a
  `torch.cat` size error 35 lines below the config read, naming neither key. It is now a
  `ValueError` naming both. The same check covers `_schedule_free_eval_warmup`, which
  rebuilds the index tensor the same way. **No working configuration changes.**

### Deprecated

- **`verbose=` on the 28 functions that take it**, in favour of `logging` ([#58](https://github.com/gattia/nsm/issues/58)).
  It still works, unchanged, for this release: passing it emits a `DeprecationWarning`
  and shows NSM's records — every level, down to `DEBUG` — on **stderr** for the
  duration of the call. It is
  removed at **v0.4.0** — that removal is Breaking and ships with its own entry.
  The replacement is one line of host configuration:
  `logging.basicConfig(level=logging.INFO)`, or
  `logging.getLogger("NSM").setLevel(logging.DEBUG)` with a handler on it. Per-subpackage
  control comes free with the `NSM.*` hierarchy — `logging.getLogger("NSM.datasets")`
  silences the cache chatter without silencing reconstruction.
  **`config["verbose"]` is not deprecated** and keeps its meaning: it is an on-disk
  format contract carried by every `model_params_config.json` ever written, and there is
  no `log_level` key to replace it with yet.
  The flag is *honoured* rather than made a no-op because a `DeprecationWarning` is
  invisible under Python's default filter outside `__main__`, so warn-and-no-op would be
  indistinguishable, for a caller invoking NSM from inside a module, from deleting their
  output with no notice at all.

### Changed

- **The version is the git tag** (plan §8.0.O, §10.1). `pyproject.toml` derived it from
  `NSM.__version__`, a string literal, and that literal said `0.2.0` for 269 commits and
  every breaking change in this release — the same mechanism that left it at `0.0.1` for
  years. setuptools-scm now derives the distribution version from the tag, and
  `NSM.__version__` reads the installed distribution's metadata. A tree with no git
  metadata (a GitHub source zip, a vendored copy) builds as `0.0.0+unknown` rather than
  failing, and a checkout that is not installed at all reports the same rather than raising
  — `importlib.metadata` answers about installed distributions, and NSM is reachable
  without being one.

  A commit **between** releases is named for the release it is past, not one that does not
  exist yet: five commits after `v0.3.0` is `0.3.0.post1.dev5+g<sha>`, never `0.3.1.dev5`.
  It is a prerelease under PEP 440, so `pip install nsm` will not resolve to one without
  `--pre`.

- **One message names every architecture key a config is missing** (plan §8.0.O). The three
  required triplanar keys used to be three separate refusals, so a pre-Aug-2026 config was
  repaired one key per attempt — four attempts for a fresh one, two for either shipped
  ShapeMedKnee config. The message now lists every missing key with what each decides, and
  ends with a JSON object that repairs the config in a single edit. It is raised as a
  `KeyError` subclass whose `__str__` is the message itself, because `KeyError`'s own
  renders newlines as `\n` escapes and would have delivered that block unusable.

- **`train_epoch`'s diagnostics answer to the host's logging config, not to
  `config["verbose"]`** (plan §8.0.L). Its 20 `logger.debug` records sat behind eight
  `if config["verbose"] is True:` gates. All 20 were already at `DEBUG`, so the gate only
  ever subtracted: the shipped `default_config.json` sets `verbose: true`, which left it
  permanently open and the level doing the filtering, and a config with `verbose: false`
  hid all 20 from a host that had configured `DEBUG` and asked for them. Four of the
  records were then deleted — one repeated another 39 lines up, two had no message text at
  all, one repeated the record seven lines above it — and three were renamed to describe
  what they log. `config["verbose"]` keeps its other meaning (`_run_validation` forwards
  it) and is still not deprecated.

- **`reconstruct_latent`'s own diagnostics answer to the host's logging config, not to
  `verbose=`** (plan §8.0.K). 25 of its 30 log records sat under `if verbose is True:`,
  three of them warnings that the fit had dropped a surface from its objective — so a host
  configured at `WARNING` was told nothing about it. Nothing is taken from a `verbose=True`
  caller: the bridge attaches at `DEBUG`. Three per-surface, per-step `debug` records of
  the loss tensor's shape, mean and standard deviation are deleted rather than ungated: the
  mean is already in the step record above them, and `.std()` on a one-point chunk emits a
  torch warning. Other modules still carry the same gates.

- **A multi-surface draw is balanced.** Every surface now contributes the same number of
  points, held to what the smallest one can supply; it used to get
  `min(n_samples // n_surfaces, its own count)`, so a surface too small for its share fell
  below the others and the shortfall was silently dropped. `pts_surface` decides where in
  space samples come from, not which loss they feed, so the old draw weighted the fit
  towards whichever surface had the most vertices. Unchanged for single-surface fits and
  for any fit whose every surface has at least its share — which includes both shipped
  configs and kneepipeline. The cost of balance is explicit: with unequal surfaces the
  whole cloud is now unreachable, and raising `n_samples` past
  `n_surfaces × smallest_surface` does nothing. § History 24.

- **`reconstruct_latent` returns a loss under every convergence mode.** With
  `convergence="recon_loss"` — the mode `NSM/configs/default_config.json` ships — the first
  element of `(loss, latent)` was the literal `100` the comparison sentinel was initialised
  to. No reconstruction result is affected: `reconstruct_mesh` never reads that value.
  § History 21.

- **`hybrid_optimizer`'s learning-rate schedule spans the phase it steps.** It was derived
  from `num_iterations` while the loop ran `adam_iterations + lbfgs_iterations`, so
  `n_lr_updates=2` could apply 11 decays and reach exactly 0.0. Non-hybrid runs are
  bit-identical. § History 22.

  The sampling *cadence* is deliberately unchanged: `reconstruct_latent` draws a new
  random subsample on every loss evaluation, so LBFGS draws several times per step. §8.0.K
  proposed making that once-per-step — a line search over a moving objective is undefined —
  and the measurement refused it: the redraw is how the fit covers the point cloud, and at
  a 5% sampling ratio dropping it took median held-out error from 0.007 to 0.029. See
  `TestTheDrawIsPerEvaluation`.

- **`reconstruct_mesh`'s own diagnostics answer to the host's logging config, not to
  `verbose=`** (plan §8.0.J). Ten of its fifteen log records sat under `if verbose is
  True:` — a faithful conversion of the `print` gates they replaced, and the reason a host
  that ran the exact replacement the deprecation notice names (`logging.getLogger("NSM")`
  at `DEBUG`) saw none of them. One is a warning that a surface was skipped. Nothing is
  taken from a `verbose=True` caller: the bridge attaches at `DEBUG`. Other modules still
  carry the same gates.

- **`register_similarity` is read one way.** The mean-mesh build tested
  `register_similarity is True` while the forward to the samplers tested truthiness, so a
  truthy non-`True` value — `1`, `"similarity"` — skipped the build and then raised
  `Exception: Must provide mean mesh to register to` from inside `datasets/mesh_sampling`.
  Truthiness is the reading kept, so those values now register.

- **`scale_jointly` no longer builds a mean mesh**, which it built and discarded: the mean
  mesh has one reader, under `register_to_mean_first`, which comes from
  `register_similarity` alone. Measured, that was 876,269 decoder point-evaluations per
  call at the `n_pts_per_axis_mean_mesh=128` default. `NoZeroLevelSetError` no longer
  fires on a `scale_jointly`-only call either — it was aborting the run over a mesh nothing
  was going to consult. Reconstruction output is byte-identical either way.

- **`return_timing` returns `time_calc_recon_loss`**, which the body measured on every
  scored call and no branch ever put in the result dict. It is the timing of the one
  optional stage, so `return_timing` was silent about exactly the stage a caller profiling
  a reconstruction would be looking for.

- **NSM's diagnostics go to `logging`, not `print`, and therefore to stderr rather than
  stdout** ([#58](https://github.com/gattia/nsm/issues/58)). A caller that captured NSM's
  stdout to read its messages now finds them on stderr; a caller that parses its *own*
  output out of stdout — which is what the pipeline consumer does — now has that stream
  to itself.

- **Importing NSM no longer reconfigures the host process's root logger.**
  `NSM/reconstruct/main.py` called `logging.basicConfig(...)` at module scope, and
  `NSM.reconstruct` star-imports it, so any `import NSM.reconstruct` set the root
  logger's level to `INFO` and attached a `StreamHandler` in the caller's process.
  `NSM/__init__.py` now installs the stdlib `NullHandler` on the `"NSM"` logger instead,
  so NSM emits nothing until a host asks for it.

### Fixed — affects results

- **`model_params_config.json` records the subjects that trained the model** (plan §8.0.M).
  `save_model_params` placed its `list_mesh_paths` argument and then merged the config over
  it, so a config key of the same name won. The shipped `default_config.json` carries
  `"list_mesh_paths": null`, which means every run started from it recorded no subject list
  at all; a config round-tripped from an earlier run's saved file recorded that run's
  subjects. Provenance only — no NSM code path reads the entry back, so no weights, latents
  or metrics move. See `docs/KNOWN_ISSUES.md` § History 26.


- **The logged `mean_vec_length` / `std_vec_length` are epoch means at every `batch_split`**
  (plan §8.0.L). #59 fixed this on the batch loop; underneath it is a split loop, and both
  statistics were computed inside that one and read after it, so with `batch_split` above 1
  the logged value was the last split's. Measured on one fixture, same seed, only
  `batch_split` changing: `mean_vec_length` 0.1445 / 0.2026 / 0.3201 for 1 / 2 / 4, against
  a loss invariant to 1.5e-08 — and `NaN` wherever a split held a single row, since
  `torch.std` of one value is undefined. `batch_split: 1` is bit-unchanged, which is the
  shipped default. Weights, gradients and checkpoints were never affected at any setting.
  See `docs/KNOWN_ISSUES.md` § History 25.

- **The logged `mean_vec_length` / `std_vec_length` are epoch means** (#59). They were
  assigned (`=` for `+=`) and then divided by the batch count, so every wandb run since
  Nov 2024 logged the last batch's stat shrunk by ~×n_batches. Weights, gradients and
  checkpoints were never affected — the two stats sat outside the loss path. See
  `docs/KNOWN_ISSUES.md` § History §12.

- **`resume_epoch: 1` resumes from the epoch-1 checkpoint** (#49). The resume guard
  read `> 1` while the epoch loop starts at `resume_epoch + 1`, so such a run loaded
  nothing and trained a fresh model for epochs 2..`n_epochs` — one epoch short, from
  random init, silently. `resume_epoch` now uniformly names the last completed epoch:
  `>= 1` loads that checkpoint and continues at the next; `0` is unchanged. A post-fix
  `resume_epoch: 1` run actually resumes, so it produces different (correct) weights.
  See `docs/KNOWN_ISSUES.md` § History §11.

- **`reconstruct_mesh` honours `n_pts_random`** (#16). It forwarded the value as
  `n_pts_random=` to readers whose parameter is `n_pts=`; their `**kwargs` swallowed it,
  so every `get_rand_pts=True` call drew the readers' 200,000-point default per surface
  regardless of what was asked. **Numerical output changes** for post-fix reruns of such
  calls (never a shipped configuration) — `docs/KNOWN_ISSUES.md` § History 9.

- **`include_surf_in_pts` on `read_meshes_get_sampled_pts` appends each surface's own
  vertices** (#17). It appended a leaked loop variable instead: on the one
  configuration that ran (centering on, numeric sigmas) that was the *last* surface's
  *pre-normalization* vertices, once per surface — wrong surface, wrong frame; every
  other configuration crashed (`UnboundLocalError` / `ValueError`) and now works as
  documented. Training data is unaffected — the dataset classes never pass the flag —
  and both shipped ShapeMedKnee configs leave `get_rand_pts_recon: false`. A
  multi-object reconstruction with `get_rand_pts=True` on a `scale_jointly=False`
  model now fits against different points. See `docs/KNOWN_ISSUES.md` § History §7.

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

- **`NSM/configs/default_config.json` ships in built distributions** (plan §8.0.O).
  `[tool.setuptools.packages] find` packages modules, not data, and there was no
  `package-data` entry, so no wheel has ever contained the file every doc in this repo
  calls the shipped default config. It worked only because every install so far has been
  editable. The generator beside it was shipping already — `find` takes `namespaces = true`
  by default, so `NSM.configs` is picked up despite having no `__init__.py`.

- **`save_model_params`' write-once refusal says so, and names what diverges from it**
  ([#50](https://github.com/gattia/nsm/issues/50), plan §8.0.M). It is called on every
  checkpoint and returns without writing once the file exists — deliberate, because the
  file records the configuration that produced the weights — but it returned silently, so
  a re-configured run in the same `experiment_directory` left a file describing a different
  model from the one saved beside it. The current config is now compared with what is on
  disk and every added, removed or changed key is named in one `WARNING`.

- **Config values that cannot be JSON-encoded are named when the record is written**
  ([#50](https://github.com/gattia/nsm/issues/50), plan §8.0.M). `filter_non_jsonable`
  dropped them silently. The log sits at the write rather than at the filter, so it fires
  once per run: in a normal run the set is exactly `lr_schedules`, the schedule objects the
  trainer inserts into the caller's config, and anything joining it there is a value the
  caller set that the record will not carry.

- **`clear_gpu_cache` accepts a `torch.device`** (plan §8.0.M); `"cuda" in device` needs a
  `str`, so the object form raised `TypeError: argument of type 'torch.device' is not
  iterable`. NSM's own callers pass `config["device"]`, which comes from JSON.

- **`is_jsonable` returns `False` for a circular value** (plan §8.0.M) instead of
  propagating `ValueError: Circular reference detected` out of the checkpoint write; it
  caught `TypeError` and `OverflowError` only.


- **`batch_split` values `torch.chunk` cannot honour no longer crash the epoch** (plan
  §8.0.L). `torch.chunk(t, k)` returns *at most* `k` pieces — it splits into pieces of
  `ceil(len / k)` and stops when the tensor runs out — and both split loops in
  `train_deep_sdf.py` were bounded by the requested `k`, so `batch_split=5` on a 16-row
  batch raised `IndexError: tuple index out of range` while 3 and 6 on the same batch
  worked. The chunks still partition the batch, so the affected values now produce the loss
  they would have produced, not a new one. Second site is `_schedule_free_eval_warmup`.

- **wandb is optional** (#5). It was an undeclared import-time dependency: without it,
  `import NSM.reconstruct` — the inference path, which never logs — and
  `import NSM.train` both died with `ModuleNotFoundError`. Both packages now import
  without wandb; every wandb use is behind an explicit request (`log_wandb`,
  `use_wandb`, `config["log_latent"]`), which raises `ImportError` by name, at entry,
  when wandb is absent. One deliberate skip instead of a raise: `get_mean_errors`'
  per-metric histograms (nothing requests wandb on that path) become `None` without
  wandb — which is what lets a training run's validation epochs complete in a
  wandb-less environment. With wandb installed, nothing changes.

- **schedule_free training runs survive their checkpoint and validation epochs** (#42).
  The eval warm-up handed the decoder the raw dataloader item, so every
  `schedule_free_*` run died with `TypeError` in the decoder's forward at its first
  checkpoint or validation epoch — which every run reaches. The warm-up now unpacks the
  batch the way `train_epoch` does (latent lookup, variational sampling, `batch_split`
  chunking included). Always crashed, so no results are affected and there is no
  History entry.

- **`predict_val_variables` runs to completion** (#48). `get_mean_errors` handed
  `Regress.add_latent` the whole result dict rather than the fitted latent, so a run
  that enabled the latent-to-factor validator died with `TypeError` in `calc_r2` at its
  first validation pass — after all its reconstructions had run. The seam now passes
  the latent as a flat float vector. Always crashed, so no results are affected and
  there is no History entry.

- **`norm_penalty_type='barrier'` raises by name outside its `(min, max)` range instead
  of returning NaN** (#48). Below the range — where every run starts unless
  `latent_init_std` puts the initial norm inside it — the log term's value was NaN but
  its gradient was finite and pushed the norm further *away* from the range; the run
  completed with `nan` in every loss readout. Strictly inside the range nothing
  changed, and a single-target barrier still silently computes the quadratic penalty
  (now stated in the docstring). Neither shipped config sets `latent_norm`, so no
  production path is affected. See `docs/KNOWN_ISSUES.md` § History §8.

- **`scale_jointly=True` works with `store_data_in_memory=True`** (#69). The in-memory
  branch of `norm_and_scale_all_meshes` read the flattened `new_pts_0`-style keys that
  exist only in the `.npz` cache layout, so the combination raised `KeyError` at
  construction on both dataset classes — and it omitted `joint_scale_buffer`, so a
  KeyError-only fix would have put in-memory runs in a different coordinate frame than
  disk-backed ones. Both storage modes now compute the shared frame the same way and
  `__getitem__` applies it per batch; disk-backed numerics are unchanged. Always
  crashed, so no results are affected.

- **`store_data_in_memory=True` constructs, yields items, and trains** (#22).
  `MultiSurfaceSDFSamples.__getitem__` now guards the load-timing block the way
  `SDFSamples` always did, and `train_epoch` treats the four timing keys as optional
  diagnostics instead of reading them unconditionally — so in-memory datasets train, and
  the keys appear (in batches and in the epoch log) only when a disk load was actually
  timed.

- **A zero sampling probability samples nothing instead of crashing** (#23).
  `p_near_surface=0` / `p_further_from_surface=0` produced a zero-count combo that was
  handed to `point_cloud_utils` anyway; both classes now skip empty combos.

- **`LOC_SDF_CACHE` is read when a dataset is constructed, not when the module is
  imported** (#24). Setting the variable before construction now works; an empty value
  counts as unset. Pass `loc_save` explicitly to override either way.

- **A surface with no positive or no negative SDF samples raises a `ValueError` naming
  the surface** (#41) instead of `ZeroDivisionError` — e.g. one surface nested inside
  another loses every interior point to overlap removal. A surface nothing draws from (a
  missing/`None` surface, or one allotted no subsample share) yields empty index lists
  and is handled.

- **`MultiSurfaceSDFSamples` accepts `joint_scale_buffer`** (#43) and forwards it to
  joint normalization. It was refused with `TypeError`; the parent's default (0.1)
  happens to equal the production value, which is why nothing noticed. Not yet in the
  cache key — that is #19's business (it does not change cached bytes).

- **`reference_mesh=<int>` with multi-surface registration builds** (#61). The path
  raised `UnboundLocalError`, and `combine_meshes` returned a pyvista `PolyData` (no
  `save_mesh`) whenever it actually combined meshes; it now keeps its declared `Mesh`
  return type.

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

### Added

- **`reconstruct_latent(n_samples_per_chunk=)` and `reconstruct_mesh(
  n_samples_per_chunk_latent_recon=)`** ([#75](https://github.com/gattia/nsm/issues/75)):
  split each optimization step's forward *and* backward into chunks of that many points,
  accumulating the gradient on the latent, so a step's memory stops scaling with
  `n_samples`. Measured on a Tesla T4 at 200,000 points with a latent-256 8×512 decoder:
  peak allocation **4128 → 623 MiB**, with the fitted loss agreeing to 1.2e-07 relative.
  `None`, the default, is the single unchunked pass every earlier run took, bit for bit;
  setting it changes the order the per-point losses are summed, so it is a
  numerics-affecting option rather than a transparent optimization.

- **`MultiSurfaceSDFSamples` accepts `mesh_names`, and `train_deep_sdf` trusts the
  dataset over the config** (#52). Surface identity is defined by the order of each
  subject's mesh-path list, so the names are declared there: the dataset validates them
  against its own per-subject surface count at construction, and the trainer adopts
  them into `config["mesh_names"]` — raising at entry if a config declaration
  disagrees — before anything is persisted to `model_params_config.json`. A config-only
  declaration with a nameless dataset behaves as before. Deliberately not in the cache
  key: names do not change sampled data.

- **`train_deep_sdf` returns its per-epoch history** (#28). One dict per trained epoch:
  the wandb payload (validation metrics included on validation epochs) plus `epoch`,
  per-param-group `lrs`/`targets`, and per-subject `latent_norms`. It used to return
  `None`, so a caller without a wandb key could learn nothing about a run except by
  reading checkpoints back off disk. The wandb payload itself is unchanged.

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
