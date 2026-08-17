# Audit findings register

> ## ⚠️ These are hypotheses, not findings. Do not cite an entry without re-running it.
>
> **178 of the 216 entries below were produced by reading the code, not by executing it**,
> and every inferred claim that has since been tested was **wrong, in the direction of
> overstatement** — the triplanar "affine map" claim (`ARCHITECTURE.md` §7.1) and the
> eikonal "forward+backward runs" claim. Treat any entry as *a line number and a
> suspicion*.
>
> The 38 executed entries are no safer to cite blind: they were run against `main` at
> `73a0326`, and `sdf_dataset.py` alone has moved 104 lines since. **Executed once is not
> true now.**
>
> **Delete when:** every entry is either a GitHub issue, an entry in `KNOWN_ISSUES.md`, a
> ruling in `SCOPE.md`, or deleted. Per `CLAUDE.md` § Four rules, inference is not a
> finding and does not belong in `docs/` — this file is transitional and its continued
> existence is a debt, not an asset. Anything that does not survive re-running goes in a
> plan's **Next**, not into `KNOWN_ISSUES.md`. Git has the rest.

Raw output of the Phase 1 mapping pass. Every entry carries a `file:line`. Nothing here has
been fixed.

**This is not a to-do list** — GitHub issues are. It is the pool that `KNOWN_ISSUES.md`
entries were promoted out of, and an entry earns a History entry only if it silently
changed numerical output for inputs that previously ran without error (the rule in
`CLAUDE.md` § Numerical-behaviour changes).

## Severity vocabulary

| Severity | Count | Meaning |
|---|---|---|
| **landmine** | 71 | Wrong or surprising behaviour that produces no error. The reader gets a plausible number. |
| **defect** | 58 | Broken behaviour that does raise, or is unambiguously incorrect. |
| **rot** | 54 | Stale comment, false docstring, dead parameter, unused import. |
| **note** | 33 | Observation worth recording; not itself a bug. |
| | **216** | |

Landmines outnumber defects. That is the expected shape for this codebase and it is the
reason Phase 3 leads with a numerical-regression harness rather than unit tests: almost
nothing here would be caught by a test that only asserts "it ran".

## How to read this document — and how much to trust it

**Do not read it front to back.** It is an index for triage, not a report. Use it to answer
"what is known about this file" when you are about to touch that file.

**Confidence is not uniform, and the difference matters more than the severity.**

| | Executed — a script was run | Inferred — read only | Total |
|---|---|---|---|
| landmine | 19 | **52** | 71 |
| defect | 11 | **47** | 58 |
| rot | 3 | 51 | 54 |
| note | 5 | 28 | 33 |
| **Total** | **38** | **178** | 216 |

An inferred finding is a hypothesis with a line number attached. Two have already been
tested and **both were wrong in the direction of overstatement**:

- `triplanar.py:87` — claimed the VAE decoder is "an affine map." False for both shipped
  models; LayerNorm supplies the missing nonlinearity. Corrected in place.
- `losses.py` — the register claimed eikonal forward+backward "runs for 1, 2 and 3
  surfaces." Re-run: it raises `RuntimeError` every time. It has never worked.

Two for two against, on the only two that got tested. Treat the inferred set accordingly —
and note that *both* corrections came from execution, neither from re-reading.

**The policy going forward is not a separate verification pass.** Findings get settled as a
by-product of Phase 3 building tests, so the effort leaves permanent tests behind instead of
throwaway scripts. This register's job is to tell Phase 3 *where to point*. The 30 landmines
that are (a) inferred, (b) on the production path — `sdf_dataset.py`, `reconstruct/main.py`,
`mesh/main.py`, `models/` — are the highest-value targets, because being wrong about them is
expensive in both directions: a false one wastes a fix, a true one is shipping today.

---

## Landmines (71)

Silent wrong behaviour. Each of these returns a number rather than raising.

### `NSM/_lr_migration.py`

**NSM/_lr_migration.py:55 — migration_error decides the historical LR mapping by substring-sniffing the optimizer name**

`schedule_free = "schedule_free" in str(optimizer)` picks between `_HISTORICAL_TARGETS_ADAM` and `_HISTORICAL_TARGETS_SCHEDULE_FREE` (lines 23-24, 56), which are OPPOSITE mappings. Anything not matching that substring — including a typo'd or future optimizer name — is silently told the Adam history. Given that the two answers are exact inverses, a wrong guess here tells a user to reproduce their run with the schedules swapped. The same substring test is duplicated in the train loop at NSM/train/train_deep_sdf.py:290 (`if not ("schedule_free" in config["optimizer"])`).

### `NSM/configs/default_config.json`

**NSM/configs/default_config.json:1 — No shipped config can construct a triplanar model faithfully**

The 61 keys in default_config.json cover the deepsdf vocabulary (latent_size, layer_dimensions, weight_norm, activation, final_activation, xyz_in_all, layers_with_*, dropout_prob) but contain none of conv_hidden_dims, conv_deep_image_size, conv_norm, conv_norm_type, conv_start_with_mlp, sdf_latent_size, sdf_hidden_dims, sum_conv_output_features, conv_pred_sdf, padding, or objects_per_decoder. Loading a triplanar model from it would silently fall back to every default in _get_triplanar_params (loader.py:120-133), which is a different architecture from the production models. The production consumer avoids this by requiring all those keys explicitly from the model's own saved config (kneepipeline/steps/run_nsm.py:94-110) and raising KeyError if absent.

### `NSM/configs/generate_sdf_default_config.py`

**NSM/configs/generate_sdf_default_config.py:1 — NSM/configs is not a package and will not ship in a built distribution**

`NSM/configs/` has no `__init__.py` (only `__pycache__`, `deep_sdf_config`, `default_config.json`, `generate_sdf_default_config.py`). Verified: `setuptools.find_packages(include=['NSM','NSM.*'])` returns ['NSM','NSM.datasets','NSM.dependencies','NSM.mesh','NSM.models','NSM.reconstruct','NSM.train'] — no NSM.configs. pyproject.toml:44-45 uses exactly that include list, and there is no `[tool.setuptools.package-data]`, no `include-package-data`, and no MANIFEST.in. So a wheel/sdist of NSM contains neither the generator nor `default_config.json`. It works today only because installs are editable (source dir on sys.path). Both `testing/NSM/configs/test_default_config_sync.py:15-18` and `testing/NSM/test_lr_schedules.py:547` would fail against a non-editable install.

### `NSM/datasets/sdf_dataset.py`

**NSM/datasets/sdf_dataset.py:87 — get_pts_center_and_scale ignores its center= and scale= flags (verified at runtime)**

Both boolean parameters are dead: `center` is rebound to np.mean(pts) at line 88 (or 90) before it is ever read, and `pts -= center` at 91 is unconditional; `scale` is rebound at 94 and `pts /= scale` at 95 is unconditional whenever scale_method == 'max_rad'. Runtime check: get_pts_center_and_scale(pts, center=False, scale=False) returns a real center/scale and leaves the input normalized to max radius 1.0. Consequence at the call site: read_meshes_get_sampled_pts:609-616 passes center=center_pts, scale=norm_pts believing they select the operation, then line 621 applies BOTH unconditionally -- so SDFSamples' documented default of center_pts=True, norm_pts=False (lines 815-816) silently normalizes to the unit sphere anyway. There is no reachable configuration that centers without scaling.

**NSM/datasets/sdf_dataset.py:91 — get_pts_center_and_scale mutates the caller's array in place, undocumented**

Lines 91 and 95 do `pts -= center` / `pts /= scale` on the passed array. The docstring (59-85) never mentions it. All three in-repo callers pass np.copy() defensively (279, 610, NSM/reconstruct/reconstruct_latent_S3.py:131) -- the convention is real but exists only as a habit at the call sites, so any new caller that omits np.copy() silently destroys its source coordinates.

**NSM/datasets/sdf_dataset.py:184 — n_pts_random is silently swallowed by **kwargs in both read_*_get_sampled_pts functions**

Neither function has an `n_pts_random` parameter; the count parameter is `n_pts` (line 171, line 407). NSM/reconstruct/main.py:985 and :1002 pass `n_pts_random=n_pts_random`, which lands in **kwargs (184 / 425) and is discarded -- the deprecation loop at 229-231 / 469-471 only recognizes five other names, so not even a warning is printed. The consumer forwards a real value for this: /mnt/data/programming/kneepipeline/steps/run_nsm.py:201 passes n_pts_random=model_config['n_pts_random_recon'] (=100000 in both shipped configs). When get_rand_pts is enabled, sampling would silently use the 200000 default instead. This is the same class of bug as the param-group ordering bug that motivated the audit: a caller-supplied number that looks honoured and is not.

**NSM/datasets/sdf_dataset.py:314 — read_mesh_get_sampled_pts returns 'xyz' or 'pts' depending on get_random; one consumer reads 'pts' unconditionally**

get_random=True populates results['xyz'] (314); get_random=False populates results['pts'] (318). The multi-mesh function always uses 'pts' (692, 741). The internal callers hedge -- `result_['pts'] if 'pts' in result_ else result_['xyz']` at 1237 and 1855 -- but NSM/reconstruct/main.py:1009 reads result_['pts'] unconditionally after the single-object call at 976, so reconstruct_mesh(path=<str>, get_rand_pts=True) raises KeyError. Currently unreachable from the consumer only because steps/run_nsm.py always passes a list (making multi_object True) and get_rand_pts_recon is false. UNVERIFIED whether any out-of-repo caller passes a bare string with get_rand_pts=True.

**NSM/datasets/sdf_dataset.py:820 — loc_save default is evaluated at import time, so setting LOC_SDF_CACHE later has no effect**

`loc_save=os.environ.get('LOC_SDF_CACHE', ~/.cache/nsm_sdf_cache)` is a default ARGUMENT (820-822, and again 1609-1611), evaluated once when the class body executes at import. Any process that imports NSM.datasets (which NSM.reconstruct does transitively) and then sets LOC_SDF_CACHE gets the stale value. The consumer does exactly this ordering-sensitive thing at /mnt/data/programming/kneepipeline/steps/run_nsm.py:172, setting LOC_SDF_CACHE to the empty string after the import at line 170 -- and note os.environ.get returns '' rather than the fallback when the key exists but is empty, which would make cache_folder a relative path in the CWD if a dataset were ever constructed in that process.

**NSM/datasets/sdf_dataset.py:900 — Undocumented subclass initialization-order contract enforced by hasattr**

SDFSamples.__init__ does `if not hasattr(self, 'reference_object'): self.reference_object = 0` and the same for n_meshes (900-903). This is a load-bearing contract: a subclass MUST set those attributes before calling super().__init__, which MultiSurfaceSDFSamples does at 1645-1650. Nothing documents it, and getting it wrong silently reverts to reference_object=0 / n_meshes=1, which changes the centering reference in norm_and_scale_all_meshes (1024, 1061) rather than raising. Same shape of hazard as the param-group ordering bug: correctness carried by statement order, not by a name.

**NSM/datasets/sdf_dataset.py:1097 — ISSUE #3 CONFIRMED: sigma_near/sigma_far change coordinate space with scale_jointly**

Verified. preprocess_inputs (1092-1105) REQUIRES center_pts=False and norm_pts=False when scale_jointly=True (it raises ValueError otherwise). get_sample_data_dict then forwards those flags into read_mesh_get_sampled_pts (1223-1224) / read_meshes_get_sampled_pts (1828-1829). Inside those functions, normalization happens at lines 276-290 (single) / 565-634 (multi) and sigma is consumed AFTER it, at line 296-298 (single) / 654-660 (multi). So: scale_jointly=False -> the mesh is already centered+divided by max radius, sigma is in unit-sphere units; scale_jointly=True -> no normalization has happened, sigma is in raw mesh units (mm), and the joint normalization is instead deferred to __getitem__ at 1546-1550 / 2143-2146, i.e. after sampling. Empirically corroborated: the shipped production configs use scale_jointly=true with sigma_near=0.743, sigma_far=2.35 (/mnt/data/programming/kneepipeline/NSM_MODELS/647_nsm_femur_v0.0.1/model_params_config.json), roughly 100x the 0.01/0.1 defaults documented at lines 781-782 for the other mode. Note the mechanism is subtler than the issue text: because of the get_pts_center_and_scale flag bug above, it is scale_jointly alone (not norm_pts) that decides the space.

**NSM/datasets/sdf_dataset.py:1305 — sdf_pos_neg_idx divides by zero when a surface has no positive or no negative samples**

`pos_idx.repeat(samples_per_sign // pos_idx.size(0) + 1)` at 1305-1306 and the multi-surface copy at 2029-2030 both divide by the count of found indices. A degenerate/thin surface, an all-NaN column from a None mesh, or a get_random=False configuration where a surface's own SDFs are all exactly 0 (719) yields size(0)==0 -> ZeroDivisionError deep inside a multiprocessing worker.

**NSM/datasets/sdf_dataset.py:1396 — uniform_pts_buffer and subsample also affect cached content but are not hashed**

uniform_pts_buffer changes the sampling domain (301-303, 647-648, 309-310) and is in neither get_hash_params (1396-1409, 1973-1999). subsample determines samples_per_sign, which determines the repeat factor baked into the cached pos_idx/neg_idx (1304-1306, 2029-2030) -- also unhashed. On reload the presence check at 1181 / 1764-1771 accepts the stale arrays, and the shortfall is silently papered over by the random top-up at 1524-1529 / 2122-2127, so the positive/negative balance quietly stops matching the requested subsample.

**NSM/datasets/sdf_dataset.py:1406 — reference_mesh is hashed by str(), so passing a Mesh object makes the cache key its memory address**

get_hash_params puts self.reference_mesh into the list (1406, and 1981 for multi); create_hash stringifies everything at 1437. A str path hashes deterministically, an int index hashes deterministically, but a pymskt Mesh instance stringifies to '<...Mesh object at 0x...>' -- different every process -- so the cache never hits and a fresh .npz is written on every run. Note also the ordering: get_hash_params runs at line 911, BEFORE load_reference_mesh (924) converts/normalizes self.reference_mesh, so the hash sees the raw user input, and after construction self.reference_mesh has been set to None (1386) when multiprocessing is on.

**NSM/datasets/sdf_dataset.py:1635 — n_meshes and n_pts are derived from len(list_mesh_paths[0]), which is a character count for a string path**

Lines 1635-1638 build n_pts as [n_pts] * len(list_mesh_paths[0]) and 1649 sets self.n_meshes = len(list_mesh_paths[0]). If the caller passes a flat list of string paths (which preprocess_inputs at 1688-1689 explicitly anticipates), len() of a path string is its character count -- so total_n_pts (1646) is computed from a 40-element list. preprocess_inputs later repairs self.n_meshes but never recomputes self.total_n_pts, which then feeds get_samples_per_sign (1895).

**NSM/datasets/sdf_dataset.py:1759 — Cache-upgrade path resaves stale pos/neg indices after removing overlapping points**

On a cache hit, MultiSurfaceSDFSamples.get_sample_data_dict calls remove_overlapping_points at 1759, which DELETES rows from xyz and gt_sdf (1932-1933). The pos_idx/neg_idx/surf_idx already loaded from that cache still index the pre-deletion array. The recompute at 1773-1776 only fires if a key is missing or the list length != n_meshes -- neither of which a row deletion changes -- so resave_data=True at 1762 writes the STALE indices back to disk (1787). test_if_idx_in_range (1705-1716) catches only out-of-range indices; indices that remain in range but now point at different points pass silently and are baked into the cache for every subsequent run. The freshly-generated branch gets the order right (remove at 1873, then compute at 1876).

**NSM/datasets/sdf_dataset.py:1973 — mesh_to_scale is not part of the multi-surface cache hash**

MultiSurfaceSDFSamples.get_hash_params (1973-1999) lists center_pts, norm_pts, scale_method, rand_function, scale_all_meshes, center_all_meshes, reference_mesh, reference_object, a bare False, fix_mesh, scale_jointly, then the per-mesh n_pts/p_near/p_far/sigma_near/sigma_far. mesh_to_scale is absent -- yet it selects which surface(s) drive ICP registration (527-534) AND which drive centering/scaling (579-595). Two runs that differ only in mesh_to_scale produce the same hash and the second silently reuses the first's cache with the wrong alignment and the wrong normalization.

**NSM/datasets/sdf_dataset.py:1983 — Unexplained bare `False` literal inside the multi-surface hash parameter list**

get_hash_params builds a list of meaningful attributes and drops a bare `False` at position 9 (line 1983) with no comment. It is presumably a frozen placeholder for a removed parameter, kept so existing cache hashes still resolve. Anyone tidying this list will silently invalidate every cached .npz on disk; anyone reading it cannot tell what it stood for.

### `NSM/losses.py`

**NSM/losses.py:110 — losses.py builds model input as cat([latent, points]) — an undocumented latent-first ordering, the same bug class as the LR mapping**

`model_input = torch.cat([latent, points], dim=-1)` at losses.py:110 (compute_sdf_gradients) and losses.py:213 (combined_sdf_loss). Neither docstring states that the decoder's legacy `x` interface expects the latent code first and xyz last (TriplanarDecoder.forward, NSM/models/triplanar.py:330-342, accepts either `x` or `latent`+`xyz`), and nothing validates that `latent.shape[-1] + 3` matches the decoder's expected width. Swap the two and the model consumes garbage silently. This is exactly the undocumented-positional-ordering pattern the audit was opened over.

### `NSM/mesh/correspondence_metrics.py`

**NSM/mesh/correspondence_metrics.py:333 — self_intersection_count's runtime guard does not guard against its actual runtime**

`_aabb_broadphase` is a pure-Python nested loop over triangle ranks (:333-348) with an O(n^2) worst case, and the narrow phase (:312-317) is another Python loop calling `_tri_tri_intersect` per candidate pair. `max_triangles` defaults to 50_000 (:256) and the docstring frames it as the protection against 'excessive runtime' (:271-273) / 'rather than hanging' (:265) — but a 50k-triangle mesh under this implementation will not finish in a usable time. The tests only ever exercise small spheres/planes (testing/NSM/mesh/test_correspondence_metrics.py:250-288).

**NSM/mesh/correspondence_metrics.py:537 — Adjacent metrics take the same two arrays in opposite positional order**

`roundtrip_distance(original_points, roundtrip_points)` (:511) and `forward_backward_disagreement(roundtrip_points, original_points)` (:537) are defined 26 lines apart, both take two (N,3) float arrays, and their parameters are in REVERSED order. Swapping them is undetectable at the call site: roundtrip_distance is symmetric (norm of a difference) so it silently returns the identical number, and forward_backward_disagreement silently returns a sign-flipped `field` with identical `magnitude_percentiles`. Exactly the audited bug class.

**NSM/mesh/correspondence_metrics.py:676 — score_correspondence fabricates an 'original' when source_mesh is missing**

`original_pts = _mesh_points(source_mesh) if source_mesh is not None else _mesh_points(warped_mesh)`. If a caller supplies roundtrip_points but not source_mesh, roundtrip_distance and forward_backward_disagreement are computed against the WARPED points and return a plausible-looking but meaningless number instead of the `{"skipped": True}` the function uses everywhere else (:639, :657, :669, :690). The docstring (:596-600) says only that roundtrip_points is required.

### `NSM/mesh/interpolate.py`

**NSM/mesh/interpolate.py:116 — sdf_gradients returns a gradient array whose first D_lat columns are always zero**

`grad_latent_zeros = torch.zeros(B, D_lat, ...); full_grad = torch.cat([grad_latent_zeros, grad_pos], dim=1)` (:116-117 and :130-131) — the returned (B, D_lat+3) array is 90%+ zero padding for a 256-dim latent. The docstring says only 'gradients for that surface only (B, latent_dim + 3)' (:67); a caller who slices `[:, :D_lat]` expecting dSDF/dz gets silent zeros, not an error. .claude/plans/completed/...md:806 and :1056 confirm the latent gradient was deliberately never wired in. The function has zero callers in the repo.

**NSM/mesh/interpolate.py:307 — faces argument silently accepts pyvista's VTK-style face array and builds garbage**

`build_mesh_laplacian` (:307) and `compute_feature_mask` (:349) both do `np.asarray(faces).reshape(-1, 3)` with no validation. pyvista's `mesh.faces` is the flat VTK form `[3, i, j, k, 3, i, j, k, ...]`; for a triangular mesh its length is 4*M, which is divisible by 3 whenever M is divisible by 3, so the reshape SUCCEEDS and produces nonsense connectivity (the leading 3s become vertex indices). The correct input is `mesh.regular_faces`, which the tests use (testing/NSM/mesh/test_interpolate.py:125, 155, 181) and which .claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md:240 instructs the nsosim consumer to pass — but nothing in the code enforces or documents it.

**NSM/mesh/interpolate.py:519 — The is_mesh path mutates the caller's mesh in place and does not say so**

`data.point_coords = points...` (:519), `data.mesh.subdivide_adaptive(..., inplace=True)` (:520-528), `data.mesh.smooth(inplace=True, ...)` (:531), `data.mesh.smooth_taubin(inplace=True, ...)` (:533), plus `add_cell_idx(data)` which adds a 'cell_idx' cell array (:38-44, called at :499). The function then also RETURNS `data` (:536), which reads as a pure function. interpolate_mesh's docstring (:657-662) never mentions the mutation.

**NSM/mesh/interpolate.py:625 — interpolate_points / interpolate_mesh call interpolate_common with 8 positional args**

`interpolate_common(model, latent1, latent2, n_steps, points1, surface_idx, verbose, spherical, is_mesh=False, ...)` at :625-634 and the parallel call at :664-671. `points1` / `mesh` bind to the parameter named `data`. interpolate_common's signature (:473-492) has 18 parameters; inserting one before `data` silently mis-binds the point set as the step count. Same class as the param-group-ordering bug.

### `NSM/mesh/main.py`

**NSM/mesh/main.py:126 — scale_mesh_ 's trailing underscore promises in-place semantics it only sometimes has**

`if not issubclass(type(mesh), mskt.mesh.Mesh): mesh = mskt.mesh.Mesh(mesh)` (:127-128) — verified empirically that mskt.mesh.Mesh copies its input, so a pyvista/vtk/path argument is NOT mutated, while an mskt.mesh.Mesh argument IS mutated in place at :136. Two different aliasing contracts behind one name, with no docstring.

**NSM/mesh/main.py:171 — scale_mesh silently overrides the caller's scale and offset**

When `old_mesh is not None`, `offset = np.mean(old_pts, axis=0)` (:171) and `scale = np.max(...)` (:175) overwrite whatever the caller passed for those two named parameters, with no warning and no docstring (the function has none at all, :151-159). A caller supplying both old_mesh and an explicit scale gets the explicit value discarded.

**NSM/mesh/main.py:690 — Fallback path passes 17 positional arguments to create_mesh**

`return create_mesh(decoder, latent_vector, n_pts_per_axis, voxel_origin, voxel_size, batch_size, scale, offset, path_save, filename, path_original_mesh, scale_to_original_mesh, icp_transform, objects, verbose, device, use_vtk)` — all 17 positional. It currently matches create_mesh's signature (:185-203) exactly, but inserting or reordering a single parameter in create_mesh silently mis-binds every argument after it with no error (all trailing params are same-ish types: bools, floats, tuples). This is the same failure class as the param-group-ordering bug that motivated the audit.

**NSM/mesh/main.py:711 — create_mesh_adaptive silently discards the caller's voxel_origin**

`samples, grid_dims, voxel_origin = create_grid_samples_in_bounds(...)` rebinds the `voxel_origin` parameter. On the adaptive (non-fallback) path the caller's value is never read. NSM/reconstruct/main.py:1107 passes `voxel_origin=(-recon_grid_origin,)*3` in good faith and it is thrown away. The parameter is only honoured on the fallback branch (:690-695). The 33-line Args block (:600-627) does not say this.

**NSM/mesh/main.py:836 — get_sdfs prints one unconditional line per batch in the production path**

`print(f"Processed {current_idx} / {n_pts_total} points (batch {batch_num+1}: CNN+MLP, size={current_batch_size})")` has no `verbose` gate, and neither does the WARNING at :821-823. get_sdfs is on the live path kneepipeline/steps/run_nsm.py:170 -> NSM.reconstruct.reconstruct_mesh -> create_mesh_adaptive -> get_sdfs (:662, :722). Every subprocess NSM fit spams stdout; the kneepipeline orchestrator parses the LAST stdout line as JSON, so this is noise directly in a machine-read stream.

**NSM/mesh/main.py:862 — decode_sdf's fast path passes an unbatched latent, the legacy path an expanded one**

`return decoder(latent=latent_vector.squeeze(), xyz=queries)` (:862) hands the decoder a (D,) latent, while the legacy branch three lines later hands it `latent_vector.expand(num_samples, -1)` -> (N, D) (:865-867). Undocumented in the docstring (:843-850). `.squeeze()` is also a silent no-op for a (B, D) latent with B > 1, and it would squeeze a latent of dim 1 out of existence. Additionally, `inspect.signature(decoder.forward)` is re-evaluated on EVERY batch (:859).

### `NSM/mesh/refine_mesh.py`

**NSM/mesh/refine_mesh.py:239 — add_vertex_if_new returns a tuple, its docstring promises an int, and its index lives in a third array's space**

Docstring Returns: 'The index of the vertex in the combined list' (:239-241). It actually returns `(new_vertices, index)` (:257, :261). Worse, the index is into the CONCATENATED `[mesh.points; new_vertices]` array built locally at :245 — an array that does not exist anywhere else until `update_mesh` reconstructs it in exactly the same order at :312. That cross-function index-space contract is stated nowhere. The `threshold=1e-10` parameter (:229) is also undocumented. This is precisely the audited bug class.

**NSM/mesh/refine_mesh.py:278 — create_new_faces depends on an unstated midpoint ordering produced two functions away**

`AB, BC, CA = midpoint_indices` (:278) requires the list to be in the edge order (0,1), (1,2), (2,0). That order is produced only as a side effect of the `for i in range(3)` loop in new_vertices_faces (:100-106) feeding `midpoint_indices.append(...)` (:120). Neither docstring names the required order. Reordering that loop silently produces an incorrectly triangulated mesh, not an error.

**NSM/mesh/refine_mesh.py:399 — get_target_cells raises UnboundLocalError on its own default arguments**

`max_length_binary = np.zeros_like(max_length_binary)` references the name it is assigning. When `max_length_threshold is None` (the default for every caller) this raises `UnboundLocalError: local variable 'max_length_binary' referenced before assignment`. Verified empirically: `get_target_cells(sphere, area_threshold=0.5)` -> UnboundLocalError; `get_target_cells(sphere, area_threshold=0.5, max_length_threshold=1.0)` -> OK. This makes BOTH public entry points of refine_mesh.py (`subdivide_large_triangles`:412, `subdivide_triangles_on_base_mesh`:438) unusable unless the caller explicitly passes max_length_threshold. Strong evidence the module has not been executed since the linting commit aa48fcc (2025-08-24).

**NSM/mesh/refine_mesh.py:465 — subdivide_triangles_on_base_mesh assumes two meshes share cell indexing**

`cells_to_divide = get_target_cells(mesh, ...)` (:465) then `mesh_ = subdivide_triangles(base_mesh, cells_to_divide)` (:471) — the cell indices computed on `mesh` are applied to `base_mesh`. This is only valid if the two meshes have identical connectivity and cell ordering. The docstring (:446-463) says only that 'The base mesh is usually the original mesh before it was interpolated' and never states the index-correspondence requirement. It also silently adds a 'cell_color' cell array (:473-478) that the docstring does not mention.

### `NSM/mesh/triangle_metrics.py`

**NSM/mesh/triangle_metrics.py:37 — Undocumented edge ordering shared by two modules with no cross-reference**

`get_edge_lengths` returns [len(p0,p1), len(p1,p2), len(p2,p0)] (:37-42), so `TriangleProperties.edge_lengths[:, k]` means edge k of the (i, (i+1)%3) sequence (:63-70). refine_mesh.py:100-106 independently reproduces the identical ordering when building its `edges` dict, and create_new_faces (:264-284) depends on that order being [AB, BC, CA]. Nothing documents the convention or links the two implementations.

**NSM/mesh/triangle_metrics.py:51 — TriangleProperties.areas returns a dimensionless deviation, not areas, by default**

`def areas(self, norm=True)` returns `(self._areas - ref_area) / ref_area` (:56-57) unless norm=False. The name says areas; the default output is a mean-relative deviation centred on zero (verified: sphere gives -0.434 with norm=True, 0.0171 with norm=False). No docstring anywhere in the class. Consequence downstream: refine_mesh.py:380 calls `areas(norm=True)` and compares to `area_threshold`, whose docstring (refine_mesh.py:370, 419, 455) says 'The maximum area of a triangle before it is subdivided' — false; it is a relative deviation.

### `NSM/models/deep_sdf.py`

**NSM/models/deep_sdf.py:171 — progressive_add_depth path propagates None through the layer stack**

When the current epoch precedes a progressive layer's start_epoch, forward_branch_ executes a bare `return` (line 171), yielding None. forward assigns that to x (line 216) and then `if x is None: continue` (lines 218-219) moves to the next layer with x still None. Verified: `Decoder(latent_size=8, dims=[16]*8, progressive_add_depth=True)` called with epoch=10 raises `TypeError: linear(): argument 'input' (position 1) must be Tensor, not NoneType`. The whole progressive-depth feature is non-functional as written.

**NSM/models/deep_sdf.py:180 — Decoder indexes self.bn by absolute layer index but appends only for norm layers**

`self.bn` is appended to only for layers listed in norm_layers (line 137), so bn[k] is the k-th NORM layer, while forward reads `self.bn[layer_idx]` (line 180) using the absolute layer index. Verified: `Decoder(latent_size=8, dims=[16,16,16], weight_norm=False, norm_layers=(2,))` builds with len(bn)==1 and raises `IndexError: index 2 is out of range` on the first forward. Unreachable whenever weight_norm=True (the elif at line 136 skips bn entirely), which is why it has survived.

**NSM/models/deep_sdf.py:308 — activation='linear' builds a Decoder that crashes on first forward**

get_activation returns None for 'linear'. Decoder guards for that on final_activation (line 224) but not on the hidden activation (line 181). Verified: `Decoder(latent_size=8, dims=[16,16], activation='linear')` constructs fine and then raises `TypeError: 'NoneType' object is not callable`. 'linear' is an advertised value — loader's template lists 'relu', 'leaky_relu', 'sin', etc. at loader.py:307 and 'linear' as a final_activation option at line 306.

### `NSM/models/loader.py`

**NSM/models/loader.py:119 — Decoder output column -> surface identity is nowhere in the models package**

The loader reads `objects_per_decoder` into `n_objects` (lines 119, 153, 218) but never reads or attaches `mesh_names`. Grep confirms `mesh_names` appears zero times under NSM/models/. So the mapping from SDF output column index to anatomical surface (bone / cart / med_men / lat_men, per CLAUDE.md) exists only in the training config and in downstream convention. A model returned by load_model is indistinguishable from one with a different surface order. This is the same undocumented-positional-ordering hazard the audit was called to find.

### `NSM/models/triplanar.py`

**NSM/models/triplanar.py:87 — VAEDecoder builds no activation functions between its conv layers**

`activation = activation_fn()` constructs an activation module and immediately discards it — it is never appended to self.layers (contrast lines 74 and 85, which do append). It also rebinds the loop-invariant `activation` string parameter to a module, so the name means two different things across iterations. Verified by printing a built VAEDecoder: the Sequential is (ConvTranspose2d -> norm) x N -> (Conv2d + Tanh), with **zero pointwise nonlinearities** between conv layers. The `activation` constructor argument (line 34) has no effect.

**CORRECTION (2026-08-15).** This entry originally concluded "the entire feature-plane generator is therefore an affine map plus a final Tanh." That is FALSE for the shipped models and was falsified by ~20 lines of torch. Measured additivity error of the latent -> pre-Tanh map in eval mode: `conv_norm_type="layer"` (which BOTH shipped models, 647 and 551, use) gives 3.74e+00 against a value scale of 8.44e+00 -- **not affine**, because LayerNorm divides by a standard deviation computed from its own input and is nonlinear in its own right. It silently supplies the nonlinearity the missing activation was meant to provide. Only `conv_norm_type="batch"` (2.89e-07) and `conv_norm=False` (2.92e-07) are genuinely affine at eval. The production models work, and work by accident.

The sharper hazard this exposes: with `norm_type="batch"` -- **the VAEDecoder constructor default** -- the model TRAINS nonlinear (batch statistics couple samples; additivity error 4.08) and EVALUATES affine. The function fit is not in the same expressive class as the function deployed. In all configurations the depth is still largely wasted: N stacked ConvTranspose2d with no activation between them buy far less than N layers of a normal decoder. Every shipped triplanar checkpoint was trained with this topology, so fixing it changes the architecture and invalidates existing weights. Related: if `activation` is neither 'leakyrelu' nor 'relu', `activation_fn` is never bound (lines 49-52) and line 87 raises NameError rather than a clear error.

**NSM/models/triplanar.py:99 — VAEDecoder registers every submodule twice, doubling checkpoint keys**

`self.layers` (nn.ModuleList) and `self.decoder = nn.Sequential(*self.layers)` both register the same modules. Verified on a small model: 16 state_dict keys under vae_decoder.layers.* and 16 identical-shaped duplicates under vae_decoder.decoder.*. The parameters are shared so training is unaffected, but every shipped .pth carries the VAE twice, and any cleanup that removes self.layers (the natural refactor) makes strict load_state_dict reject every existing checkpoint — including the production femur models the consumer loads at kneepipeline/steps/run_nsm.py:114.

**NSM/models/triplanar.py:158 — Latent gradients are scaled by the number of query points**

Both latent paths hand back N times the gradient of the mathematically equivalent single-use computation, where N is the number of query points. Verified at N=10: fast path and legacy path both give exactly 10x a manual reference that decodes the latent once. FastUnique.backward reaches that result by returning a (num_points, D) gradient for a (D,) input (lines 160-163) and relying on autograd's implicit sum-to-size reduction — a shape mismatch that torch tolerates rather than an intentional contract. The two paths agree, so this is not a regression, but it means the effective learning rate on latent codes scales with points-per-batch, and the docstring's claim that FastUnique "provides the same gradient expansion as unique_consecutive" (line 150) hides both facts.

**NSM/models/triplanar.py:197 — The consumer's hand-rolled param mapping omits padding**

kneepipeline/steps/run_nsm.py:94-111 reconstructs the config->constructor mapping by hand and passes 15 of TriplanarDecoder's 16 meaningful arguments — it never passes `padding`, so the model always uses 0.1. `padding` is not a learned parameter, so a checkpoint trained with a different value loads cleanly under strict load_state_dict and then samples the feature planes at the wrong scale, with no error. Nothing in NSM/ writes `padding` into a saved config today (grep finds it only in NSM/models/), so this is latent rather than active — but the duplicated mapping in the consumer is the reason a new constructor argument can be added here and silently ignored there.

### `NSM/models/two_stage.py`

**NSM/models/two_stage.py:65 — TwoStageDecoder permanently corrupts its own module-level default dicts**

Lines 65-68 write latent_dim/n_objects/latent_size into the dicts passed as `triplanar_params`/`mlp_params`, which default to the module-level `default_triplanar_params` (line 7) and `default_mlp_params` (line 22). Verified: after a single `TwoStageDecoder(latent_size=8, n_objects=5)`, default_triplanar_params['latent_dim'] is 4 and ['n_objects'] is 5 process-wide, so a subsequent default-constructed model silently inherits the previous model's geometry. The mutation happens before construction, so it persists even when construction then fails.

### `NSM/reconstruct/__init__.py`

**NSM/reconstruct/__init__.py:1 — `from .main import *` with no __all__ leaks the entire main.py import namespace onto the package**

`NSM.reconstruct` publicly exposes `os`, `sys`, `torch`, `np`, `wandb`, `copy`, `time`, `mskt`, `logging`, `logger`, `fnmatch`, `sinkhorn`, `create_mesh_adaptive`, `combine_meshes`, `eikonal_loss`, `read_mesh_get_sampled_pts`, `read_meshes_get_sampled_pts`, and `adjust_learning_rate` alongside the intended API (verified by `dir(NSM.reconstruct)`). Any of these can be shadowed or accidentally depended on by a downstream `from NSM.reconstruct import *`.

### `NSM/reconstruct/cartilage_func.py`

**NSM/reconstruct/cartilage_func.py:50 — cartilage_func's mesh slicing is a hardcoded positional layout with no validation**

`compare_cart_thickness_whole_joint` assumes meshes arrive as exactly [femur_bone, femur_cart, tibia_bone, tibia_cart, patella_bone, patella_cart] via `[:2]`, `[2:4]`, `[4:6]` (lines 53-72), while the tibia/patella/femur single-joint variants all slice `[:2]` (lines 26, 35, 44) and differ only in the label set they look up. Nothing checks the length or identity of the incoming lists; a 4-surface model (bone, cart, med_men, lat_men -- the `mesh_names` example in CLAUDE.md) silently produces wrong numbers. Undocumented, no docstrings.

### `NSM/reconstruct/main.py`

**NSM/reconstruct/main.py:248 — sdf_gt is mutated in place through the type-check and preprocess helpers**

`reconstruct_latent_preprocess_sdf_gt` writes `sdf_gt[sdf_idx] = sdf.to(device)` into the caller's list and returns it (line 248-249); `reconstruct_latent_sdf_gt_type_check` returns the caller's list unwrapped when it is already a list (line 176-177). reconstruct_mesh passes `result_['sdf']` straight through (line 1010, 1048) and separately mutates it at lines 1023 and 1026. Neither helper has a docstring, so a caller reusing the same sdf list for a second fit gets tensors already pinned to the first call's device. `pts_surface` is likewise re-typed and moved to device in place-ish fashion at lines 198-211.

**NSM/reconstruct/main.py:253 — project_latent is labelled legacy but is still the only path honoured under LBFGS-with-hard-constraint**

Docstring: 'Legacy explicit projection function - use latent_norm_penalty for smoother optimization'. It is still called at lines 709 and 736 whenever `use_soft_norm_constraint=False`. .claude/plans/HYBRID_OPTIMIZER_REPORT.md:78 records a sweep that set `latent_norm` and `norm_penalty_weight` but not `use_soft_norm_constraint`, and therefore silently took the soft path while the author believed otherwise -- i.e. the two mechanisms are already documented to have confused their own author. `project_latent` also mutates `latent` in place under `no_grad` (line 266) with no note that it does so.

**NSM/reconstruct/main.py:536 — pts_surface encoding is an undocumented positional contract**

`pts_surface` must be a per-point integer array whose values index into `sdf_gt` in the same order (`(pts_surface == surface_idx)` at lines 536 and 555, against `range(len(sdf_gt))`). Nothing documents this: `reconstruct_latent` has no docstring at all, and `reconstruct_latent_pts_surface_type_check` (line 198) only checks the container type. A caller who orders sdf_gt differently from the surface ids gets a silently wrong fit rather than an error.

**NSM/reconstruct/main.py:605 — Single-object decoder branch indexes sdf_gt by decoder index**

When `pred_sdf.shape[1] == 1` the loss uses `sdf_gt_[decoder_idx]` -- the *decoder* index used as a *surface* index. This is correct only under the unstated rule 'each decoder emits exactly one surface and decoder i owns surface i'. The multi-output branch below (line 612) correctly iterates `sdf_idx` over `pred_sdf.shape[1]` but then also indexes the flat `sdf_gt_` by it, so with two multi-output decoders the second decoder re-reads the first decoder's surfaces. The docstring on reconstruct_mesh (lines 864-869) gestures at the flattening rule but reconstruct_latent, which implements it, documents nothing.

**NSM/reconstruct/main.py:794 — reconstruct_latent can return an unbound local `latent_`**

`return loss, latent_` -- `latent_` is only bound inside the loop body (lines 771, 782, 792). With `convergence='overall_loss'` or `'recon_loss'`, binding happens only when the loss improves on the running best (initialised to the literal 100 at lines 464-465). If the very first steps produce NaN, `loss_ < loss` is False, `patience` increments, and once `patience > convergence_patience` the loop breaks at line 778/789 with `latent_` never assigned -> `UnboundLocalError`. Same failure if `actual_num_iterations` is 0. The production consumer sets convergence from `model_config['convergence_type_recon']` (kneepipeline/steps/run_nsm.py:205), so this path is reachable in production.

**NSM/reconstruct/main.py:840 — `chamfer_norm` is a power, not a norm, and its default disagrees across the three layers**

reconstruct_mesh defaults `chamfer_norm=2` (line 840) and get_mean_errors also defaults 2 (line 1281), but compute_recon_loss defaults `chamfer_norm=1` (recon_evaluation.py:26) and passes it straight through as `power=` to `compute_chamfer` (recon_evaluation.py:81, utils.py:100 `np.mean(d1**power) + np.mean(d2**power)`). With the shipped default the returned `chamfer_*` values are squared distances (mm^2), not distances -- undocumented anywhere, and the name suggests an Lp norm on the coordinate difference rather than an exponent on the resulting distance.

**NSM/reconstruct/main.py:873 — The consumer's `batch_size_latent_recon` is a no-op absorbed by **kwargs, while the real `batch_size` is left at its default**

`batch_size_latent_recon` was removed from the signature (commented out at line 804) and now only triggers a printed deprecation warning. kneepipeline/steps/run_nsm.py:199 still passes it from `model_config`, as do NSM_analysis.py, NSM_analysis_bone_only.py, and get_mean_errors at line 1326 (so the warning prints once per validation mesh per validation epoch during training). Meanwhile the live `batch_size=32**3` parameter (line 803) -- which controls the marching-cubes chunking in `create_mesh_adaptive` (lines 926, 1116) -- is a different knob that nobody sets. A reader tuning memory via config will change a value that does nothing.

**NSM/reconstruct/main.py:1118 — Undocumented mesh-index ordering is the load-bearing contract between reconstruct_mesh and its consumer**

`meshes` is flattened decoder-major then object-major (lines 1118-1123, comment 'append sequentially so they match the order of meshes at path'), and `compute_recon_loss` emits `chamfer_{i}` / `assd_{i}` / `emd_{i}` keyed by that same position (recon_evaluation.py:84, 103, 114). The consumer hardcodes index 0 = bone and index 1 = cartilage (kneepipeline/steps/run_nsm.py:216, 220, 232, 235). Nothing in the signature, the docstring, or the returned dict names the surfaces. The repo already has a `mesh_names` config field for exactly this (CLAUDE.md 'Multi-Surface Config'), saved into `model_params_config.json`, and reconstruct_mesh never reads or validates it. This is the same shape as the param-group ordering bug that motivated the audit.

**NSM/reconstruct/main.py:1140 — reconstruct_mesh switches return type between a list and a dict based on seven unrelated flags**

Lines 1140-1199: if any of `calc_emd`, `calc_symmetric_chamfer`, `calc_assd`, `return_latent`, `func is not None`, `return_registration_params`, `return_timing` is truthy the function returns a dict; otherwise it returns a bare list of meshes. The docstring (lines 861-870) does not mention the return value at all. Every first-party caller happens to trip the dict branch (get_mean_errors forces `return_latent=True` at line 1302; the consumer sets return_latent, calc_assd and return_registration_params), so the list branch is effectively untested.

### `NSM/reconstruct/utils.py`

**NSM/reconstruct/utils.py:104 — Two unrelated functions named adjust_learning_rate; the reconstruct one is re-exported from the NSM.reconstruct package namespace**

NSM/reconstruct/utils.py:104 `adjust_learning_rate(initial_lr, optimizer, iteration, decreased_by, adjust_lr_every)` -- step-decay for the *latent-fit* loop, sets EVERY param_group to one lr, no docstring. NSM/utils.py:202 `adjust_learning_rate(lr_schedules, optimizer, epoch, verbose=False)` -- the *training* per-target scheduler, raises KeyError on any group missing `target`. Different arity, different semantics, same name. Because NSM/reconstruct/main.py:4 does `from .utils import adjust_learning_rate` and NSM/reconstruct/__init__.py:1 does `from .main import *` with no `__all__`, `NSM.reconstruct.adjust_learning_rate` resolves to the reconstruct/utils version -- verified by introspection. NSM/train/train_deep_sdf.py imports the NSM.utils one at line 3 and imports from NSM.reconstruct by explicit name at lines 13-20, so it is safe today. Change line 13 to a star import and `adjust_learning_rate(config['lr_schedules'], optimizer, epoch)` at train_deep_sdf.py:289 silently rebinds to the 5-arg version and fails with a confusing `missing 2 required positional arguments` instead of running the schedule. This is a live footgun in exactly the area docs/KNOWN_ISSUES.md was written about.

**NSM/reconstruct/utils.py:104 — Two different `adjust_learning_rate` functions in the same package; the reconstruct one shadows via star-import**

`NSM.utils.adjust_learning_rate(lr_schedules, optimizer, epoch, verbose=False)` (NSM/utils.py:202) and `NSM.reconstruct.utils.adjust_learning_rate(initial_lr, optimizer, iteration, decreased_by, adjust_lr_every)` (NSM/reconstruct/utils.py:104) are unrelated functions with incompatible signatures. `NSM/reconstruct/__init__.py:1` (`from .main import *`, and main.py:4 imports the reconstruct one) re-exports the reconstruct variant, so `from NSM.reconstruct import adjust_learning_rate` silently gives you the step-decay one. Verified by introspection: `adjust_learning_rate` appears in `dir(NSM.reconstruct)`. Anyone who reaches for the name after reading the Aug 2026 LR docs gets the wrong function with no error until the call fails on arity.

### `NSM/train/train_deep_sdf.py`

**NSM/train/train_deep_sdf.py:108 — The shipped default_config.json cannot drive train_deep_sdf — six unconditionally-read keys are missing**

NSM/configs/default_config.json has 61 keys and is pinned to its generator by testing/NSM/configs/test_default_config_sync.py, but train_deep_sdf reads these unconditionally and they are absent from it: `prefetch_factor` (line 108, DataLoader construction), `profiler` (via get_profiler, NSM/train/utils.py:116, reached at line 162), `save_frequency` (line 183, every epoch), `verbose` (line 310, every batch), `code_regularization_warmup` (line 544 — reached because default_config sets `code_regularization: True`), and `assd` (line 230, on the first validation epoch). Anyone who starts from the shipped default hits a chain of KeyErrors. `config.setdefault` is applied to only 6 keys (lines 56-61) and none of these are among them.

**NSM/train/train_deep_sdf.py:121 — resume_epoch == 1 silently skips epoch 1 without resuming anything**

The resume block is guarded by `if config["resume_epoch"] > 1:` (line 121), but the epoch loop is `range(config["resume_epoch"] + 1, config["n_epochs"] + 1)` (line 164). With `resume_epoch=1` the loop starts at epoch 2 while no checkpoint is loaded — the run silently trains from random initialization one epoch short, with no warning. The two migration guards below (lines 138-156) also never run for that value.

**NSM/train/train_deep_sdf.py:195 — schedule_free eval warm-up passes a (dict, tensor) dataloader tuple straight into the decoder**

Lines 193-195: `with torch.no_grad(): for batch in itertools.islice(data_loader, 50): model(batch)`. The dataset's `__getitem__` returns `data_, idx` (NSM/datasets/sdf_dataset.py:1569, 2164), so `batch` is a 2-tuple of (dict-of-tensors, index tensor). Decoder forwards expect a single tensor: DeepSDF does `xyz = input_[:, -3:]` (NSM/models/deep_sdf.py:195) and TriplanarDecoder takes `x=None, latent=None, xyz=None` (NSM/models/triplanar.py:330). `model((dict, tensor))` raises TypeError. This block runs on the FIRST checkpoint or validation epoch for any `schedule_free_*` optimizer, so the whole schedule-free path fails as soon as it reaches a checkpoint. The commented-out `raise Exception('HOW TO IMPLEMENT BATCH NORM FIX? ...')` on line 192 shows the block was never finished. Also note it re-iterates the dataloader mid-epoch, re-paying data-loading cost for 50 batches.

**NSM/train/train_deep_sdf.py:198 — save_model_params is called on every checkpoint but silently no-ops after the first**

Line 198 calls `save_model_params(config=config, list_mesh_paths=...)` inside the per-checkpoint block, implying the recorded config is kept current. `save_model_params` returns immediately if the file already exists (NSM/utils.py:312-313). On a resumed run (lines 121-159) the config genuinely differs from the original — `resume_epoch` changes, and any hyperparameter the user edited before resuming is silently not recorded — yet model_params_config.json keeps the first run's values, and that file is exactly what downstream consumers read to reconstruct the model.

**NSM/train/train_deep_sdf.py:333 — The per-surface index ordering is a fully undocumented positional contract spanning four modules**

`surf_idx` is a bare integer that must mean the same thing in five independent places, with nothing cross-checking any of them: (a) `sdf_data["gt_sdf"][:, :, surf_idx]` (line 333) — axis 2 is the dataset's mesh order, set by `loc_meshes` enumeration at NSM/datasets/sdf_dataset.py:1803,1860; (b) `pred_sdf[:, surf_idx]` (line 416) — the decoder's output column order; (c) `config["surface_weighting"][l1_idx]` (lines 482-490) — a user-supplied list whose only validation is a length assert (line 478); (d) `config["mesh_names"][i]` — validated for LENGTH only (lines 64-70), never for correspondence; (e) the wandb metric names `l1_loss_{idx}` (line 620). Swapping two meshes in the dataset path list silently trains bone weights against cartilage targets and mislabels every logged metric. This is the same class of bug as the [0]=model/[1]=latent param-group ordering that was just fixed in NSM/utils.py — it survives untouched here.

**NSM/train/train_deep_sdf.py:433 — surface_accuracy curriculum is inverted relative to sample_difficulty, so schedule='constant' disables it entirely**

Line 433 computes `weight_schedule = 1 - calc_weight(...)` for the surface-accuracy epsilon, while line 449 uses `calc_weight(...)` directly for sample difficulty. Both read a user-supplied `*_schedule` string handled by the same `calc_weight` (NSM/train/utils.py:10-26). Consequence: `"surface_accuracy_schedule": "constant"` returns 1.0, so `weight_schedule` becomes 0 and the epsilon is never subtracted — 'constant' turns the feature OFF here but full ON at line 449. `calc_weight` has no docstring and nothing records the sign convention.

**NSM/train/train_deep_sdf.py:575 — train_epoch hard-requires optional data-loading telemetry keys from the dataset**

Lines 575-578 unconditionally read `sdf_data["size"]`, `["time"]`, `["mb_per_sec"]`, `["whole_load_time"]`. SDFSamples only populates them when `(self.test_load_times is True) and (self.store_data_in_memory is False)` (NSM/datasets/sdf_dataset.py:1563-1567). The defaults (store_data_in_memory=False at :834, test_load_times=True at :836) happen to satisfy this, but `store_data_in_memory=True` is a supported constructor option — turning it on silently converts training into a KeyError on the first batch. Nothing in train/ documents that the training loop depends on a profiling flag of the dataset. MultiSurfaceSDFSamples gates on `test_load_times` alone (:2158), a third variant of the same condition.

### `NSM/train/utils.py`

**NSM/train/utils.py:63 — add_plain_lr_to_config retains a positional back door that a test pins to mislabelled output**

The Aug-2026 fix keyed schedules by their declared `Target` (NSM/utils.py:87-136), and the docstring at lines 65-70 says labels 'always carry the correct labels'. But the `idx_model` / `idx_latent` parameters survive as an explicit override (line 71: `if idx_model is None or idx_latent is None`), bypassing `resolve_schedule_targets` entirely. testing/NSM/test_lr_schedules.py:535-538 (`test_explicit_indices_still_override`) asserts that `add_plain_lr_to_config(make_config(), idx_model=1, idx_latent=0)` yields `config["model_lr_initial"] == LATENT_LR` — i.e. the suite now pins the swapped-label behaviour as intended for this path. No caller in the repo passes these arguments (train_deep_sdf.py:80, multi_head:54, and both deprecated files all call it with config alone).

### `NSM/utils.py`

**NSM/utils.py:283 — save_model's on-disk subdirectory naming is an undocumented contract**

Lines 283-287: a single decoder is written to `<experiment_directory>/model/<epoch>.pth`, but a list of two or more is written to `model_0/`, `model_1/`, ... The docstring (lines 255-262) documents only the optimizer-target validation and says nothing about this. Any loader must reproduce the branch exactly, and the boundary case (a one-element list) silently takes the singular path.

**NSM/utils.py:312 — save_model_params silently refuses to overwrite and silently drops non-JSON config values**

Lines 312-313 `if os.path.exists(path_save): return` — a re-run or resumed experiment in the same directory keeps the FIRST run's config file, so the saved model_params_config.json can describe different hyperparameters than the run that produced the weights. Line 320 then runs `filter_non_jsonable`, dropping every config key whose value will not serialize, with no log of what was dropped. No docstring on the function.

**NSM/utils.py:343 — get_latent_vecs silently ignores config['latent_bound'] and doubles latent_size when variational**

Lines 343-348: when `config['variational']` is True the function uses `latent_size = config['latent_size'] * 2` and hard-codes `latent_bound = 1000`, discarding the configured `latent_bound` entirely. The function has no docstring, so a user who sets `latent_bound: 1.0` (the shipped default, default_config.json:109) and `variational: true` gets 1000 with no warning, and every downstream consumer of `latent_size` sees a different number than the config says.

### `pyproject.toml`

**pyproject.toml:95 — pytest testpaths points at a directory that does not exist**

`testpaths = ["tests"]` but the tests live in `testing/` (CLAUDE.md: 'Tests live in testing/ directory (not tests/)', Makefile:33: `pytest testing/ -v`). A bare `pytest` in the repo root collects nothing and exits clean, which reads as a green run.

---

## Defects (58)

Raises, or is unambiguously incorrect.

### `NSM/datasets/sdf_dataset.py`

**NSM/datasets/sdf_dataset.py:301 — uniform_pts_buffer expands the max side more than the min side**

`mins = mins - uniform_pts_buffer/2*(maxs-mins)` then `maxs = maxs + uniform_pts_buffer/2*(maxs-mins)` -- the second line uses the ALREADY-SHRUNK mins, so the max side is expanded by a factor (1 + buffer/2) more than the min side, making the 'cube' asymmetric about the mesh. Identical copy-paste at 647-648. Harmless at the default 0.0; wrong for any nonzero buffer, which is exactly what commit 48c5f60 added the parameter for.

**NSM/datasets/sdf_dataset.py:308 — Uniform-sample clipping exists only in the single-mesh function**

read_mesh_get_sampled_pts clips random points to +/-(1 + uniform_pts_buffer/2) when norm_pts is True (308-310). read_meshes_get_sampled_pts has no equivalent -- line 662 samples the cube and the clip never happens. Two functions that are otherwise parallel diverge silently on the bound of the sampled domain.

**NSM/datasets/sdf_dataset.py:665 — include_surf_in_pts in the multi-mesh path concatenates a leaked loop variable**

Inside `for new_pts_idx, new_mesh_ in enumerate(new_meshes):` (line 650) the body does `rand_pts_ = np.concatenate([rand_pts_, new_pts_], axis=0)` at 665. `new_pts_` is not bound by that loop. It leaks from whichever earlier statement ran last: line 644 (`new_pts_ = [x for x in new_pts if x is not None]`, a LIST, when any sigma is None), or line 618's loop (the LAST mesh's pre-normalization array), or is unbound entirely (NameError) if neither centering nor uniform sampling ran. It is never the current surface's points, which is obviously the intent -- the single-mesh version does it correctly at line 306. Reachable from production config: NSM/reconstruct/main.py:1000 passes include_surf_in_pts=get_rand_pts. Currently dormant because both shipped model configs have get_rand_pts_recon=false; flipping that flag activates the bug.

**NSM/datasets/sdf_dataset.py:671 — pts_surface concatenation raises ValueError when any surface is allocated zero points**

Line 668 appends a 1-D Python list `[new_pts_idx] * N`; line 671 appends a 2-D `np.zeros((0, 3))` for the n_pts == 0 case. np.concatenate at 678 then mixes 1-D and 2-D inputs. Runtime-verified: numpy raises 'all the input arrays must have same number of dimensions'. Reachable whenever a per-mesh n_pts entry is 0, which get_pt_sample_combos can produce for the random bucket (1952-1957) if p_near + p_far == 1.0 -- a combination that check_probabilities_sum explicitly permits (767).

**NSM/datasets/sdf_dataset.py:849 — check_probabilities type gate rejects integer probabilities**

The elif at 849 is `isinstance(p_near_surface, float) & isinstance(p_further_from_surface, float)`. p_near_surface=0 or 1 (perfectly valid probabilities, and what a JSON config emits for whole numbers) is an int, matches neither branch, and falls through to the ValueError at 854-856 claiming the values 'must be floats or lists/tuples of floats'. Note also that both branches use bitwise `&` rather than `and` (842, 849).

**NSM/datasets/sdf_dataset.py:989 — os.sched_setaffinity is called unguarded on a platform-conditional API that is guarded 55 lines earlier**

Line 933-936 wraps os.sched_getaffinity in try/except AttributeError with the comment that it is unavailable on mac/windows. Line 989 then calls os.sched_setaffinity(0, range(multiprocessing.cpu_count())) with no guard, inside every worker, whenever self.multiprocessing is True (the default). Same platform, opposite treatment.

**NSM/datasets/sdf_dataset.py:1046 — joint_scale_buffer is applied on the disk path and silently ignored on the in-memory path**

The store_data_in_memory=False branch inflates the joint radius by (1 + self.joint_scale_buffer) at line 1046 with the comment that this lets the model generalize to unseen larger data. The store_data_in_memory=True branch (1074-1082) computes max_radii and divides by it at 1085-1090 with no buffer at all. The same dataset config therefore produces two different normalizations depending on a flag that is nominally about memory, not geometry.

**NSM/datasets/sdf_dataset.py:1061 — ISSUE #1 REFUTED AS STATED, AND INVERTED: norm_and_scale_all_meshes works on disk and crashes in memory**

Open issue #1 alleges norm_and_scale_all_meshes cannot work when NOT loading in memory. The code says the opposite. The store_data_in_memory=False branch (1017-1054) np.load()s each cached .npz and reads data_['new_pts_{i}'] -- those keys DO exist, because save_data_to_cache flattens the 'new_pts' list into new_pts_0, new_pts_1... via get_dict_pts (1107-1114, 1129). Runtime-verified against a synthetic npz: the branch completes and sets self.max_radius/self.center. The store_data_in_memory=True branch (1056-1090) reads data['new_pts_{i}'] from the IN-MEMORY dict, but nothing ever puts those keys in an in-memory dict -- get_sample_data_dict stores data['new_pts'] as a LIST (line 1249 for SDFSamples, line 1853 for MultiSurfaceSDFSamples) and unpack_numpy_data also returns it under the single key 'new_pts' (line 397-399). Runtime-verified: `SDFSamples.norm_and_scale_all_meshes(stub)` with a get_sample_data_dict-shaped in-memory dict raises KeyError: 'new_pts_0' at line 1061. So scale_jointly=True + store_data_in_memory=True is a hard crash at construction time. It survived because the shipped production models were all trained with store_data_in_memory=False (see /mnt/data/programming/kneepipeline/NSM_MODELS/*/model_params_config.json).

**NSM/datasets/sdf_dataset.py:1487 — Joint (scale_jointly) normalization is skipped entirely when subsample is None**

The `xyz = (xyz - self.center) / self.max_radius; sdf = sdf / self.max_radius` block sits inside `if self.subsample is not None` (1487 -> 1546-1550, and 2081 -> 2143-2146). With subsample=None the raw unnormalized cached data is returned as-is at 1569 / 2164, still carrying the orig_pts/new_pts lists, so the returned dict has a different shape AND a different coordinate space than the subsampled case. MultiSurfaceSDFSamples defaults subsample to None (1598).

**NSM/datasets/sdf_dataset.py:1595 — joint_scale_buffer cannot be set on MultiSurfaceSDFSamples**

SDFSamples accepts joint_scale_buffer=0.1 (line 819) and uses it at 1046. MultiSurfaceSDFSamples.__init__ (1595-1632) neither accepts it nor forwards it in the super().__init__ call (1652-1681), so multi-surface datasets are permanently pinned to 0.1 and passing joint_scale_buffer=... raises TypeError. It is also absent from both get_hash_params implementations (1396-1409, 1973-1999).

**NSM/datasets/sdf_dataset.py:1598 — MultiSurfaceSDFSamples default subsample=None is unusable**

subsample defaults to None (1598), but run_before_loading_data (1702-1703) calls get_samples_per_sign, which does `int((n_pts_/self.total_n_pts) * self.subsample)` at 1895 -> TypeError on None. So the documented default cannot construct. SDFSamples by contrast makes subsample a required positional (808).

**NSM/datasets/sdf_dataset.py:1928 — remove_overlapping_points hard-codes a two-surface assumption**

The function signs the SDF columns to -1/0/+1, sums across surfaces (1921), and drops points where the sum == -2 (1928). That identifies 'inside both' only when there are exactly two non-None surfaces. With three surfaces, inside-all-three sums to -3 and inside-exactly-two sums to -1, so neither is removed; the local names out_out/out_in/in_in (1923-1925) encode the same two-surface assumption. CLAUDE.md documents 4-surface models (bone/cart/med_men/lat_men) as a supported config, and objects_per_decoder=2 is already shipped in production, so this scales wrong exactly where the library is heading.

**NSM/datasets/sdf_dataset.py:2158 — MultiSurfaceSDFSamples.__getitem__ raises UnboundLocalError when store_data_in_memory=True (verified)**

`time_` and `size` are only assigned in the store_data_in_memory=False branch (2050-2057). Line 2158 does `if self.test_load_times is True:` -- with no store_data_in_memory guard -- and then reads time_ and size at 2159-2161. test_load_times defaults to True (1621). The parent class got this right and guards on both flags at line 1563. Runtime-verified: `MultiSurfaceSDFSamples.__getitem__(stub, 0)` with in-memory data raises UnboundLocalError: local variable 'time_' referenced before assignment.

**NSM/datasets/sdf_dataset.py:2193 — combine_meshes returns a pyvista PolyData, not a pymskt Mesh, whenever it actually combines (verified)**

Runtime check: pymskt Mesh.__add__ is `return self.merge(dataset)` and yields <class 'pyvista.core.pointset.PolyData'> with no `save_mesh` and no `point_coords`. So combine_meshes returns a pymskt Mesh for an int (2183) or 1-element list (2186) and a bare PolyData for 2+ (2189-2193), contradicting its own 'Returns: Mesh' at 2175. Downstream: load_reference_mesh assigns the result straight to self.reference_mesh at line 1355 and then calls self.reference_mesh.save_mesh(...) at 1385 whenever multiprocessing=True (the default) -> AttributeError. read_meshes_get_sampled_pts avoids this only because it re-wraps with Mesh(combined_mesh) at 531. The repo's own test hides the inconsistency with a hasattr fallback (testing/NSM/datasets/test_multi_surface_registration.py:70-71).

### `NSM/dependencies/sinkhorn.py`

**NSM/dependencies/sinkhorn.py:49 — Unreachable duplicated type check in sinkhorn; max_iters is never type-checked**

Lines 49-50 repeat `if not isinstance(p, int)` but raise a TypeError worded for max_iters. Line 41 already caught non-int `p`, so lines 49-50 can never fire, and `max_iters` gets no type validation at all.

**NSM/dependencies/sinkhorn.py:92 — sinkhorn validates w_x's length twice and never validates w_y's**

Inside the `if w_y is not None:` block, line 92 reads `if w_x.shape[0] != x.shape[0]` while the error message it raises (lines 93-96) talks about y and w_y. The intended `w_y.shape[0] != y.shape[0]` check does not exist, so a wrongly sized w_y passes validation and fails later inside keops with an opaque shape error.

**NSM/dependencies/sinkhorn.py:107 — sinkhorn's default uniform weights cannot sum equal when the two point clouds differ in size, so it always raises**

With `w_x is None and w_y is None`, lines 108-110 set `w_x = ones(n)/n` (sum 1) and `w_y = ones(m)/m * (n/m)` (sum n/m). Lines 112-119 then raise ValueError whenever `|1 - n/m| > 1e-5`, i.e. whenever n != m. Verified numerically (n=5, m=7 -> sums 1.0 vs 0.714). `NSM/reconstruct/recon_evaluation.py:112` calls `sinkhorn(xyz_orig_, pts_recon_)` with no weights on an original mesh's points versus a marching-cubes reconstruction's points, which have no reason to match in count. Any EMD evaluation on unequal point clouds dies with a misleading 'weights do not sum to the same value' error.

### `NSM/losses.py`

**NSM/losses.py:13 — The eikonal loss is still untested, as CLAUDE.md says, while being wired into both live loss paths**

CLAUDE.md:133-136 states 'NOTE - EIKONAL LOSS HAS NOT BEEN TESTED.' Verified still true: `grep -rn losses testing/` returns nothing — no test file imports NSM.losses, and testing/NSM/mesh/test_interpolate.py:86's `test_baseline_lands_on_target_non_eikonal_field` is about an SDF field, not this loss. Meanwhile NSM/train/train_deep_sdf.py:506-514 and NSM/reconstruct/main.py:665-677 both add it to the optimized loss whenever `eikonal_weight > 0`. It is off by default (generate_sdf_default_config.py:52 / default_config.json:70 set 0.0), so the untested path is one config edit away from a production run.

**CORRECTION (2026-08-15).** This entry originally added: "I hand-checked that eikonal_loss forward+backward runs for 1, 2 and 3 surfaces." **That is false and was the more damaging of the two overstatements in this register, because it was asserted as an executed check.** Re-run against the shipped function: it raises `RuntimeError: Trying to backward through the graph a second time` for 1, 2 AND 4 surfaces. `losses.py:54` sets `retain_graph=False` on the last (or only) surface, freeing the forward graph that the double-backward graph still needs. Beyond that one-line bug, the production triplanar architecture cannot support the loss at all — it requires a second derivative through `grid_sample`, unimplemented in PyTorch on CPU and CUDA — and even on MLP architectures the loss opposes NSM's clamped training regime. **`eikonal_weight > 0` is now gated with `NotImplementedError` at both entry points** (pinned by `testing/NSM/test_losses.py`); repair is tracked in `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md` §8.2.

### `NSM/mesh/correspondence_metrics.py`

**NSM/mesh/correspondence_metrics.py:604 — score_correspondence's documented return shape omits its error branch**

Returns is documented as 'either the metric result dict / scalar, or {"skipped": True, "reason": ...}'. There is a third shape the function emits on eight separate paths: `{"error": str(exc)}` at :615, :629, :630, :637, :648, :655, :667, :682, :688. A consumer branching on 'skipped' will treat an error dict as a successful result.

### `NSM/mesh/interpolate.py`

**NSM/mesh/interpolate.py:98 — Three divergent hand-rolled decoder invocation conventions in one subsystem**

NSM/mesh/main.py:842 `decode_sdf` inspects `decoder.forward`'s signature and dispatches to the fast `decoder(latent=..., xyz=...)` interface when available (:858-862) — which it IS for TriplanarDecoder, the production model (NSM/models/triplanar.py:330). interpolate.py never does this: `sdf_gradients` (:98), `_sdf_step_eval` (:225) and `_sdf_only` (:249) all hard-code the legacy concatenated `model(torch.cat([latent, pos], dim=1))` form. Three copies of the same decision, two of which are already out of sync with the third. A decoder supporting only the fast interface would break half the subsystem.

**NSM/mesh/interpolate.py:473 — interpolate_common is public, 18 parameters, zero docstring**

The only undocumented public function in an otherwise thoroughly documented module. It is the shared implementation behind both documented wrappers, has two mutually exclusive execution paths selected by `is_mesh`, silently ignores `tangent_laplacian`/`faces` on the mesh path (inline comment only, :504-507), and raises a bare `Exception("Not implemented")` when `data is None` (:493-494) — which is the DEFAULT for both wrappers' `points1=None` (:581) and `mesh=None` (:647).

### `NSM/mesh/main.py`

**NSM/mesh/main.py:280 — sdf_grid_to_mesh crashes on numpy input while its VTK twin does not**

`sdf_values = sdf_values.cpu().numpy()` unconditionally (:280), versus `if hasattr(sdf_values, 'cpu'): sdf_values = sdf_values.cpu().numpy()` in sdf_grid_to_mesh_vtk (:398-399). The two are selected by the single `use_vtk` boolean at :241-244 and :750-753, so which input types are accepted depends on an unrelated flag.

**NSM/mesh/main.py:638 — Fallback grid origin can disagree with search_bounds**

create_mesh_adaptive derives `voxel_size` from `search_bounds` (:638-639) but the fallback (:690) hands `create_mesh` the untouched `voxel_origin`, whose default is (-1,-1,-1) (:559). NSM/reconstruct/main.py:919 calls with `search_bounds=(-recon_grid_origin, recon_grid_origin)` and NO voxel_origin, so if `recon_grid_origin != 1.0` the fallback grid is anchored at -1 while its spacing assumes an extent of 2*recon_grid_origin. Latent, not live: recon_grid_origin defaults to 1.0 (NSM/reconstruct/main.py:846).

### `NSM/mesh/refine_mesh.py`

**NSM/mesh/refine_mesh.py:46 — find_all_faces_to_split: docstring promises a 2-tuple, mutates its loop target, dead counters**

Docstring Returns lists two values, `cells_to_divide` and `list_adjacent` (:46-47); the function returns one array (:73). It appends to `cells_to_divide` at :68 while iterating over that same list at :59, so newly-appended faces are visited in the same pass and the result depends on traversal order. `unique` / `not_unique` (:54-55, :66, :69) are incremented and never read.

### `NSM/models/deep_sdf.py`

**NSM/models/deep_sdf.py:47 — xyz_in_all is accepted and documented but never used**

The parameter appears only twice in the file: in the signature (line 47) and in the docstring that describes it as functional (line 66, "for deepSDF decoder, include XYZ at each layer"). It is never stored on self and never read. loader.py:159 dutifully forwards config['xyz_in_all'] into it, and the shipped NSM/configs/default_config.json contains an `xyz_in_all` key — so a user can set it, see it plumbed through two layers of config translation, and have it do nothing.

**NSM/models/deep_sdf.py:241 — NameError disguised as an error path in progressive_layer**

`raise exception("Epoch is before start of progressive depth")` — `exception` (lowercase) is not a builtin and is not defined in the module. If this branch is ever reached the user gets `NameError: name 'exception' is not defined` instead of the intended message. Also note the branch below it uses a strict `start < self.epoch < end` (line 242), so epoch == start falls through to the else and applies the layer at full weight, skipping the warmup entirely.

### `NSM/models/loader.py`

**NSM/models/loader.py:123 — conv_norm_type default differs depending on which code path builds the triplanar model**

_get_triplanar_params defaults conv_norm_type to 'batch' (line 123) and the triplanar template says 'batch' (line 301), while _get_two_stage_params defaults the inner triplanar to 'layer' (line 189), the two_stage template says 'layer' (line 346), and two_stage.py:13 says 'layer'. BatchNorm2d and LayerNorm produce different state_dict keys (running_mean/running_var vs weight/bias), so a config that omits the key loads under one path and fails under the other.

**NSM/models/loader.py:228 — The 'implicit' config vocabulary is incompatible with real training configs**

_get_implicit_params requires latent_dim / hidden_dim / num_layers, while the other three extractors require latent_size (and layer_dimensions). Every saved model_params_config.json derives from the training config, which uses `latent_size` (NSM/configs/default_config.json). So `load_model(config, path, model_type='implicit')` on any real experiment directory raises KeyError. Only the hand-written template at lines 369-382 satisfies it, which is why the tests pass.

### `NSM/models/modulated_periodic_activations.py`

**NSM/models/modulated_periodic_activations.py:196 — ModulationNetwork concatenates in the opposite order from its docstring**

The __init__ docstring (lines 170-172) describes "the latent (in_dim) is concatenated with the output of the previous layer (mod_dims[i-1])" — latent first. forward does `torch.cat([out, input], dim=-1)` (line 196) — previous output first. The widths sum identically so nothing errors, but the learned weight columns are laid out opposite to the only written description of them.

**NSM/models/modulated_periodic_activations.py:211 — ImplicitDecoder defaults to a sigmoid output, which cannot represent a signed distance**

final_activation defaults to torch.sigmoid, range (0,1) — an SDF needs negative values inside the surface. loader.py:241 reinforces this by defaulting final_activation to 'sigmoid' for the implicit type, unlike every other type which defaults to 'tanh' (loader.py:128, 161, 194).

### `NSM/models/triplanar.py`

**NSM/models/triplanar.py:12 — That same block documents the plane channel order backwards**

Lines 12-15 state "the first 1/3 of the channels as features for the xy plane, the second 1/3 for the xz plane, and the last 1/3 for the yz plane". The code slices them as xz, yz, xy (lines 266-268). This is the same class of defect that motivated this audit: an index-ordering contract with no named constant, described incorrectly in the only place it is described. The ordering is baked into every trained checkpoint, so the comment is what must change, not the code.

**NSM/models/triplanar.py:262 — sum_sdf_features=False silently produces two empty feature planes**

When sum_sdf_features is False the VAE is sized to emit `sdf_latent_size` channels total (line 220), but forward_with_plane_features slices the planes as [0:L], [L:2L], [2L:] where L = sdf_latent_size (+conv_pred_sdf) (lines 262-268). Verified with sdf_latent_size=6: the three slices are (6,8,8), (0,8,8) and (0,8,8) — the yz and xy planes are empty tensors. The concatenation at line 281 still yields the width the SDF decoder expects, so the model builds, runs and trains with no error while two thirds of the triplanar representation are structurally absent. The correct divisor for the non-summing case is sdf_latent_size//3.

### `NSM/models/two_stage.py`

**NSM/models/two_stage.py:24 — TwoStageDecoder cannot be constructed with its own defaults**

`default_mlp_params['dims']` is a tuple, and deep_sdf.Decoder does `self.dims = [latent_size + 3] + dims` (deep_sdf.py:83). Verified: `TwoStageDecoder(latent_size=8, n_objects=5)` raises `TypeError: can only concatenate list (not "tuple") to list`. The loader's non-nested path dodges this by wrapping in list() (loader.py:204) and the template uses a list literal (loader.py:355), so the broken default is only reachable by a direct caller — and only after it has already corrupted the module-level dicts.

### `NSM/reconstruct/cartilage_func.py`

**NSM/reconstruct/cartilage_func.py:116 — compare_cart_thickness mutates the reconstructed meshes it is asked to evaluate**

Lines 116-121 copy the region scalars from `orig_bone` onto `recon_bone`, assign `recon_bone.list_cartilage_meshes = recon_cart`, and call `recon_bone.calc_cartilage_thickness()` -- which adds a 'thickness (mm)' array to the mesh. These are the same objects reconstruct_mesh returns in `result['mesh']` (main.py:1132 passes `meshes` directly). The comment at line 119 even says 'test to make sure doesnt cause issues', i.e. it was never checked. None of the five public functions in this module has a docstring.

### `NSM/reconstruct/main.py`

**NSM/reconstruct/main.py:178 — The string branch of the sdf_gt type check is unreachable; it raises TypeError instead of its message**

`elif type(sdf_gt) in (str):` -- `(str)` is not a tuple (no trailing comma), so this evaluates `type(sdf_gt) in str`, which raises `TypeError: argument of type 'type' is not iterable`. Verified in the nsm-dev env with `reconstruct_latent_sdf_gt_type_check('some/path.vtk')`. The intended, informative message at lines 179-182 ('Must provided xyz/sdf from mesh... Try reconstruct_mesh instead.') can never fire. testing/NSM/reconstruct/test_reconstruct_latent.py:92 uses `pytest.raises(Exception)`, so the test passes on the wrong exception and hides this.

**NSM/reconstruct/main.py:297 — latent_norm_penalty returns a Python float 0.0 inside the range, breaking .item() on the logged value**

Lines 297 and 312 set `penalty = 0.0` (a float) while every other branch returns a tensor; line 336 then returns `penalty_weight * penalty`. Downstream code guards with `hasattr(x, 'item')` at lines 748 and 764 -- so the guard exists precisely because the return type is inconsistent, and it is undocumented. The 'barrier' branch (line 316) can also return -inf/NaN when `current_norm` reaches a bound, with no clamping.

**NSM/reconstruct/main.py:393 — In hybrid-optimizer mode the LR decay interval is computed from the wrong iteration count**

`adjust_lr_every = reconstruct_latent_get_lr_update_freq(n_lr_updates, num_iterations)` is computed from `num_iterations`, but when `hybrid_optimizer=True` the loop runs `total_iterations = adam_iterations + lbfgs_iterations` (lines 431, 481-483) and `adam_iterations` itself defaults to `num_iterations` (line 427). So with any `lbfgs_iterations > 0` the requested `n_lr_updates` decays are spread over the wrong horizon. Also lines 500-516 duplicate the identical `adjust_learning_rate(...)` call in both arms of an if/else whose only difference is `current_optimizer` vs `optimizer`, which are the same object in the non-hybrid arm.

**NSM/reconstruct/main.py:445 — `optimizer` / `loss_fn` can be referenced before assignment for unrecognised names**

Lines 445-451: `optimizer` is only bound when `optimizer_name` is exactly 'adam' or 'lbfgs'; anything else falls through and the first use at line 497/510 raises UnboundLocalError with no hint of the real cause. Same at lines 454-460 for `loss_fn` when `loss_type` is not 'l1'/'l1_log'/'l2'. .claude/plans/HYBRID_OPTIMIZER_REPORT.md:149 records that this has already bitten people: model configs store `"optimizer": "AdamW"` while reconstruct_latent expects lowercase 'adam'.

**NSM/reconstruct/main.py:588 — Only TriplanarDecoder can actually be reconstructed; the other three loader targets cannot**

reconstruct's optimization loop calls `decoder(latent=latent.squeeze(0), xyz=xyz_input)` unconditionally (lines 588 and 671). Only TriplanarDecoder.forward accepts those keywords (triplanar.py:330); Decoder.forward (deep_sdf.py:191), TwoStageDecoder.forward (two_stage.py:76) and ImplicitDecoder.forward (modulated_periodic_activations.py:236) all take a single concatenated tensor and would raise TypeError. NSM/mesh/main.py:857-863 duck-types the signature before choosing an interface; reconstruct does not. So load_model advertises four architectures (loader.py:276, README.md:120-123) of which three cannot reach the reconstruction path.

**NSM/reconstruct/main.py:588 — reconstruct_latent calls decoders with keyword-only interface that only TriplanarDecoder implements**

`pred_sdf = decoder(latent=latent.squeeze(0), xyz=xyz_input)` (also line 671 for the eikonal path). Only NSM/models/triplanar.py:330 `forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False)` accepts those kwargs. NSM/models/deep_sdf.py:191 `forward(self, input_, epoch=None)`, NSM/models/two_stage.py:76 `forward(self, input, epoch=None)`, and NSM/models/modulated_periodic_activations.py:236 `forward(self, input_, epoch=None)` do not. Verified empirically in the nsm-dev env: `Decoder(latent_size=8, dims=[16,16])(latent=..., xyz=...)` -> `TypeError: forward() got an unexpected keyword argument 'latent'`. Consequence: `reconstruct_latent` -> `reconstruct_mesh` -> `get_mean_errors` (the training validation path, NSM/train/train_deep_sdf.py:213) raises TypeError for every non-triplanar model. Introduced in fee37be (2025-08-30). The production consumer is unaffected because it only uses TriplanarDecoder (kneepipeline/steps/run_nsm.py:85).

**NSM/reconstruct/main.py:919 — Mean-mesh generation and final-mesh generation call create_mesh_adaptive with different grid parameters**

The mean mesh (lines 919-928) passes only `search_bounds`, leaving `voxel_origin=(-1,-1,-1)` and `voxel_size=None` at their NSM/mesh/main.py:555-556 defaults, while the reconstruction (lines 1102-1117) explicitly derives `voxel_origin` and `voxel_size` from `recon_grid_origin`. If `recon_grid_origin != 1.0` the registration target and the reconstruction live on inconsistent grids. Nothing documents `recon_grid_origin` or this asymmetry.

**NSM/reconstruct/main.py:960 — Missing f-string prefix silently collapses per-mesh EMD results to one literal key**

`result["emd_{idx}"] = np.nan` inside `for idx in range(sum(objects_per_decoder))`. Every iteration writes the same literal key `emd_{idx}`; the neighbouring chamfer (line 954) and assd (line 957) branches correctly use f-strings. Downstream `get_mean_errors` reads `result_[f"emd_{mesh_idx}"]` (line 1382) and would KeyError. Only on the `mean_mesh is None` early-return path.

**NSM/reconstruct/main.py:1002 — mean_mesh is passed to the multi-object reader even when register_similarity is False**

The single-object call guards it (`mean_mesh=mean_mesh if register_similarity else None`, line 984) but the multi-object call at line 1002 passes `mean_mesh=mean_mesh` unconditionally, while still passing `register_to_mean_first=True if register_similarity else False`. Since the mean mesh is also built when `scale_jointly` is true (line 913), the two code paths hand `read_meshes_get_sampled_pts` different arguments for the same configuration.

**NSM/reconstruct/main.py:1372 — Regress.add_latent is handed the whole result dict, not a latent vector**

`reg.add_latent(result_)` passes the full reconstruct_mesh result dict. `Regress.add_latent` documents 'latent: list of floats' (predictive_validation_class.py:29-30) and appends it verbatim; `calc_r2` then does `np.array(self.list_latents)` (line 45) producing an object array of dicts, which `LinearRegression().fit` (line 66) cannot consume. Reachable from NSM/train/train_deep_sdf.py:250 whenever `predict_val_variables` is in the config. Also note the latent itself lives at `result_['latent']` as a torch tensor with grad, not a list of floats.

### `NSM/reconstruct/recon_evaluation.py`

**NSM/reconstruct/recon_evaluation.py:95 — compute_recon_loss mutates the caller's meshes to float32 in place**

`mesh.point_coords = mesh.point_coords.astype(np.float32)` and the same for `orig_meshes[mesh_idx]` on line 96, inside the `calc_assd` branch only. These are the same mesh objects reconstruct_mesh returns to the caller in `result['mesh']` and `result['orig_mesh']` (main.py:1149-1150) and that the consumer then writes to disk (kneepipeline/steps/run_nsm.py:216-221). So passing `calc_assd=True` -- which the consumer does by default -- silently downcasts the saved output meshes, and passing `calc_assd=False` does not. Nothing documents this.

### `NSM/reconstruct/reconstruct_latent_S3.py`

**NSM/reconstruct/reconstruct_latent_S3.py:127 — reconstruct_latent_S3 references an undefined name in its own error path and calls wandb without importing it**

Line 127: `raise ValueError(f"Inputted SDF must have shape Nx3 or Nx4 got: {new_s}")` -- `new_s` is undefined (the parameter is `new_sdf`), so the error path raises NameError. Line 316 calls `wandb.log(...)` but the module never imports wandb -> NameError whenever `log_wandb=True`. Line 320 reads `latent_loss_.item()` which is only bound when `l2reg is True` (line 246). Line 97 `adjust_lr_every = num_iterations // n_lr_updates` divides by zero if `n_lr_updates=0` (the main-path helper `reconstruct_latent_get_lr_update_freq` guards this; this copy does not). Line 313 prints `latent.norm` (the bound method).

### `NSM/train/train_deep_sdf.py`

**NSM/train/train_deep_sdf.py:152 — The param-group target key is duplicated as a bare string literal in the train loop**

`if any(group.get("target") is None for group in optimizer.param_groups)` hardcodes "target" instead of importing `NSM.utils.PARAM_GROUP_TARGET_KEY` (NSM/utils.py:23), which every other site uses (utils.py:228, 273, 379, 388). This is the last place the key is spelled out by hand; changing the constant would leave this checkpoint guard silently rejecting every checkpoint.

**NSM/train/train_deep_sdf.py:569 — step_mean_vec_length / step_std_vec_length are assigned, not accumulated — logged latent-norm metrics are wrong by a factor of len(data_loader)**

Lines 298-299 initialize `step_mean_vec_length = 0` / `step_std_vec_length = 0` as accumulators. Lines 569-570 use `=` rather than `+=`: `step_mean_vec_length = mean_vec_length.item()`. `mean_vec_length` is itself a leaked variable from the inner split loop (line 555), so it holds the last chunk of the last batch. Lines 590-591 then divide by `len(data_loader)`. Net effect: wandb's `mean_vec_length` and `std_vec_length` are (last chunk's value) / (number of batches) — an arbitrary number that shrinks as the dataset grows. Every other accumulator in the same function (step_losses:562, step_l1_loss:563, step_mean_size:575) correctly uses `+=`. The identical bug is present at deprecated/train_deep_sdf_multi_surface_orig.py:507-508.

### `NSM/train/train_deep_sdf_multi_head.py`

**NSM/train/train_deep_sdf_multi_head.py:83 — multi_head never moves latent vectors to the device and hardcodes .cuda()**

Line 83: `latent_vecs = get_latent_vecs(len(data_loader.dataset), config)` — no `.to(config["device"])`, unlike train_deep_sdf.py:112. The latent Embedding therefore stays on CPU while models go to `config["device"]` (line 60). Combined with line 256 (`sdf_gt[surf_idx][split_idx].cuda()`) and 298-299 (`.cuda()` again), the module ignores `config["device"]` for tensors and cannot run on mps or cpu at all, while simultaneously claiming device-configurability at lines 60/120-123.

**NSM/train/train_deep_sdf_multi_head.py:85 — train_deep_sdf_multi_head builds the optimizer from a leaked loop variable — only the last decoder is trained**

CONFIRMED. Lines 59-60: `for model in models:` / `    model = model.to(config["device"])`. The loop leaks `model` bound to `models[-1]`. Line 85-91: `optimizer = get_optimizer(model, latent_vecs, lr_schedules=..., ...)` passes that leaked single module, not `models`. `get_optimizer` (NSM/utils.py:373-374) wraps a non-list into `[model]`, so exactly one `model_0` param group is created, holding only the last decoder's parameters. Every other decoder in `models` stays at initialization forever while `train_epoch` still runs `model(inputs)` on all of them (line 241-245) and backprops through them (line 392). The same leaked variable is reused at line 403 (`torch.nn.utils.clip_grad_norm_(model.parameters(), ...)`), so grad clipping also only touches the last model. Documented in docs/KNOWN_ISSUES.md:226-250; module now warns (lines 27-33) but is not fixed.

**NSM/train/train_deep_sdf_multi_head.py:118 — Non-short-circuit `&` on membership tests raises KeyError instead of skipping**

Line 118: `if ("val_paths" in config) & (config["val_paths"] is not None):`. `&` is the bitwise operator and does NOT short-circuit, so `config["val_paths"]` is evaluated even when the key is absent — the guard raises the exact KeyError it was written to prevent. Same construct at lines 329-331 for `surface_weighting`. train_deep_sdf.py fixed this by using `and` (line 179-181) and `config.get(...)` (line 477), so this is a divergence, not a shared idiom.

**NSM/train/train_deep_sdf_multi_head.py:123 — torch.mps.empty_cache() is called on the CPU branch**

Lines 120-123: `if config["device"] == "cuda": torch.cuda.empty_cache()` / `elif config["device"] == "cpu": torch.mps.empty_cache()`. The mps cache clear is reached only when the device is CPU. NSM/utils.py:439-443 has the correct version (`clear_gpu_cache`), which train_deep_sdf.py uses at lines 208 and 267; multi_head never adopted it. Also `== "cuda"` fails to match the shipped default device string `"cuda:0"` (NSM/configs/default_config.json), so the CUDA branch is dead for the default config and the CPU/mps branch is what gets tested against.

**NSM/train/train_deep_sdf_multi_head.py:359 — multi_head accumulates per-surface L1 with .append() into a fixed-size list, then discards it**

Line 227 initializes `batch_l1_losses = [0.0 for _ in range(n_surfaces)]`. Line 358-359 then does `batch_l1_losses.append(l1_loss_.sum().item())` instead of `batch_l1_losses[l1_idx] += ...`, so the list grows by n_surfaces entries per split and the leading zeros are never updated. It is then never read. Lines 399-400 instead accumulate `step_l1_losses` from `l1_losses` — the loop variable leaked out of the `split_idx` loop — so the reported per-surface losses come from the last split only, not the batch. train_deep_sdf.py has the correct form at lines 500-501 and 566-567.

**NSM/train/train_deep_sdf_multi_head.py:382 — multi_head hardcodes a 100-epoch code-regularization warm-up instead of reading the config key**

Line 382: `config["code_regularization_weight"] * min(1, epoch / 100) * l2_size_loss`. train_deep_sdf.py:544 uses `min(1, epoch / config["code_regularization_warmup"])`. Setting `code_regularization_warmup` in a config has no effect through the multi_head entry point — the knob silently does nothing. Both deprecated files use the config key (orig:277, multi_surface_orig:483), so multi_head is the odd one out.

### `NSM/train/utils.py`

**NSM/train/utils.py:90 — add_plain_lr_to_config raises KeyError on a Constant learning-rate schedule**

Line 90 unconditionally reads `schedule_["Initial"]`. But `get_learning_rate_schedules` builds a ConstantLearningRateSchedule from `schedule_spec["Value"]` (NSM/utils.py:183) — a Constant entry has no `Initial` key. Since `add_plain_lr_to_config` is the FIRST thing every train entry point calls (train_deep_sdf.py:80, multi_head:54, both deprecated files), configuring either schedule as `{"Type": "Constant", "Value": ...}` crashes training before the model is even moved to the device — with a bare KeyError('Initial') from a logging helper. Line 89 (`schedule_["Type"]`) is safe; only line 90 is fatal. The docstring (64-70) documents neither this nor any Raises.

### `NSM/utils.py`

**NSM/utils.py:26 — LearningRateSchedule base class returns None instead of raising**

`class LearningRateSchedule: def get_learning_rate(self, epoch): pass` (lines 26-28). A subclass that forgets to override yields `param_group['lr'] = None` at utils.py:237, which torch will not reject until the optimizer step produces a TypeError far from the cause. `raise NotImplementedError` costs nothing here.

**NSM/utils.py:394 — get_optimizer silently drops weight_decay for the default 'Adam' optimizer**

`weight_decay=0.0001` is a parameter of `get_optimizer` (line 362) and `weight_decay` is a shipped config key (default_config.json:82), but line 395 constructs `torch.optim.Adam(list_params)` without it. Only the AdamW (line 397) and schedule_free_AdamW (line 401) branches pass it. With the shipped default `optimizer: "Adam"`, the configured weight decay is silently inert. The docstring (lines 363-372) does not mention this.

**NSM/utils.py:410 — symmetric_chammfer is an empty stub with an empty docstring**

`def symmetric_chammfer(p1, p2, n_pts): """ """; pass` — returns None, has a whitespace-only docstring (so it passes a naive has-docstring check), is misspelled ('chammfer'), and has zero callers anywhere in NSM/ or testing/. Anyone who calls it gets None with no error.

---

## Rot (54)

False documentation, dead parameters, stale comments.

### `NSM/configs/deep_sdf_config`

**NSM/configs/deep_sdf_config:25 — A scratch notes file ships inside the package and preserves the obsolete two-positional-entry LR shape**

`NSM/configs/deep_sdf_config` is a 404-byte extensionless ASCII file — neither valid JSON nor valid Python, just an outline of config keys — untouched since the initial commit 5188417 'Updating to NSM (neural shape model)'. Lines 25-29 sketch `'LearningRateSchedule': [{}, {}]`, i.e. the anonymous two-positional-entry shape that the Aug 2026 `Target` work exists to eliminate. It sits next to the real default config where a new reader will find it.

### `NSM/datasets/sdf_dataset.py`

**NSM/datasets/sdf_dataset.py:2 — Seven unused imports at module top**

pymskt as mskt (2), vtk (5), numpy_to_vtk and vtk_to_numpy (6), warnings (10), point_cloud_utils as pcu (12) and pympler's muppy (19) are never referenced in executable code -- 'vtk'/'mskt' appear only inside docstrings and a filename literal (1383), 'pcu' only as the string argument method='pcu' (312, 688, 722), 'muppy' only in commented-out code (964, 971, 973). Because __init__.py does `import *` with no __all__, all of them are also re-exported as NSM.datasets.vtk, NSM.datasets.pcu, etc.

**NSM/datasets/sdf_dataset.py:169 — `mean` parameter is documented and accepted by both sampling functions but never used**

read_mesh_get_sampled_pts(mean=[0,0,0]) at 169 documented at 191; read_meshes_get_sampled_pts(mean=[0,0,0]) at 407 documented at 432. The identifier `mean` appears nowhere in either body. Three call sites dutifully pass mean=[0,0,0] (1219, 1824, NSM/reconstruct/main.py:992), which reads as meaningful configuration and is not.

**NSM/datasets/sdf_dataset.py:457 — False comment: 'vtkAppendPolyData' is claimed in three places and used nowhere**

read_meshes_get_sampled_pts Notes (457) and the MultiSurfaceSDFSamples docstring (1589) both state that surfaces are combined 'using VTK's vtkAppendPolyData'. docs/MULTI_SURFACE_REGISTRATION.md repeats the multi-surface story. The string vtkAppendPolyData does not occur in the file; the actual mechanism is combine_meshes -> Mesh.__add__ -> pyvista merge (2193). A reader debugging combine ordering or point-data merging will look for the wrong API.

**NSM/datasets/sdf_dataset.py:1131 — save_data_to_cache serializes three keys that nothing ever produces**

additional_keys at 1131-1138 includes 'center', 'max_radius', 'max_radius_xyz'. No code path anywhere in the file writes data['center'], data['max_radius'] or data['max_radius_xyz'] into a sample dict (self.center / self.max_radius are dataset attributes, set at 1049-1050, never per-sample). The `if key in data` guard at 1139 means these three are permanently dead. A commented-out alternative implementation sits at 1142.

**NSM/datasets/sdf_dataset.py:1146 — Cache key names are renamed on write and triple-guessed on read**

np.savez writes coordinates as 'pts' and SDFs as 'sdfs' (1146), while everything in memory calls them 'xyz' and 'gt_sdf'. unpack_numpy_data then accepts 'pts' or 'xyz' (376-379) and 'sdfs' or 'gt_sdf' or 'sdf' (384-391). The three-way fallback is archaeology of at least two past cache formats, undocumented, with no version field to tell them apart.

**NSM/datasets/sdf_dataset.py:1366 — Stale TODOs that name work the refactor plan should absorb**

Line 1366: 'TODO: Why is reference_object different from mesh_to_scale?' -- the two indices genuinely select different things (reference_object drives centering in norm_and_scale_all_meshes at 1024/1061; mesh_to_scale drives ICP and per-sample scaling at 530/588) and nothing documents the split. Line 1792: 'TODO: crat' -- truncated to meaninglessness. Lines 2039-2043: a TODO asserting that storing pts and sdfs separately is 'something that we are constantly undoing/re-doing elsewhere in the code'. Lines 211-218: a Notes block that is really a TODO about read_mesh_get_sampled_pts being over 100 lines. NSM/datasets/utils.py:1-2 is itself a TODO proposing the function/class split. Five separate notes all describing the same decomposition.

### `NSM/dependencies/sinkhorn.py`

**NSM/dependencies/sinkhorn.py:12 — sinkhorn's `p` is annotated float but rejected unless it is an int**

Signature declares `p: float = 2` (line 12) while line 41 raises TypeError for any non-int and the docstring (line 28) says 'Must be an integer greater than 0'. Calling `sinkhorn(x, y, p=1.5)` — which the annotation invites — raises TypeError.

**NSM/dependencies/sinkhorn.py:31 — sinkhorn docstring calls eps the 'reciprocal' of the regularization parameter**

The code uses eps as the divisor in the Gibbs kernel (`(-M_ij + v_j) / eps`, lines 140-146, 155), i.e. larger eps = more entropic smoothing, so eps IS the regularization parameter, not its reciprocal. A reader tuning eps from the docstring will move it the wrong way.

### `NSM/losses.py`

**NSM/losses.py:1 — losses.py is the one file in the subsystem that fails the repo's own Black check**

`black --check --line-length 100` over the seven subsystem files reports 'would reformat NSM/losses.py'; the other five .py files pass. CLAUDE.md mandates Black at 100 chars and Makefile exposes `make format-check`, so `make format-check` fails on a clean checkout of main.

**NSM/losses.py:82 — Three of losses.py's five public functions have never been called by anything**

`compute_sdf_gradients` (line 82) and `combined_sdf_loss` (line 156) have zero call sites in NSM/ and testing/ (grep for the names returns only their definitions). `l1_loss` (line 224) and `l2_loss` (line 229) are labelled 'Legacy function aliases for backward compatibility' (line 223), but `git log --follow NSM/losses.py` shows exactly one commit (468d687 'Add eikonal loss') — there is no earlier version for them to be backward-compatible with, and nothing imports them. Only `eikonal_loss` is imported (NSM/train/train_deep_sdf.py:12, NSM/reconstruct/main.py:13).

### `NSM/mesh/__init__.py`

**NSM/mesh/__init__.py:1 — Package __init__ star-exports main.py's third-party imports and hides four modules**

`from .main import *` with no `__all__` in main.py, so `NSM.mesh.os`, `.torch`, `.np`, `.pv`, `.mskt`, `.vtk`, `.inspect` and `.marching_cubes` are all part of the public surface (main.py:16-23). Meanwhile interpolate, correspondence_metrics, refine_mesh and triangle_metrics are NOT re-exported and must be reached by full dotted path (as NSM/reconstruct/main.py:12 does for create_mesh_adaptive and testing/NSM/mesh/* do for the rest). Inconsistent package surface.

### `NSM/mesh/correspondence_metrics.py`

**NSM/mesh/correspondence_metrics.py:1 — Plan and lint config point at an experiments/ tree that does not exist on main**

.flake8:26 excludes `experiments`, and .claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md:316, :496-501 references `experiments/mesh_interpolation/config.py`, `subjects.py`, `fit_cache.py`, `compare_mesh_path.py` — no `experiments/` directory exists in the repo. Those scripts were correspondence_metrics.py's only non-test consumers. The same plan claims 'Interpolate tests (31 tests)' (:496) while testing/NSM/mesh/test_interpolate.py now defines 9 (trimmed by commit fa862aa, 'Trim mesh interpolation to production config'). Correspondence-metrics' claimed 39 tests is accurate.

**NSM/mesh/correspondence_metrics.py:30 — Unused import, masked by a project-wide flake8 ignore**

`get_edge_lengths` is imported and never used. flake8 --isolated reports F401 here, but .flake8:20 sets `extend-ignore = ..., F401` ('unused imports (several scratch / timing scripts)'), so `make lint` is silent on it repo-wide.

**NSM/mesh/correspondence_metrics.py:224 — Two divergent implementations of the edge-ratio statistic**

triangle_health recomputes min/max edge and the ratio by hand (:224-237) rather than calling `TriangleProperties.edge_ratio()` (triangle_metrics.py:72-85), which it already has an instance of (:215). The reason is a behaviour difference: edge_ratio RAISES on a zero-length edge (triangle_metrics.py:79-81) while triangle_health deliberately degrades (degenerate mask + nan handling). Two policies for degenerate triangles now coexist in one subsystem.

### `NSM/mesh/interpolate.py`

**NSM/mesh/interpolate.py:291 — build_mesh_laplacian does not return a Laplacian**

The name and the docstring's first line say 'graph Laplacian'; the function returns the row-normalised ADJACENCY matrix (:318-321, and the Returns line at :305 says so). The actual Laplacian displacement is formed at the single use site as `torch.sparse.mm(laplacian, points) - points` (:404). A reader who takes the name at face value and applies the returned matrix as a Laplacian gets neighbour-averaged positions instead of a displacement, with no error.

### `NSM/mesh/main.py`

**NSM/mesh/main.py:169 — Dead local and formatting drift against the project's own stated standard**

`new_pts = new_mesh.point_coords` assigned and never used (flake8 F841). Separately, CLAUDE.md mandates Black at 100 chars but `black --check --line-length 100 NSM/mesh/` reformats main.py and interpolate.py, and flake8 --isolated reports E501 at main.py:74, 95, 318, 470, 822, 836 and refine_mesh.py:89, 126, 151, 273, 447, 448, 449, 461.

**NSM/mesh/main.py:185 — Five public functions in main.py have no docstring at all**

`create_mesh` (:185, 17 params), `scale_mesh` (:151), `scale_mesh_` (:126), `sdf_grid_to_mesh` (:271), `create_grid_samples` (:779). Their direct siblings in the same file (`create_mesh_adaptive`:551, `sdf_grid_to_mesh_vtk`:373, `create_grid_samples_in_bounds`:497) each carry a full Args/Returns block, so the file reads as documented until you hit one of the five.

**NSM/mesh/main.py:323 — band_width documented as world units, used as a voxel multiplier**

Docstring: 'band_width: Width of narrow band in world units (multiplier of voxel_size)' (:323, repeated at :391 for sdf_grid_to_mesh_vtk). The two halves contradict each other; the code (:339) is `band = band_width * voxel_size`, i.e. a pure voxel multiplier.

**NSM/mesh/main.py:341 — crop_sdf_to_narrow_band names every index variable for the wrong axis**

`z, y, x = np.where(mask)` (:341) — but the module docstring (:6-10) declares these arrays are (X, Y, Z), so `z` indexes X. The shape unpack is `orig_nx, orig_ny, orig_nz = sdf_values.shape` (:333) yet the clamp is `xe = min(x.max() + pad_voxels + 1, orig_nz)` (:350), pairing `x` with `orig_nz`. The code is FUNCTIONALLY CORRECT (crop_origin at :360-364 maps axis0->origin[0]), but every name is inverted relative to the convention the module docstring establishes 330 lines above.

**NSM/mesh/main.py:603 — create_mesh_adaptive docstring understates what n_pts_per_axis controls**

'n_pts_per_axis: Dense grid resolution (for fallback only)'. It is also what sets the FINE voxel size on the main adaptive path when voxel_size is None: `voxel_size = original_extent / (n_pts_per_axis - 1)` (:638-639), and NSM/reconstruct/main.py:919-923 relies on exactly that (it passes n_pts_per_axis_mean_mesh and no voxel_size).

### `NSM/mesh/refine_mesh.py`

**NSM/mesh/refine_mesh.py:142 — Stale 'Implement this' comment on already-implemented code**

`new_faces = create_new_faces(faces[face_idx], midpoint_indices)  # Implement this` — create_new_faces is fully implemented at :264. A second instance at :355-357 (`# Implement mesh update logic` on a line calling the implemented `update_mesh`:287).

**NSM/mesh/refine_mesh.py:438 — Plan claims a symbol that exists nowhere, leaving refine_mesh.py orphaned**

.claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md:111 says the rejected Fix 8 'hand-built code is still available as `interpolate_points_refined`', and :495 tabulates 'Hand-built subdivision (used by Fix 8) | NSM/mesh/refine_mesh.py::subdivide_triangles_on_base_mesh'. `interpolate_points_refined` is grep-absent from the entire repo. refine_mesh.py is therefore the remnant of a wrapper that was removed — the only stated reason for keeping it no longer exists on main.

### `NSM/mesh/triangle_metrics.py`

**NSM/mesh/triangle_metrics.py:1 — triangle_metrics.py has zero docstrings on every public symbol**

No module docstring; `get_triangle_area` (:5), `calculate_triangle_areas` (:19), `length` (:28), `get_edge_lengths` (:32), class `TriangleProperties` (:45) and all five of its public methods (`areas`:51, `compute_edge_lengths`:63, `edge_ratio`:72, `edge_sd`:87, `edge_length_max`:93) are undocumented. `length` is a maximally generic module-level public name. This file is a transitive dependency of correspondence_metrics.py, which is otherwise the best-documented module in the subsystem.

### `NSM/models/deep_sdf.py`

**NSM/models/deep_sdf.py:27 — Sine.__init__ is misspelled and never runs**

`def __init(self)` — missing the trailing double underscore. The method is unreachable; nn.Module.__init__ runs instead, which happens to be exactly what the body would have done, so the typo is harmless today and invisible to tests.

**NSM/models/deep_sdf.py:87 — latent_noise_sigma is stored and never read**

Assigned to self at line 87 from the constructor arg at line 53, then never referenced anywhere in the file or repo. loader.py:165 forwards it from config. Same dead-option shape as xyz_in_all.

### `NSM/models/loader.py`

**NSM/models/loader.py:147 — Two contradictory deprecation messages for latent_dropout, and the shipped config triggers one**

loader.py:145-148 warns "latent_dropout is deprecated in config. Use dropout_prob instead"; deep_sdf.py:71-72 warns "latent_dropout is deprecated. Use dropout instead". `dropout` and `dropout_prob` are different parameters (indices vs probability), so one of the two messages sends the user to the wrong option. NSM/configs/default_config.json contains a `latent_dropout` key, so every deepsdf load from the shipped default emits the loader warning.

### `NSM/models/modulated_periodic_activations.py`

**NSM/models/modulated_periodic_activations.py:43 — Two different Sine classes in one package, with incompatible defaults**

deep_sdf.py:26 defines Sine with w0 hardcoded to 30 inside forward; this file defines a Sine with w0 as a constructor argument defaulting to 1.0. Because __init__.py:1 wildcard-imports deep_sdf first and line 2 imports only three explicit names, `NSM.models.Sine` resolves to the hardcoded-30 variant (verified). A reader who imports Sine from the package and passes w0 gets a TypeError.

**NSM/models/modulated_periodic_activations.py:244 — Debug print left on ImplicitDecoder's forward path**

`print(xyz.shape)` executes on every forward whenever modulation is disabled (the default). It is exercised by the passing test suite (testing/NSM/models/test_loader.py:269).

### `NSM/models/triplanar.py`

**NSM/models/triplanar.py:5 — Unused imports**

triplanar.py imports `time` (line 5) and `logging` (line 6); neither is referenced anywhere in the file. modulated_periodic_activations.py imports `pi` (line 5) from math; only `sqrt` is used (line 76). Leftovers from the removed caching work (commit fee37be).

**NSM/models/triplanar.py:9 — triplanar.py's apparent module docstring is a no-op string literal**

The 14-line architecture description at lines 9-22 sits after the import block, so it is an expression statement, not a docstring. Verified: `NSM.models.triplanar.__doc__` is None. It is invisible to help(), pydoc, and any doc tooling.

**NSM/models/triplanar.py:219 — Assertion message states the opposite of the branch it guards**

Inside `if self.sum_sdf_features is False:` the assertion message reads "sdf_latent_size must be divisible by 3 if sum_sdf_features is True". Anyone who hits this assertion is told to change the flag to the value it already is not.

**NSM/models/triplanar.py:312 — normalize_coordinates ignores its own padding parameter**

The signature declares `padding=0.1` (line 312) but the body uses `self.padding` (line 322); the sole call site passes no padding at all (line 296). A caller who passes padding= gets silently ignored. The same line uses the literal `10e-6` (= 1e-5), which reads like an intended 1e-6.

### `NSM/reconstruct/cartilage_func.py`

**NSM/reconstruct/cartilage_func.py:141 — Dead locals left behind by a commented-out KL-divergence metric**

`orig_array = orig_bone.get_scalar('thickness (mm)')` and `recon_array = ...` (lines 141-142) are computed and never used; the only consumer was the `thickness_kld` block commented out at lines 144-147. `from scipy.stats import entropy` (line 3) is now an unused import kept alive only by that dead code. Similarly `CART_REGIONS` (lines 5-14) still carries three commented-out tibial/patellar entries.

### `NSM/reconstruct/main.py`

**NSM/reconstruct/main.py:420 — `latent_input` is computed and never used**

`latent_input = latent.expand(n_samples, -1)` -- dead since the decoder call moved to the `latent=/xyz=` kwargs form at line 588. It is a leftover from the concatenated-input interface still used by reconstruct_latent_S3.py:235-236, and it makes a reader believe `n_samples` fixes the latent broadcast width when it does not.

**NSM/reconstruct/main.py:750 — Latent-norm progress print emits the bound method instead of the value**

`print("\tLatent norm: ", latent.norm)` prints `<built-in method norm of Tensor object at ...>`. Lines 728, 759, and 494 all correctly use `latent.norm().item()`. Duplicated in NSM/reconstruct/reconstruct_latent_S3.py:313.

**NSM/reconstruct/main.py:826 — `mesh_to_scale` inline comment is stale since multi-surface registration landed**

`mesh_to_scale=0,  # PRETTY MUCH ASSUME ALWAYS SCALING FIRST MESH` and line 827 `decoder_to_scale=0,  # PRETTY MUCH ASSUME ALWAYS SCALING FIRST DECODER`. Lines 934-941 now explicitly accept a list/tuple for `mesh_to_scale` and combine the mean meshes via `combine_meshes` -- the documented behaviour is in docs/MULTI_SURFACE_REGISTRATION.md:61. The comment tells a new reader the opposite of what the code supports.

**NSM/reconstruct/main.py:1167 — time_calc_recon_loss is measured and thrown away while return_timing claims to report timings**

`time_calc_recon_loss = toc - tic` is computed inside the metrics block but never added to `result`; the `return_timing` block at lines 1179-1184 reports the other five timings only. Since ASSD/chamfer over full point clouds is usually the most expensive part of a validation pass, the one timing a tuner would want is the one dropped.

**NSM/reconstruct/main.py:1202 — tune_reconstruction is uncalled and passes a parameter get_mean_errors no longer honours**

Zero callers anywhere in the repo, testing/, or the consumer checkout. It reads 24 required keys off `config` with no `.get` defaults, hard-requires `os.environ['WANDB_KEY']` (line 1207), passes `batch_size_latent_recon` (line 1233) which now only produces a deprecation print, and discards the return value of `get_mean_errors`. `compute_correlation_coefficient` (line 1437) is likewise uncalled.

**NSM/reconstruct/main.py:1299 — get_mean_errors sets register_similarity twice and its error message contains a typo**

`register_similarity` is written into `reconstruct_inputs` at line 1299 and again into `reconstruct_inputs_` at line 1324, which then overwrites it at line 1347. Harmless but confusing. Line 1343-1345 raises `f'model_type must be either "deepsdf" or "diffusion"m received {model_type}'` -- stray 'm', and 'diffusion' is not actually accepted by any branch.

### `NSM/reconstruct/recon_evaluation.py`

**NSM/reconstruct/recon_evaluation.py:34 — compute_recon_loss docstring documents a parameter that no longer exists and omits three that do**

The docstring documents `orig_pts (list): A list of pts from ground truth meshes.` The actual parameter is `orig_meshes` (line 21); `orig_pts` was commented out at line 20 and remains commented at the call site (main.py:1158). Also undocumented: `orig_meshes`, `n_samples_assd` (line 24 -- and it is dead: the only ASSD call at line 97 delegates to `mesh.get_assd_mesh` and never uses it), and `calc_assd` (line 26). The assertion message at line 55 also still says 'number of original points'.

### `NSM/reconstruct/utils.py`

**NSM/reconstruct/utils.py:42 — get_pt_cloud_distances docstring has d1 and d2 swapped**

Docstring lines 42-43 claim 'd1: distances from each point in pts1 to its nearest neighbor in pts2' and the mirror for d2. The code at lines 49-53 builds `kd1` on pts1 and `kd2` on pts2, then does `d1, _ = kd1.query(pts2)` and `d2, _ = kd2.query(pts1)` -- so d1 has len(pts2) entries measured against pts1, exactly the reverse of the docstring. compute_assd (line 76) divides by `pts1.shape[0] + pts2.shape[0]`, which happens to be correct only because len(d1)+len(d2) equals that sum; anyone who reads the docstring and 'fixes' the denominator will introduce a bug.

**NSM/reconstruct/utils.py:58 — compute_assd is defined but its only import is commented out**

`from .utils import compute_chamfer  # , compute_assd` (recon_evaluation.py:12). ASSD is instead computed by delegating to `mesh.get_assd_mesh(...)` (recon_evaluation.py:97), a pymskt method with different semantics from this numpy implementation. Two ASSD implementations now exist and the one in this repo is unreachable; the `n_samples_assd` parameter that used to feed it is still in compute_recon_loss's signature (line 24) doing nothing.

### `NSM/train/deprecated/train_deep_sdf_multi_surface_orig.py`

**NSM/train/deprecated/train_deep_sdf_multi_surface_orig.py:47 — deprecated/train_deep_sdf_multi_surface_orig.py is an 85%-identical stale fork of the live training loop**

difflib SequenceMatcher ratio against NSM/train/train_deep_sdf.py is 0.853 (562 vs 629 lines). It predates every recent fix: it loads the same checkpoint file twice (lines 93-108, fixed at train_deep_sdf.py:124-131), has no optimizer-state or param-group-target migration guards (train_deep_sdf.py:138-156), hardcodes `objects_per_decoder=2` in the validation call (line 194, now `config["objects_per_decoder"]` at train_deep_sdf.py:236), hardcodes `torch.cuda.empty_cache()` (lines 166, 224) and `.cuda()` throughout instead of honouring `config["device"]`, and lacks the mesh_names validation entirely. It carries the same `step_mean_vec_length =` accumulator bug (lines 507-508). Any future edit to the live loop has an 85%-similar decoy sitting next to it.

### `NSM/train/deprecated/train_deep_sdf_orig.py`

**NSM/train/deprecated/train_deep_sdf_orig.py:125 — deprecated/train_deep_sdf_orig.py returns an undefined name**

`return loss` — `loss` is never bound anywhere in `train_deep_sdf` (the epoch loop assigns `log_dict` at line 68). Any successful run of this function ends in `NameError: name 'loss' is not defined`. Independent confirmation that this file is unreachable: it has zero importers and zero callers in the repo, and the current train_deep_sdf.py:269 ends with a bare `return`.

### `NSM/train/train_deep_sdf.py`

**NSM/train/train_deep_sdf.py:84 — Dead duplicate of the resume_epoch default**

Lines 84-85 (`if "resume_epoch" not in config: config["resume_epoch"] = 0`) can never fire: line 58 already ran `config.setdefault("resume_epoch", 0)`. Leftover from the merge of train_deep_sdf_multi_surface into train_deep_sdf that the comment at lines 54-55 describes.

**NSM/train/train_deep_sdf.py:279 — train_epoch accepts return_loss and verbose parameters that are never read**

`return_loss=True` (line 279) and `verbose=False` (line 280) appear in the signature but neither name occurs in the body; verbosity is taken from `config["verbose"]` (lines 310, 339, 355, ...) and the function always returns `log_dict`. Callers pass `return_loss=True` anyway (line 174), which reads as if it were load-bearing. Identical dead parameters in multi_head:166-167 and both deprecated files.

### `NSM/train/train_deep_sdf_multi_head.py`

**NSM/train/train_deep_sdf_multi_head.py:27 — CLAUDE.md still advertises train_deep_sdf_multi_head as a supported training pipeline**

The repo's CLAUDE.md line 120 reads '`train_deep_sdf_multi_head.py`: Multi-head training for multiple surfaces' with no qualification, under a heading describing NSM/train/ as 'Training pipelines'. The module itself raises a DeprecationWarning calling itself 'DEPRECATED and known to be broken' (lines 27-33), and docs/KNOWN_ISSUES.md:226-250 documents that all runs through it are affected. A reader following CLAUDE.md picks the broken entry point.

### `NSM/train/utils.py`

**NSM/train/utils.py:4 — Unused imports and a duplicated import in train/utils.py**

`import torch` appears twice (lines 2 and 4). Line 5 imports `profile` from torch.profiler but it is never used — line 117 calls the fully-qualified `torch.profiler.profile` instead. Only `tensorboard_trace_handler` from that import is used (line 119).

**NSM/train/utils.py:41 — cyclic_anneal_linear computes an unused `cycle` local**

Line 41 `cycle = epoch // cycle_length` is assigned and never read; only `cycle_progress` (line 42) is used. The docstring (37-39) is a bare URL and documents none of the six parameters, the return type (a numpy float64, not a Python float, because of `np.min` on line 45), or the behaviour when `n_epochs` is not divisible by `n_cycles` (the final partial cycle uses a `cycle_length` that no longer divides the remaining epochs).

**NSM/train/utils.py:51 — get_kld's docstring describes a different computation than the code performs**

The docstring (lines 51-55) shows `kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)` — the per-sample KLD of a network-predicted diagonal Gaussian, reduced over dim=1, returning one value per sample. The body (56-58) instead computes the EMPIRICAL mean and variance of the batch along `samples_dim` and returns a single scalar summed over all remaining dims. These are different estimators: the docstring's uses per-sample predicted parameters; the code's measures how far the batch's aggregate distribution is from N(0,1). The `samples_dim` argument is undocumented, and `torch.var` applies Bessel's correction by default (undocumented, and it makes the value depend on batch size). Used as a code-regularization prior at train_deep_sdf.py:535 and multi_head:374.

### `NSM/utils.py`

**NSM/utils.py:19 — CLASSIFICATION_HEADS_GROUP_NAME is documented as a real param group but nothing ever creates one**

`CLASSIFICATION_HEADS_GROUP_NAME = "classification_heads"` has zero references anywhere in NSM/ or testing/ (grep for the identifier returns only its definition). The docs around it assert the group exists: utils.py:77-78 says LR_TARGET_MODEL drives 'when present, ``classification_heads``', and utils.py:213-214 says 'every decoder and the classification heads all take the model schedule'. `get_optimizer` (utils.py:376-392) builds only a `latent` group and `model_{idx}` groups. The only classification_heads group in the repo is hand-built inside testing/NSM/test_lr_schedules.py:150. A reader will look for the code that adds this group and find none.

### `testing/NSM/test_lr_schedules.py`

**testing/NSM/test_lr_schedules.py:569 — Stale comment claims the config generator still writes on import**

The comment reads 'NB: importing this module writes ./default_config.json as a side effect, so run the import from a tmp cwd rather than littering the repo root', and the test still monkeypatches chdir for that reason. Commit d1fd05f moved the write behind the `if __name__ == "__main__"` guard (NSM/configs/generate_sdf_default_config.py:106-112), and testing/NSM/configs/test_default_config_sync.py:43-56 now asserts the import writes nothing. The comment is false and directly contradicts a sibling test.

### `testing/testing_h5_vs_np_loading/save_and_load_h5_vs_np.py`

**testing/testing_h5_vs_np_loading/save_and_load_h5_vs_np.py:74 — unpack_pts/unpack_numpy_data are duplicated verbatim in a testing script**

testing/testing_h5_vs_np_loading/save_and_load_h5_vs_np.py:74 and :84 define their own unpack_pts and unpack_numpy_data rather than importing NSM/datasets/sdf_dataset.py:335 and :367. A change to the cache key fallbacks has to be made twice, and this copy is where the h5-vs-npy caching decision is being benchmarked -- exactly the seam the plan will want to touch.

---

## Notes (33)

Observations. Not bugs.

### `NSM/__init__.py`

**NSM/__init__.py:1 — NSM/__init__.py leaks `os` into the public namespace for the sake of commented-out code**

`import os` at line 1 exists only to support lines 3-6, which are four commented-out `os.environ[...] = "1"` thread-limit settings. Verified: `sorted(n for n in vars(NSM) if not n.startswith('__'))` == ['os', 'utils']. So `NSM.os` is part of the de-facto API surface, and the entire deliberate surface is `utils` plus `__version__`. There is no `__all__` anywhere under NSM/ (grep returns nothing), which is why the four star-importing subpackage __init__ files re-export their modules' incidental imports too.

### `NSM/_lr_migration.py`

**NSM/_lr_migration.py:7 — _lr_migration.py states its own delete-when condition and it is not yet met**

Quoted verbatim from the module docstring: 'DELETE THIS FILE once no config still in use predates the ``Target`` key. The only caller is ``resolve_schedule_targets`` in ``NSM/utils.py``, which imports it lazily and needs a plain one-line ValueError in its place.' docs/KNOWN_ISSUES.md:189-190 repeats it. The lazy import sits at NSM/utils.py:116 with a comment (lines 113-115) explaining that placement is what keeps the removal a one-liner. Nothing tracks when the condition is satisfied, so the deletion depends on someone remembering to check.

### `NSM/configs/default_config.json`

**NSM/configs/default_config.json:1 — Nothing in the library ever loads default_config.json**

Grepping NSM/ for 'default_config' finds only NSM/configs/generate_sdf_default_config.py's own path constant and write path. No training, reconstruction, or loader code reads the shipped JSON; its only readers are testing/NSM/test_lr_schedules.py:547 and testing/NSM/configs/test_default_config_sync.py:23. It is a copy-paste template that the Aug 2026 sync test now pins, not a runtime default.

### `NSM/datasets/sdf_dataset.py`

**NSM/datasets/sdf_dataset.py:316 — pts_surface return type differs between the single- and multi-mesh functions**

read_mesh_get_sampled_pts sets results['pts_surface'] to a plain Python list `[0] * n` (316, 320); read_meshes_get_sampled_pts sets it to an np.ndarray (678, 738). Callers that np.concatenate, index, or .shape the value behave differently depending on which function produced the dict, and neither docstring states the type.

**NSM/datasets/sdf_dataset.py:685 — Unconditional debug prints on the SDF hot path**

Lines 685-687 print rand_pts.shape, dtypes and type(rand_pts) for every mesh of every sample, outside any verbose guard, alongside per-stage timing prints at 500, 562, 628, 690, 747 and 'Fixed mesh...' at 126. There is no logger in the module; multiprocessing workers interleave all of it on stdout. Note the consumer parses subprocess stdout for its result JSON (kneepipeline CLAUDE.md: 'stdout: progress lines followed by a JSON result as the last line'), so library-level unconditional printing is an integration hazard, not just noise.

**NSM/datasets/sdf_dataset.py:832 — `multiprocessing` is simultaneously a module, a constructor parameter, and an instance attribute**

The stdlib module is imported at 14; the constructor takes multiprocessing=True at 832 (shadowing the module inside __init__, where line 939's `Pool` comes from the separate line-13 import); self.multiprocessing is the bool at 881; and load_mesh_step at 989 calls multiprocessing.cpu_count() on the module in a different scope. Reading line 988-989 (`if self.multiprocessing is True: os.sched_setaffinity(0, range(multiprocessing.cpu_count()))`) requires knowing which of the three is meant on each line.

**NSM/datasets/sdf_dataset.py:1021 — norm_and_scale_all_meshes reads every cache file twice from disk**

The disk branch loops over self.data np.load-ing each .npz to accumulate centers (1021-1024), then loops over all of them a second time np.load-ing each again to accumulate max radii (1032-1042). The second pass needs the global center, but the per-file new_pts could be retained from the first pass. On a large training set this is a full extra pass over the cache at startup.

**NSM/datasets/sdf_dataset.py:1310 — find_hash returns the first match anywhere under loc_save, across all date folders**

find_hash os.walks the entire loc_save tree and returns on the first filename match (1321-1327), so a cache written on any previous date is reused, while writes always go to today's folder (914). Combined with the hash gaps above (mesh_to_scale, uniform_pts_buffer, subsample), the reuse window is wide and silent. os.walk order is also filesystem-dependent, so which duplicate wins is not deterministic.

**NSM/datasets/sdf_dataset.py:1726 — get_sample_data_dict writes an unconditional append-only log into the cache root**

Every multi-surface sample load opens os.path.join(self.loc_save, 'list_meshes_started_loading.log') in append mode and writes the mesh paths (1726-1727). It is unconditional (not behind self.verbose), never truncated, written concurrently from multiprocessing workers, and targets self.loc_save rather than self.cache_folder -- so it raises FileNotFoundError in the save_cache=False + store_data_in_memory=True configuration, where nothing has created that directory (the makedirs at 913-915 is inside `if save_cache is True`).

### `NSM/mesh/correspondence_metrics.py`

**NSM/mesh/correspondence_metrics.py:291 — faces.reshape(-1, 4) assumes an all-triangle mesh in three places**

`mesh.faces.reshape(-1, 4)[:, 1:]` at correspondence_metrics.py:291 (self_intersection_count) and :484 (foldover_count), and refine_mesh.py:31 (get_faces). Verified: a pure-quad PolyData raises ValueError, but a MIXED mesh whose flat face-array length happens to be divisible by 4 reshapes without error into wrong connectivity. The module docstring says 'Meshes are accepted as pyvista.PolyData (triangular)' (:6) — nothing validates it. Related to the audit question: `polydata._faces` (the private attribute in open issue #6) appears NOWHERE in the repo; only the public `.faces` is used, and pyvista 0.46.5 emits no deprecation warning for it.

### `NSM/mesh/main.py`

**NSM/mesh/main.py:76 — coarse_bounds_from_sign_change returns None for two different reasons, one undocumented**

`if min(Z, Y, X) < 2: return None` (:76-77) — a degenerate grid — is indistinguishable from `if idx.size == 0: return None` (:106-107) — no surface. The docstring documents only 'or None if no surface found' (:72), and the caller (:684-697) responds to both by silently falling back to a full-resolution grid, so a misconfigured coarse resolution manifests as an unexplained slowdown rather than an error.

**NSM/mesh/main.py:277 — narrow_band default flips with the use_vtk flag**

`sdf_grid_to_mesh(..., narrow_band=False, ...)` (:277) vs `sdf_grid_to_mesh_vtk(..., narrow_band=True, ...)` (:378). Both are called with three positional args only (:242/:244 and :751/:753), so toggling `use_vtk` also silently toggles whether the volume is cropped. Neither docstring notes the asymmetry.

**NSM/mesh/main.py:440 — find_object_bounds_random_sampling is dead and was explicitly superseded**

Zero call sites anywhere in the repo (grep over all .py/.md excluding build/), zero tests, absent from the consumer. It is the random-sampling bounds finder that the deterministic coarse pass replaced — create_mesh_adaptive's own docstring advertises 'avoids the randomness/clumping of point sampling' (:597-598). It is also non-deterministic (`torch.rand` at :470, unseeded), which the docstring does not mention. Still re-exported to the world by `from .main import *`.

**NSM/mesh/main.py:525 — create_grid_samples_in_bounds silently requires numpy arrays, not the tuples its docstring implies**

`padded_min = bounds_min - pad_world` (:525-526) is elementwise scalar arithmetic; a tuple raises TypeError. The docstring calls the parameters '(x, y, z) minimum bounds' (:508-509), which reads as a tuple. Only ever reached with the numpy output of coarse_bounds_from_sign_change (:699, :711).

**NSM/mesh/main.py:667 — Multi-object adaptive meshing shares one AABB across all surfaces**

`coarse_sdf_flat = torch.min(coarse_sdf_values_flat, dim=1)[0]` — the bounds are the union over all decoder outputs, so a small cartilage surface is meshed on the grid sized for the femur, losing the resolution benefit the function exists to provide. Stated only as the inline comment 'Union across objects' (:667); the Args/Returns block (:600-631) is silent. NSM/reconstruct/main.py:1102-1114 calls this with objects_per_decoder up to 4.

### `NSM/models/__init__.py`

**NSM/models/__init__.py:1 — Public API surface is polluted by a wildcard import**

`from .deep_sdf import *` with no __all__ in deep_sdf.py binds torch, nn, np, F and warnings as attributes of NSM.models (verified). `from NSM.models import *` therefore shadows a caller's own torch/nn/np. It also determines which of the two Sine classes wins (see separate finding).

### `NSM/models/triplanar.py`

**NSM/models/triplanar.py:384 — Legacy triplanar path has a silent performance cliff on ungrouped latents**

torch.unique_consecutive only collapses adjacent duplicate rows. Point ordering is preserved either way (groups are contiguous runs by construction, so the cat at line 398 reproduces the input order — no misalignment), but if a batch interleaves latents rather than grouping them, the VAE decoder is invoked once per point (line 391) instead of once per object. Nothing documents that callers must group rows by latent.

### `NSM/reconstruct/__init__.py`

**NSM/reconstruct/__init__.py:1 — Star-import __init__ files re-export third-party modules as part of the package API**

With no `__all__` anywhere, `from .main import *` exports every non-underscore module-level name. Measured by importing each subpackage: NSM.reconstruct exposes 44 names including `torch`, `np`, `os`, `sys`, `copy`, `time`, `wandb`, `mskt`, `fnmatch`, `logging`, `logger`, plus `sinkhorn` and `eikonal_loss`; NSM.datasets (`from .sdf_dataset import *`) exposes 33 including `torch`, `np`, `vtk`, `pcu`, `meshfix`, `hashlib`, `zipfile`, `gc`; NSM.models (`from .deep_sdf import *`) exposes 24 including `torch`, `nn`, `F`, `np`, `warnings`; NSM.mesh (`from .main import *`) exposes 22 including `torch`, `vtk`, `pv`, `inspect`. Every one is a name a consumer can bind to and that the library cannot rename or drop without a breaking change. By contrast NSM.dependencies (explicit single import) exposes exactly 1 and NSM.train exactly 3.

### `NSM/reconstruct/main.py`

**NSM/reconstruct/main.py:616 — In-code TODO admits the multi-surface truncation is a hack that assumes surface 0 is the bone**

Lines 613-624: when the decoder emits more surfaces than the caller supplied ground truth for, the loop `break`s. The comment says 'this is a bit of a hack, should be handled better / right now it assumes the first surface is the bone / only of interest'. This is the mechanism by which a bone+cart decoder is fit to a bone-only target, and it silently depends on the bone being decoder output 0. Work the plan should absorb.

**NSM/reconstruct/main.py:834 — `sigma_rand_pts` default differs by 10x between reconstruct_mesh and get_mean_errors**

reconstruct_mesh:834 `sigma_rand_pts=0.001`; get_mean_errors:1276 `sigma_rand_pts=0.01`. get_mean_errors always forwards its own value (line 1331), so calling reconstruct_mesh directly versus through get_mean_errors samples random points at a different noise scale by default. Neither default is documented. Same pattern with `n_pts_per_axis_mean_mesh=128` (line 824) vs `n_pts_per_axis=256` (line 817).

**NSM/reconstruct/main.py:1153 — Unconditional debug prints on the production reconstruction path**

Lines 1153-1154 (`print('length of meshes: ', ...)`, `print('length of orig_mesh: ', ...)`), line 1165 (`print('finished computing recon loss')`), line 1190 (`print('done wandb stuff')`), and line 1408 (`print(key, item)` in get_mean_errors) are not gated on `verbose`. Everything else in these functions is. Since the pipeline parses step stdout as `[PROGRESS]` lines plus a trailing JSON blob (kneepipeline CLAUDE.md), stray stdout from a library is a hazard -- the NSM fit is run in a subprocess by steps/run_nsm.py.

### `NSM/reconstruct/reconstruct_latent_S3.py`

**NSM/reconstruct/reconstruct_latent_S3.py:58 — reconstruct_latent_S3 is exported as public API but has never been exercised**

Exported at NSM/reconstruct/__init__.py:2 yet called from nowhere in the repo, testing/, or the kneepipeline consumer. It carries the NameError at line 127, the missing wandb import at line 316, the `latent_loss_` unbound at line 320, and the div-by-zero at line 97 -- all of which mean any real run would have surfaced at least one. The module-level TODO at lines 10-14 describes the feature (Sim(3) pose+scale optimisation, arXiv 2004.09048) as still unimplemented in the main path.

### `NSM/train/train_deep_sdf.py`

**NSM/train/train_deep_sdf.py:210 — TODO in the validation block describes work the refactor should absorb**

Lines 210-212: '# TODO: Change this to just accept the config? / or... update all parameters to be the same in the config and the function call? / this will just allow unpacking of the config dict.' The block below it is a single 46-line call to `get_mean_errors` with ~30 keyword arguments (lines 213-258), six of which are commented-out placeholders (lines 220-221, 223-224, 226, 242-243). The same call exists in a drifted form in multi_head:125-149 (missing 8 arguments) and deprecated/train_deep_sdf_multi_surface_orig.py:171-215 (hardcoded objects_per_decoder=2, missing `device`). This argument list is the largest single duplication in the subsystem.

**NSM/train/train_deep_sdf.py:281 — train_epoch's n_surfaces default of 2 contradicts train_deep_sdf's objects_per_decoder default of 1**

`train_epoch(..., n_surfaces=2)` (line 281) versus `config.setdefault("objects_per_decoder", 1)` (line 56). The only in-repo caller passes it explicitly (line 175), so the default is reachable only by direct use of `train_epoch` — which is a public, exported symbol with no docstring. Calling it directly with a single-surface model silently takes the multi-surface branch at line 331 and indexes `gt_sdf[:, :, 1]`.

**NSM/train/train_deep_sdf.py:422 — multi_object_overlap is a config key whose only implementation is an unconditional raise**

Lines 421-428: `if config.get("multi_object_overlap", False) == True: raise Exception("Not implemented yet")` followed by seven lines of design commentary. The key is a documented-looking config knob that can only crash training mid-epoch, after data loading and the first forward pass. Present identically in multi_head:259-266 and deprecated/train_deep_sdf_multi_surface_orig.py:356-363.

**NSM/train/train_deep_sdf.py:510 — Enabling eikonal loss silently doubles the forward-pass cost and is untested**

Lines 506-514 run a SECOND full forward pass (`pred_sdf_grad = model(inputs_grad, epoch=epoch)`) per chunk purely to obtain gradients, on top of the pass at line 388, plus an autograd.grad per surface inside eikonal_loss (NSM/losses.py:45-56). Setting `eikonal_weight > 0` therefore more than doubles training time with no note anywhere. The repo's own CLAUDE.md says 'EIKONAL LOSS HAS NOT BEEN TESTED'; default_config.json ships `eikonal_weight: 0.0`, and there is no test covering this branch. The loss is also computed on the UNCLAMPED prediction while the L1 term uses the clamped one (line 398) — an undocumented inconsistency.

**NSM/train/train_deep_sdf.py:573 — grad_clip is applied to the model only, never to the latent codes**

Line 572-573: `torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])`. `latent_vecs` is an nn.Embedding that is a first-class optimizer param group (NSM/utils.py:376-383) and receives gradients from both the L1 term and the code-regularization term, but its gradients are never clipped. Nothing documents this asymmetry, and `grad_clip` reads as a global setting. Identical in multi_head:403 and both deprecated files.

**NSM/train/train_deep_sdf.py:620 — mesh_names exists in config but is never used to label anything**

`mesh_names` is defaulted (line 57), length-validated (lines 64-70), warned about (71-78), and persisted via save_model_params (line 198) — but no code in train/ ever reads its contents. Per-surface metrics are still logged positionally as `l1_loss_0`, `l1_loss_1` (line 620), and the per-surface print at line 603 is an unlabelled list. The one place the names would eliminate the positional ambiguity they were added to fix, they are not consulted.

### `NSM/train/utils.py`

**NSM/train/utils.py:76 — The positional indexing the LR fix removed still survives in the logging helper**

`add_plain_lr_to_config` calls `resolve_schedule_targets(...)` then converts the result straight back into list positions with `targets.index(LR_TARGET_MODEL)` / `targets.index(LR_TARGET_LATENT)` (lines 76-78) so it can index `schedule_specs[idx]` (line 87). It is correct today, but it is the one remaining place where a schedule entry is reached by position, and it still accepts caller-supplied `idx_model`/`idx_latent` overrides (line 62) that bypass the target lookup entirely (exercised by testing/NSM/test_lr_schedules.py:535-538).

**NSM/train/utils.py:87 — add_plain_lr_to_config mutates the caller's config in place while also returning it**

Lines 87-97 write six new keys directly into the passed dict and line 97 returns the same object. All four call sites use the return-value idiom `config = add_plain_lr_to_config(config)` (train_deep_sdf.py:80, multi_head:54, deprecated orig:26, deprecated multi_surface_orig:49), which reads as if it were pure. The docstring says nothing about mutation. testing/NSM/test_lr_schedules.py:520 defends against this with `copy.deepcopy(raw)` — evidence the aliasing is already a known trap. Because `save_model_params` (line 198) dumps the whole config, these derived logging keys are also persisted into model_params_config.json as if they were user-supplied settings.

**NSM/train/utils.py:115 — get_profiler hardcodes a schedule that only profiles the first 8 steps and has no docstring**

Lines 115-125: `torch.profiler.schedule(wait=0, warmup=2, active=6)` with no `repeat` argument and traces written to a fixed relative `./log` directory (line 119). Because the profiler is entered ONCE around the entire epoch loop (train_deep_sdf.py:162) and `profiler.step()` is called per EPOCH (line 265), the 'steps' are epochs — so the profiler captures epochs 3-8 and then goes inert for the rest of the run, and `./log` lands in whatever the process CWD happens to be. None of this is documented; the function has no docstring, and `config["profiler"]` (line 116) is read with `[]` so a missing key is fatal.

### `NSM/utils.py`

**NSM/utils.py:9 — Importing NSM prints to stdout unconditionally when schedulefree is absent**

`except ImportError: print("schedulefree not found, skipping import")` runs at import time, and NSM/__init__.py:9 imports utils, so every `import NSM` — including the downstream consumer's `from NSM.models import TriplanarDecoder` (kneepipeline/steps/run_nsm.py:85) — emits this line on stdout. Observed on every python invocation during this audit. The consumer's step protocol parses stdout ('stdout: progress lines followed by a JSON result as the last line'); the print lands before the JSON so it does not break the last-line contract today, but a library writing to stdout at import is a hazard for exactly that reason. `warnings` is already imported in the same file (line 11).

### `docs/KNOWN_ISSUES.md`

**docs/KNOWN_ISSUES.md:183 — Open action recorded in the LR post-mortem that the refactor plan should absorb**

'**Open action:** re-tune learning rates under the fixed mapping and compare against the current production models before assuming either is better. Not yet done.' Every pre-Aug-2026 hyperparameter search optimized under the swapped mapping (lines 176-182), so the LR pair shipped in default_config.json:83-98 is tuned for a mapping the code no longer implements.

---

## Documentation inaccuracies (25)

Prose documentation checked line-by-line against the code. `claim` quotes the document;
`reality` is what the code does, with the line that proves it.

### [misleading] `CLAUDE.md:120`

> - `train_deep_sdf_multi_head.py`: Multi-head training for multiple surfaces

**Reality.** The module is deprecated and known broken; the Architecture section lists it as an ordinary training pipeline with no caveat, while docs/KNOWN_ISSUES.md §2 documents it as silently optimizing only the last model. Calling it emits a DeprecationWarning saying so, and `get_optimizer` at line 85 is passed the leaked loop variable `model` rather than `models`, so every decoder but the last stays at initialization. An agent reading only CLAUDE.md's architecture map would pick this entry point for multi-surface work.

**Evidence.** NSM/train/train_deep_sdf_multi_head.py:27-33,85-91

### [misleading] `CLAUDE.md:34`

> # Quick dev cycle (format + run loader tests)
make quick-test

**Reality.** `quick-test` cannot reach its test phase. It depends on `format`, which runs `black NSM/ testing/`; black exits 123 because testing/testing_h5_vs_np_loading/save_and_load_h5_vs_np.py line 1 is a shell command (`salloc -c 2 --mem=12gb ...`), not Python. I ran the exact command on a copy of the tree: '9 files reformatted, 43 files left unchanged, 1 file failed to reformat', exit 123. make aborts the target chain there, so test-loader never runs. `make format-check` fails identically (exit 123).

**Evidence.** testing/testing_h5_vs_np_loading/save_and_load_h5_vs_np.py:1; Makefile:49-53,83

### [misleading] `CONTRIBUTING.md:87`

> make dev
    make requirements

**Reality.** Neither target exists. The Makefile defines help, install, install-dev, test, test-loader, test-coverage, lint, format, format-check, clean, env-setup, quick-test. Running either prints 'No rule to make target'. The correct command is `make install-dev`. The same phantom targets are wired into CI: .github/workflows/docs.yml:25 runs `make requirements dev` and :27 runs `make docs`, so the documentation-site build cannot succeed either.

**Evidence.** Makefile:23-67; .github/workflows/docs.yml:25-27

### [misleading] `CONTRIBUTING.md:102`

> $ make autoformat

**Reality.** There is no `autoformat` target; the formatting target is `format` (which itself fails, see the CLAUDE.md quick-test finding). Line 141 repeats the error in the doc's summary hint, 'make autoformat test', presented as the command to run before every push.

**Evidence.** Makefile:49-53

### [misleading] `DEVELOPMENT.md:18`

> # Install development dependencies
pip install -r requirements-dev.txt

# Install NSM in development mode
pip install -e .

**Reality.** This leaves the package unusable. requirements-dev.txt carries only test/lint/docs tooling; the runtime dependencies live in requirements.txt (mskt, pykeops, einops, pymeshfix, scikit-image, pandas, wandb, tqdm) and `pip install -e .` cannot supply them because pyproject.toml declares `dependencies = []`. `import NSM` alone survives (NSM/__init__.py imports only NSM.utils), but NSM.datasets, NSM.mesh, NSM.reconstruct and NSM.train all fail on the missing pymskt. The Option 2 venv recipe at lines 36-41 has the identical omission, and both diverge from `make install-dev`, which installs requirements.txt first.

**Evidence.** pyproject.toml:30; Makefile:26-29; NSM/datasets/sdf_dataset.py:2

### [misleading] `docs/MULTI_SURFACE_REGISTRATION.md:75`

> The `load_reference_mesh()` method now supports creating reference meshes from multiple surfaces

**Reality.** The multi-surface branch raises UnboundLocalError and can never return a combined reference mesh. Inside `elif isinstance(self.reference_mesh, int):`, the list-valued `mesh_to_scale` branch assigns `self.reference_mesh = combine_meshes(...)` but never binds the local `mesh`; control then falls through to the unconditional `self.reference_mesh = Mesh(mesh)` at line 1360, which references the unbound local. Even if `mesh` were bound, that line would overwrite the combined mesh the branch just built. Reproduced directly: constructing a MultiSurfaceSDFSamples with reference_mesh=0, list_mesh_paths=[[a,b],[a,b]], mesh_to_scale=[0,1] and calling load_reference_mesh() raises `UnboundLocalError: local variable 'mesh' referenced before assignment`.

**Evidence.** NSM/datasets/sdf_dataset.py:1348-1360

### [misleading] `.claude/plans/BREAKING_CHANGE_PROPOSAL.md:51`

> - [x] ✅ Implement warning system for potentially incorrect sigma values

**Reality.** No sigma warning system exists. `SDFSamples.preprocess_inputs()` contains only two `scale_jointly` guards on center_pts/norm_pts and never inspects sigma_near/sigma_far. Grepping sdf_dataset.py for any sigma-related warn/print/threshold turns up a single timing print at line 1849. The `warnings` module is imported (line 10) but never used for sigma.

**Evidence.** NSM/datasets/sdf_dataset.py:1092-1105

### [misleading] `.claude/plans/BREAKING_CHANGE_PROPOSAL.md:50`

> - [x] ✅ Add comprehensive documentation about current dual-mode behavior

**Reality.** No such documentation exists in the code. `grep -i 'coordinate space' NSM/datasets/sdf_dataset.py` returns nothing, and the sigma_near docstring is a single line — 'Standard deviation/scale of the distribution for points near the surface. Defaults to 0.01.' — with no mention that its coordinate space flips with `scale_jointly`.

**Evidence.** NSM/datasets/sdf_dataset.py:781

### [stale] `CLAUDE.md:39`

> - Black formatting with 100 character line length

**Reality.** Configured but not achieved on main. `black --check NSM/ testing/` reports 9 files that would be reformatted — including NSM/reconstruct/main.py, NSM/mesh/main.py, NSM/losses.py, NSM/models/triplanar.py — plus 1 unparseable. The companion claim on line 40 ('isort with black profile') is likewise unenforced: `isort --check-only NSM/ testing/` fails on 37 files. CI does not gate on either; its lint job is `continue-on-error: true`.

**Evidence.** NSM/reconstruct/main.py:1 (one of 9 unformatted files); .github/workflows/build-test.yml:13-22

### [stale] `CLAUDE.md:42`

> - Pytest is configured in pyproject.toml

**Reality.** It is configured, but the configuration contradicts the line directly above it ('Tests live in `testing/` directory (not `tests/`)'): `testpaths = ["tests"]` names a directory that does not exist. A bare `pytest` only works by accident — pytest 8.4.2 emits 'PytestConfigWarning: No files were found in testpaths; ... Searching recursively from the current directory instead' and then collects 154 items. The neighbouring `addopts = "-k 'not train_test.py'"` filters a file that also does not exist anywhere in the repo. README.md:188 and DEVELOPMENT.md:59 both tell users to run bare `pytest`, so they all ride on that fallback.

**Evidence.** pyproject.toml:93-96

### [stale] `CLAUDE.md:28`

> # Lint with flake8
make lint

**Reality.** `make lint` always fails on main: `flake8 NSM/ testing/` emits 445 violations and exits 1 (187 W293, 131 E501, 42 E231, 35 W291, 8 F841, 4 F821 undefined names, ...). This is known and accepted — the CI lint job is marked continue-on-error — but neither CLAUDE.md nor README.md nor DEVELOPMENT.md says so, and all three present it as a routine pre-commit step. The CI comment that documents it also states a count of '~600 pre-existing flake8 violations'; the actual figure is 445.

**Evidence.** .github/workflows/build-test.yml:13-22; Makefile:46-47

### [stale] `CONTRIBUTING.md:53`

> Create a development branch - all changes should merged with the `NSM`-`development` branch ... **Do not** work on the `main` branch.

**Reality.** Not how the repo is run. The three most recent merges all target main directly from topic branches: 73a0326 'Merge pull request #11 from gattia/sync-default-config', c7fb91b 'Merge pull request #10 from gattia/lr-schedule-target-key', 58066b4 'Merge pull request #9 from gattia/fix-lr-schedule-mapping'. Branch naming follows the topic-branch convention the same section offers as an alternative, not a long-lived `development` integration branch.

**Evidence.** git log --oneline (73a0326, c7fb91b, 58066b4 all merge PRs into main)

### [stale] `DEVELOPMENT.md:202`

> # Run specific performance tests
pytest testing/performance/ -v

**Reality.** There is no testing/performance/ directory. testing/ contains NSM/, testing_h5_vs_np_loading/ and testing_sdf_calculation_times/ — the last of which holds the actual timing scripts (time.py, time_pcu_sdf.py, time_vtk_vs_pcu.py, time_vtkimplicit.py), and is excluded from flake8 as scratch. The command exits 4 with 'file or directory not found'.

**Evidence.** testing/testing_sdf_calculation_times/ (time.py, time_pcu_sdf.py, time_vtk_vs_pcu.py, time_vtkimplicit.py); .flake8:20-21

### [stale] `README.md:215`

> - [`docs/MULTI_SURFACE_REGISTRATION.md`](docs/MULTI_SURFACE_REGISTRATION.md) - Multi-surface registration functionality

**Reality.** The docs/ listing is incomplete: docs/ also contains KNOWN_ISSUES.md, added Aug 2026 and the file CLAUDE.md:97 makes mandatory for any numerical-behaviour change. It is the one document a user with old training runs most needs to find from the README.

**Evidence.** docs/KNOWN_ISSUES.md:1

### [stale] `README.md:217`

> API documentation is planned for future development. Consider using `pdoc` for auto-generated docs:

**Reality.** Contradicts line 2 of the same file, which links '|[Documentation](http://anthonygattiphd.com/NSM/)|' as though a doc site exists. The workflow meant to publish it cannot run: .github/workflows/docs.yml invokes `make requirements dev` and `make docs`, neither of which is a Makefile target, so the website job fails before the pdoc step the README describes as hypothetical. (Whether the live URL currently serves anything is UNVERIFIED — no network access from this session.)

**Evidence.** .github/workflows/docs.yml:25-27; Makefile:23-67

### [stale] `docs/KNOWN_ISSUES.md:192`

> - `.claude/plans/NSM_CODE_HEALTH_REFACTOR.md` §4 — this fix as the migration template

**Reality.** ~~That file does not exist in the repo.~~ **RESOLVED by merge `458e6e6`.** It was a branch artifact: `docs/KNOWN_ISSUES.md` shipped on `main` while the plan it cites lived only on `plan-code-health-refactor`, so neither tree contained both. Merging `main` in put both in one tree and the three citations (lines 192, 221, 251) now resolve. Keep the entry as a record of the failure mode: a document whose stated purpose is to be answerable years later had two open actions pointing at nothing, purely because two branches each held half the story.

**Evidence.** .claude/plans/ (directory listing: NSM_RECTIFIED_FLOW_CORRESPONDENCE.md, NSM_TRAINING_IDEAS.md, completed/)

### [stale] `docs/MULTI_SURFACE_REGISTRATION.md:170`

> - Updated docstrings and class documentation

**Reality.** The MultiSurfaceSDFSamples class docstring still documents the superseded implementation: 'When mesh_to_scale is a list, meshes are combined using VTK's vtkAppendPolyData'. The actual `combine_meshes` uses the Mesh `+` operator, and this same document's Technical Details section (lines 127-140) explicitly contrasts that with 'the original VTK append approach'. A reader in the code sees the story this document says was replaced.

**Evidence.** NSM/datasets/sdf_dataset.py:1589 vs NSM/datasets/sdf_dataset.py:2188-2193

### [stale] `.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md:4`

> Add `sigma_coordinate_space` parameter to decouple sigma sampling from `scale_jointly` flag, enabling explicit control over coordinate space interpretation.

**Reality.** Unimplemented in its entirety — `sigma_coordinate_space` occurs in no .py file under NSM/. Neither SDFSamples.__init__ nor MultiSurfaceSDFSamples.__init__ accepts it, preprocess_inputs performs none of the Step 3 validation, and neither get_hash_params includes it, so caches built under the two coordinate interpretations still collide. Flagged as aspirational rather than wrong: the plan carries no completion markers, and the baseline code it quotes still matches the file exactly.

**Evidence.** NSM/datasets/sdf_dataset.py:806-838 (SDFSamples.__init__ signature), :1092-1105 (preprocess_inputs), :1395-1407 (get_hash_params)

### [cosmetic] `CLAUDE.md:152`

> A warning is emitted during training if `mesh_names` is not provided.

**Reality.** The warning is conditional on `objects_per_decoder > 1`, not on mesh_names being absent. In `train_deep_sdf`, the `elif config["objects_per_decoder"] > 1:` guard means a single-surface config with no mesh_names trains silently. (The deprecated multi-head path warns unconditionally, and validates length against `objects_per_decoder * len(models)` rather than `objects_per_decoder` as line 150 states.)

**Evidence.** NSM/train/train_deep_sdf.py:63-78

### [cosmetic] `CLAUDE.md:193`

> Several groups may share a target: every decoder and the classification heads all take the `model` schedule.

**Reality.** NSM never builds a classification-heads param group. `get_optimizer` emits exactly one `latent` group plus one `model_{idx}` group per decoder. `CLASSIFICATION_HEADS_GROUP_NAME` is defined at utils.py:19 but referenced nowhere else in NSM/; the only construction of such a group is a hand-built dict in a test. The code's own docstring hedges correctly ('and, when present, ``classification_heads``'); CLAUDE.md drops the hedge and asserts they exist.

**Evidence.** NSM/utils.py:376-392

### [cosmetic] `CONTRIBUTING.md:19`

> If you cannot find you bug, follow the instructions in the [Bug Report](https://github.com/gattia/cycpd/issues/new/choose) template.

**Reality.** Points bug reports at a different project, cycpd. Line 39 leaves the same template artifact in prose: 'You will need basic git proficiency to be able to contribute to cycpd.' The feature-request link on line 34 was updated to gattia/NSM, so this is an incomplete find-and-replace rather than an intentional cross-reference.

**Evidence.** CONTRIBUTING.md:5 ("This guide is inspired by DOSMA"), :34 (gattia/NSM link)

### [cosmetic] `DEVELOPMENT.md:131`

> │   ├── reconstruct/      # Reconstruction utilities
│   └── utils.py          # Utility functions

**Reality.** The project-structure tree is missing four of the package's directories/modules, including one CLAUDE.md lists as a core module: NSM/mesh/ (marching cubes, refinement, interpolation), NSM/losses.py, NSM/configs/ (default_config.json and its generator) and NSM/dependencies/ (sinkhorn). The `└──` on utils.py asserts it is the last child of NSM/, which it is not.

**Evidence.** NSM/mesh/main.py:1; NSM/losses.py:1; NSM/configs/default_config.json; NSM/dependencies/sinkhorn.py

### [cosmetic] `Makefile:20`

> @echo "  env-setup        Setup conda development environment"

**Reality.** The help listing ends here, omitting `quick-test`, which is a real target defined at line 83 and is the one CLAUDE.md:33-34 advertises as the quick dev cycle. `.PHONY` on line 4 is also incomplete — it omits test-loader, format-check, env-setup and quick-test, so any same-named file in the repo root would shadow them.

**Evidence.** Makefile:4, Makefile:83-84

### [cosmetic] `README.md:11`

> Steps to update this package for new repository: 
4. update `requirements.txt` and `dependencies` in `pyproject.toml`
     - To do - can dependencies read/update from requirements.txt?

**Reality.** Leftover cookiecutter instruction addressed to the package author, sitting in the Introduction where a reader expects a project description. The list has one item and it is numbered 4, so steps 1-3 were deleted without renumbering. Its own TODO is still open: pyproject.toml declares `dependencies = []` while requirements.txt lists 15 packages.

**Evidence.** pyproject.toml:30; requirements.txt:1-30

### [cosmetic] `docs/KNOWN_ISSUES.md:25`

> | **Fixed in** | `fix-lr-schedule-mapping`, Aug 2026 |

**Reality.** The branch named shipped only the first half. `fix-lr-schedule-mapping` (PR #9, merged as 58066b4) mapped schedules by param-group name. The `Target` key contract that the rest of this entry documents — the migration guard, the {target: schedule} dict, the swapped shipped defaults — landed on branch `lr-schedule-target-key` (PR #10, merged as c7fb91b). A reader trying to date a checkout against 'the fix' would place the cutoff one PR too early.

**Evidence.** git log: 58066b4 (Merge PR #9 fix-lr-schedule-mapping), c7fb91b (Merge PR #10 lr-schedule-target-key), 0d87e3c 'Declare LR schedule targets per entry'

---

## Document-level verdicts

| Document | Verdict |
|---|---|
| `CLAUDE.md` | **partly-stale** |
| `README.md` | **partly-stale** |
| `CONTRIBUTING.md` | **stale** |
| `DEVELOPMENT.md` | **partly-stale** |
| `Makefile` | **partly-stale** |
| `docs/MULTI_SURFACE_REGISTRATION.md` | **partly-stale** |
| `docs/KNOWN_ISSUES.md` | **partly-stale** |
| `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` | **aspirational** |
| `.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` | **aspirational** |
| `.claude/plans/HYBRID_OPTIMIZER_REPORT.md` | **accurate** |
| `examples/load_trained_model.py` | **accurate** |

