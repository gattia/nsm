# Defect work list

**Deliberately not GitHub issues.** The refactor is expected to churn this code heavily, so
these live here until they either get fixed in passing or survive to the end — whatever is
still open when Phase 4 lands is worth filing then.

Every entry below was **executed**, not inferred. The evidence, with expected-vs-actual and
`file:line`, is in [`TEST_HARNESS_NOTES.md`](TEST_HARNESS_NOTES.md); this file is the
"what to do about it" half.

**Each one has an `xfail(strict=True)` test in `testing/NSM/regression/` that asserts the
behaviour it should have.** So the workflow is:

1. `pytest testing/NSM/regression/ -q -rx` lists them, keyed to the numbers below.
2. Fix one. Its xfail starts passing, pytest reports `XPASS(strict)`, **the suite goes red.**
3. Delete the `xfail` mark and tick the box here. Now it is a real regression test.

If step 2 does not happen, the fix did not do what you think it did.

Ordered by file and function rather than by severity, because the plan is to walk the code
function by function and these should surface when you open the relevant one. The severity
triage is the table at the top.

Status: `[ ]` open · `[~]` in progress · `[x]` fixed (and its pinning test updated)

---

## Triage

| # | Defect | Severity | Cost |
|---|---|---|---|
| [1](#1-datasetssdf_datasetpy) | Cache key omits parameters that change cached content | **High** — silently wrong training data | Small fix, wide blast radius |
| [2](#1-datasetssdf_datasetpy) | `reference_mesh` hashed by object identity | Medium — cache never hits | Small |
| ~~[3](#1-datasetssdf_datasetpy)~~ | ~~Sampling not reproducible; `random_seed` is a decoy~~ | ~~**High**~~ | **DONE**, Aug 2026 |
| [4](#1-datasetssdf_datasetpy) | `store_data_in_memory=True` raises | Medium — advertised option, unusable | Trivial |
| [5](#1-datasetssdf_datasetpy) | `p_near_surface=0` crashes | Low | Trivial |
| [6](#1-datasetssdf_datasetpy) | `get_pts_center_and_scale` ignores args, mutates input | Medium | Small |
| [7](#2-modelstriplanarpy) | `padding` absent from checkpoints | **High** — silent wrong-scale sampling | Needs a format decision |
| [8](#2-modelstriplanarpy) | `normalize_coordinates(padding=)` ignored | Low | Trivial |
| [9](#2-modelstriplanarpy) | Every VAE layer registered twice → 1.92× checkpoints | Medium | Needs a migration shim |
| [10](#3-traintrain_deep_sdfpy) | `enforce_minmax` clamps predictions, killing gradients | Medium — config semantics | Docs, or a decision |
| [11](#3-traintrain_deep_sdfpy) | `train_deep_sdf` returns nothing | Low — blocks observability | Trivial |
| [12](#4-reconstructmainpy) | Early return drops requested keys | Medium | Small |
| [13](#1-datasetssdf_datasetpy) | Mesh **content** is not in the cache key | Medium — edit a mesh in place, get the old data | Small |
| [14](#1-datasetssdf_datasetpy) | `Pool` deadlocks on a second dataset in one process | Low — hangs, does not corrupt | Needs a decision |

---

## 1. `datasets/sdf_dataset.py`

### `get_hash_params` / `create_hash` (`:1388`, `:1413`, `:1973`)

- [ ] **1. Add `mesh_to_scale`, `uniform_pts_buffer` and `subsample` to the cache key.**
  All three change what is written to the cache and none are in it, so two runs differing
  only in one share a key and the second reuses the first's data. Severity is not uniform:
  `mesh_to_scale` invalidates every array (it decides which surface drives centering and
  normalization), `uniform_pts_buffer` the points, `subsample` only the index arrays.
  If fixing incrementally, `mesh_to_scale` first.

  *Blast radius:* every existing cache entry misses and is regenerated. That is the point,
  but it is not free — regeneration is not reproducible until #3 lands, so a rebuild
  produces *different* data, not the same data.
  *Needs a `docs/KNOWN_ISSUES_HISTORY.md` entry* — it silently changes training output for
  inputs that previously ran without error.
  *Pinned by:* `test_dataset_cache.TestUnhashedParametersCollide` (5 tests, including the
  measured 4.4× loss of interior samples for the small surface).

- [ ] **2. Hash `reference_mesh` by identity of the geometry, not the object.**
  A `Mesh` object is stringified into the key and `Mesh.__str__` contains its memory
  address, so the key changes on every construction and the cache never hits. Hash the
  path, or a content digest. A path string already hashes stably — that is the workaround
  people are implicitly relying on.
  *Pinned by:* `test_dataset_cache.TestReferenceMeshHashing`.

  While in here: the reload guard at `:1764-1771` compares `len(data["pos_idx"])` against
  the number of *meshes*, never against the subsample the arrays were sized for. It is the
  check that should have caught #1's `subsample` case and cannot.

- [ ] **13. Put the mesh content in the cache key, or say out loud that it is not there.**
  The key is `md5(params + mesh paths)`. Edit a mesh in place and the key does not move, so
  the stale `.npz` is served and you silently train on the old geometry. Found while fixing
  #3, and the same class as #1 and #2 — something that changes the cached content is not in
  the key.

  Not obviously worth a full content hash on every load for a large dataset; file size and
  mtime would catch the realistic case. The decision is which, not whether.

  *Not currently pinned by a test.*

- [ ] **14. `Pool` deadlocks if a dataset was already built in this process.** Constructing a
  second `SDFSamples`/`MultiSurfaceSDFSamples` with the default
  `multiprocessing=True` hangs indefinitely with idle workers (`:954-957`). Fork-after-VTK.
  Reproduces on pre-#3 code, so it is long-standing rather than new.

  Cheapest honest fix is a `spawn` context; the cheapest fix of all is documenting it on
  `multiprocessing=`. Either beats a hang with no message.

  *Worked around in:* `test_dataset_cache.TestSeedDerivation`, which builds its two
  datasets in separate subprocesses.

### `read_meshes_get_sampled_pts` / the sampling path (`:404`)

- [x] **3. Make sampling reproducible.** *Fixed Aug 2026. History:
  `docs/KNOWN_ISSUES_HISTORY.md` §4.*

  Both halves are in. Upstream, `pymskt.Mesh.rand_pts_around_surface` gained a `seed`
  ([pymskt#54](https://github.com/gattia/pymskt/issues/54) →
  [#55](https://github.com/gattia/pymskt/pull/55), released as **0.1.21**, now pinned in
  `requirements.txt`). In NSM, `read_mesh_get_sampled_pts` and
  `read_meshes_get_sampled_pts` take a `seed`, `SDFSamples.random_seed` reaches them, and
  `derive_seed` splits it per (subject, sampling pass, surface) — one shared seed would
  have handed the near- and far-surface passes the same base points and given bone and
  cartilage the same offsets.

  The subject component is keyed on **mesh content**. Not the path (moving your data would
  change your samples) and not the list position (reordering `list_mesh_paths` would change
  every subject's data while every cache filename stayed the same).

  Two things fell out of it worth remembering:

  - `include_seed_in_hash` was deleted. Harmless while the seed changed nothing; a
    cache-poisoning switch the moment it changed data, and nothing set it.
  - The `multiprocessing=True` correlation below was closed as a side effect, but only
    because the seed is derived per subject.

  *Now pinned by:* `test_dataset_cache.TestSeeding` (both former xfails, now real tests) and
  `TestSeedDerivation` (5) — different seeds differ, the two sampling passes decorrelate,
  list order does not matter, mesh location does not matter, and `multiprocessing` does not
  change the data.

### `MultiSurfaceSDFSamples.__getitem__` (`:2038`)

- [ ] **4. Fix `store_data_in_memory=True`.** `:2158-2162` reads `time_` and `size`, bound
  only in the `store_data_in_memory is False` branch, so the first `__getitem__` raises
  `UnboundLocalError`. `SDFSamples.__getitem__:1563` guards the identical block correctly —
  copy that guard.

  The obvious workaround, `test_load_times=False`, is not one: it yields items with only
  `{"xyz", "gt_sdf"}` and `train_epoch` reads all four timing keys unconditionally
  (`train_deep_sdf.py:578-581`). Decide whether those keys are part of the batch contract
  or optional, and make both classes agree.
  *Pinned by:* `test_dataset_cache.TestConfigurationsThatDoNotRun` (3 tests).

### `get_pt_sample_combos` (`:1268`, `:1945`)

- [ ] **5. Skip zero-count sampling combos.** A `[0, sigma]` combo is passed to the sampler
  regardless (`:1820`), so `p_near_surface=0` — a reasonable request — raises inside `pcu`.
  *Pinned by:* `test_dataset_cache.TestConfigurationsThatDoNotRun::test_zero_sampling_probability_must_sample_nothing`.

### `get_pts_center_and_scale` (`:56`)

- [ ] **6. Honour `center=` and `scale=`, or delete them.** Both are rebound before they are
  read (`:88`, `:94`), so centering and scaling happen unconditionally and
  `center=False, scale=False` does nothing. Separately, `pts -= center` at `:91` mutates the
  caller's array; all three in-repo callers pass `np.copy(...)` defensively, so the
  convention exists only as a habit at the call sites and a fourth caller will not have it.
  *Pinned by:* `test_dataset_cache.TestPointCenteringAndScaling`.

---

## 2. `models/triplanar.py`

- [ ] **7. Decide how `padding` travels with a checkpoint.** It is not a learned parameter,
  so a checkpoint trained at one value loads cleanly under strict `load_state_dict` at
  another and then samples the feature planes at the wrong scale. Measured: 0.35 vs 0.1 gives
  a max SDF difference of **0.063** on a `tanh`-bounded output. `load_model` silently defaults
  it to 0.1 and the downstream consumer never passes it
  (`kneepipeline/steps/run_nsm.py:94-112` passes 15 of 16 meaningful arguments).

  Options, roughly in increasing order of how much they fix: (a) refuse to load when the
  config omits it; (b) write it into the checkpoint next to the state dict; (c) give NSM the
  public "build the model this config describes" call that `SCOPE.md` §3.1 already calls the
  highest-value API change available, and have both `load_model` and the consumer use it.
  *Pinned by:* `test_model_roundtrip.TestPaddingIsNotInTheCheckpoint`.

- [ ] **8. `normalize_coordinates(query, plane, padding=0.1)` ignores `padding`** and reads
  `self.padding` (`:312`, `:322`). Same defect class as #7 and #6 — parameter accepted and
  silently ignored — which is why they should be swept together rather than one at a time.
  *Pinned by:* `test_model_roundtrip...::test_normalize_coordinates_must_honour_its_padding_argument`.

- [ ] **9. Stop registering every VAE layer twice.** `self.layers` (a `ModuleList`) and
  `self.decoder = nn.Sequential(*self.layers)` (`:58-99`) are both child modules, so
  `state_dict()` emits each tensor under two aliased names. Every shipped checkpoint stores
  39.96M elements for 20.80M parameters — **1.92×**, i.e. the 275 MB files should be ~143 MB.
  It also means editing a checkpoint by key silently loses the edit if only one name is
  written.

  **This is a checkpoint-format break, verified in both directions.** Dropping `self.layers`
  makes every existing checkpoint fail strict load with
  `Unexpected key(s): vae_decoder.layers.*`, and a new checkpoint fails against the current
  model with `Missing key(s)`. All three shipped models are affected. It needs a migration
  shim that strips or synthesises the alias at load time — per `CLAUDE.md`, in its own module
  with a delete-when condition.
  *Pinned by:* `test_model_roundtrip.TestAliasedCheckpointEntries`.

---

## 3. `train/train_deep_sdf.py`

- [ ] **10. Document what `clamp_dist` does to gradients, or change it.** With
  `enforce_minmax`, `:401` clamps the *prediction* as well as the target, and `torch.clamp`
  passes no gradient outside its bounds — so every sample predicted outside ±`clamp_dist`
  contributes exactly zero gradient however wrong it is. Measured: **44.6%** of a freshly
  built triplanar decoder's predictions are already outside ±0.1 before the first step, and
  the shipped `default_config.json` uses `clamp_dist: 0.1` while both ShapeMedKnee configs
  use 1.0.

  Whether that stalls a given run is configuration-dependent (see `TEST_HARNESS_NOTES.md`
  §2.2 — an earlier claim that it always does was false). The defect is that the name and
  the docs describe a target transform and the behaviour is a training-dynamics knob. This
  belongs with the config-documentation work in `SCOPE.md` §2.2.
  *Pinned by:* `test_training_regression.TestClampedPredictionGradients`.

- [ ] **11. Return the per-epoch log from `train_deep_sdf`.** `:272` is a bare `return`;
  `train_epoch` builds a full `log_dict` per epoch and it goes only to `wandb`. A caller
  without a wandb key can learn nothing about a run except by reading checkpoints back off
  disk. The harness has to wrap `train_epoch` to observe anything
  (`testing/NSM/regression/_harness.py:run_training`) — fixing this deletes that wrapper.

---

## 4. `reconstruct/main.py`

- [ ] **12. Make the early return honour the caller's flags.** When the mean shape has no
  zero level set, `:946-966` returns `{mesh, latent, assd_*}` and ignores
  `return_registration_params`, `return_timing` and `orig_mesh`. The two result shapes are
  not interchangeable and the consumer reads `result["center"]` unconditionally
  (`kneepipeline/steps/run_nsm.py:230`).

  The sharper problem is that the result looks successful: `latent` is a correctly shaped
  `(1, latent_size)` tensor of zeros — the untouched `mean_latent`, never fitted — so a
  caller checking "did I get a latent" gets yes. Either return the same keys with honest
  values, or fail loudly.
  *Pinned by:* `test_reconstruction_regression.TestDecoderWithNoZeroLevelSet` (5 tests, of
  which one is the xfail on the missing keys).

---

## Also on the list, from elsewhere

Not found by the harness, carried here so the work list is one list. Detail in
`TEST_HARNESS_NOTES.md` §5.

- [ ] `pyproject.toml:95` — `testpaths = ["tests"]` names a directory that does not exist.
- [ ] `pyproject.toml` — `addopts = "-k 'not train_test.py'"` filters a file that no longer
      exists, and `-k` matches test names rather than filenames, so it never worked.
- [ ] `black --check` fails on 9 files; `make lint` reports 445 flake8 violations including
      4 `F821` undefined names, in a CI job marked `continue-on-error`.
- [ ] `.github/workflows/docs.yml` invokes `make requirements dev` and `make docs`, neither
      of which is a target.
- [ ] `testing/testing_h5_vs_np_loading/save_and_load_h5_vs_np.py:1` is a shell command in a
      `.py` file, which breaks any AST-based tooling over the repo.
