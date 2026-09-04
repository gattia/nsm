# Plan: sectioned config and a model registry

**Repo:** `gattia/nsm` (NSM). **Created:** 2026-09-04. **Origin:** maintainer rulings of
2026-09-03/04 after plan §8.0.R of `NSM_CODE_HEALTH_REFACTOR.md` found that every live
instance of the "accepted and ignored" class sits in `models/loader.py`, where a
hand-written translator maps flat config keys onto constructor arguments — one per model
type, and they drift.

## State

**Updated:** 2026-09-04 · **Status:** open

- **Next:** maintainer approves or amends §2 (the layout) and §5 (the sequencing); then
  commit 1 of §5's step 2 — the MPA loader fix, which needs one of the 17 saved MPA run
  configs from the training server, verbatim.
- **Blocked on:** the layout ruling (§2), and the paper's MPA config (§5 step 2). Neither
  blocks step 1.
- **Done:** nothing yet. The rulings this plan records are the maintainer's, 2026-09-03/04:
  `two_stage` is deleted (PR #108, second commit); `ImplicitDecoder` **stays** as the
  ShapeMed-Knee paper's MPA baseline and gets the shared vocabulary; the config splits into
  sections. PR #108's first commit, which deletes `ImplicitDecoder`, is withdrawn by that
  ruling.
- **Surprises:** the claim that motivated deleting MPA — "neither survives the
  reconstruction path" — was measured false on 2026-09-04: `_decode`'s signature dispatch
  (#105) carries both `deep_sdf.Decoder` and `ImplicitDecoder` through `reconstruct_latent`.
  What MPA cannot do is be *built* by `load_model` from the config shape the paper's runs
  used. PR #108's SCOPE §1 line "one of them (`deepsdf`) cannot be reconstructed" is the
  same stale claim and is corrected in §5 step 0.

## 1. Why

Three measured facts, none of which a single fix addresses:

1. **The translator is the defect class.** `loader.py` holds four hand-written mappings
   from config vocabulary to constructor vocabulary. §8.0.R found `_get_two_stage_params`
   had drifted from its siblings by four keys, and `_get_implicit_params` never reads
   `activation`. kneepipeline's `steps/run_nsm.py:95-109` is a *fifth* copy, for triplanar,
   in the consumer. Every copy is a place to drift and none can be checked against the
   others without a test that knows all of them.
2. **The constructors themselves disagree.** `TriplanarDecoder(latent_dim=)`,
   `Decoder(latent_size=)`, `ImplicitDecoder(latent_dim=, hidden_dim=, num_layers=)`.
   That is why the translators exist at all: the config picked one spelling and the
   classes never agreed on it.
3. **115 flat keys, five readers.** By owner: the loader reads 37, `train_deep_sdf` 62
   (17 of them `*_recon` keys that only the validation hook reads), `utils` 12 (schedules,
   checkpoints, latent init), the dataset 1 — and 25 dataset keys are a specification the
   user translates by hand, because NSM never builds a dataset from a config. A user
   setting `layer_latent_in` in the shipped triplanar config is configuring nothing, and
   nothing says so (§8.0.R). The generator already groups the keys into twelve commented
   sections; the file just does not.

The ShapeMed-Knee paper (TMI 2025) benchmarks three NSMs and names this repo as the code.
Two of the three — the implicit MLP and the MPA — were trained through a launcher on the
maintainer's server that built the decoders directly, from `default_config.json` plus
overrides (`model_type: 'modulated_periodic_activation'`, `modulated: True`,
`latent_size: 512`, 8×512 from `layer_dimensions`). **No NSM tag can build the paper's MPA
through `load_model`**, because the `implicit` translator wants a vocabulary
(`latent_dim`/`hidden_dim`/`num_layers`) the paper's runs never used. The HuggingFace repo
ships only the three triplanar models. So the MPA's public half is the class, and the
recipe lives on the server. This plan makes the recipe expressible in the library, with a
test that builds the paper's architecture.

## 2. The layout — **needs the maintainer's ruling**

Five sections, named by what reads them. Every key inside `model` is the constructor's own
parameter name; `model.type` selects the class from a registry and everything else is
passed through as keyword arguments. No translator.

```json
{
  "run":     {"project_name": "nsm", "entity_name": null, "run_name": null, "tags": ["nsm"],
              "experiment_directory": null},
  "model":   {"type": "triplanar", "latent_size": 512, "n_objects": 2, "mesh_names": ["bone", "cart"],
              "padding": 0.1, "conv_norm_type": "layer", "conv_activation": null, "...": "..."},
  "dataset": {"n_pts": [500000, 500000], "sigma_near": [0.016, 0.016], "...": "..."},
  "train":   {"n_epochs": 2001, "objects_per_batch": 64, "batch_split": 1,
              "optimizer": "AdamW", "weight_decay": 1e-4, "LearningRateSchedule": ["..."],
              "code_regularization": true, "...": "..."},
  "recon":   {"num_iterations": 2000, "lr": 5e-3, "clamp_dist": 0.1, "...": "..."}
}
```

Decisions folded into that sketch, each reversible if the maintainer prefers otherwise:

- **`recon` keys lose the `_recon` suffix.** The suffix exists to disambiguate them in a
  flat namespace; the section does that. `clamp_dist_recon` → `recon.clamp_dist`.
- **`train` absorbs the generator's eight training-time sections** (data loading, training
  loop, loss, curriculum, code regularization, optimizer, latent codes, saving). Forty-two
  keys in one section is readable; eight sub-sections of five is not. Sub-nesting the
  optimizer (`train.optimizer.{name, weight_decay, schedules}`) is a later refinement if
  the flat section proves noisy, and it costs nothing to defer.
- **`mesh_names` lives in `model`**, not `dataset`: it names the decoder's outputs, and
  `model_params_config.json` is where downstream consumers read it. The dataset-side
  declaration (`MultiSurfaceSDFSamples(mesh_names=)`) is unchanged and is still checked
  against it by `train_deep_sdf`.
- **One vocabulary across the three constructors**, chosen by majority of existing use:
  `latent_size` (Decoder, every config, every consumer) over `latent_dim` (Triplanar,
  Implicit); `n_objects` (all three already); `layer_dimensions` as a list (Decoder) over
  `hidden_dim` + `num_layers` (Implicit); `activation` / `final_activation` as strings
  (Decoder) over a `block_factory` object and a callable (Implicit). `TriplanarDecoder`'s
  `sdf_*` prefix stays — its SDF head is one of two networks and the prefix says which.
  Renaming a constructor parameter is Breaking and rides the v0.4.0 boundary (§5).
- **The constructor signature is the schema.** With `**kwargs` gone from `Decoder` and
  `TriplanarDecoder` (§8.0.S item 4 — the same release), an unknown key in `model` is a
  `TypeError` from Python, naming the key. No refusal code to write or maintain.

What the maintainer is being asked to rule on: the five section names; the `_recon`
suffix removal; `latent_size` over `latent_dim`; and whether the optimizer sub-nests now.

## 3. Permanent and transitional

| | what | delete when |
|---|---|---|
| **permanent** | the five sections; `NSM.models.MODEL_REGISTRY` (`{"triplanar": TriplanarDecoder, "deepsdf": Decoder, "mpa": ImplicitDecoder}`) and `load_model` building from `config["model"]`; the harmonized constructor vocabularies; `NSM.configs.read(path)` returning a sectioned config; `save_model_params` writing sectioned with a `"config_layout": 2` marker | — |
| **transitional** | `NSM/_config_layout_migration.py`: `to_sections(flat) -> (sectioned, notes)`, the inverse for consumers still reading flat, and the four `_get_*_params` translators demoted to its legacy-reader half | when no `model_params_config.json` in use predates the marker — realistically at 1.0.0, per `_config_migration.py`'s own header |
| **transitional** | the paper's-vocabulary reader for MPA (`modulated` → `modulation`, `latent_size` + `layer_dimensions` → the harmonized constructor) | same file, same condition |

Size the permanent part justifies: the registry and `read` are **~80 lines**; the four
translators they replace are **~250**, so the permanent code is net negative in
`loader.py`. The migration file is budgeted at **~150** and is not counted, being deletable
by condition. Constructor harmonization is renames, net ~0. Growth past **+120 permanent**
is scope creep.

## 4. The consumer contract, stated once

kneepipeline's `steps/run_nsm.py` reads 29 flat keys from `model_params_config.json` by
name — 15 model keys through its own triplanar translator, 13 `*_recon` keys, and
`batch_size_latent_recon`, which NSM deprecated. nsosim reads the same file by hand and is
pinned at `nsm@b7cfd49`, so it is unaffected until it bumps. Two consequences:

- `NSM.configs.read` must accept both layouts for as long as the migration file exists, and
  every NSM entry point that takes a config (`load_model`, `train_deep_sdf`,
  `get_mean_errors`, `reconstruct_mesh`'s callers) normalizes at its top, once.
- kneepipeline's `_load_nsm_model` switches to `load_model(...)` and deletes its fifth
  translator. That is the consumer's change and its own PR; until it lands, kneepipeline
  reads archived flat files exactly as today, and new sectioned files through `read`. The
  §7.5a harness (archived jobs, production env, bone-only BScore ≤ 1e-4 band) is what says
  the switch moved nothing.

## 5. Sequence

**Step 0 — reconcile the two open PRs (before anything here).** PR #107 (§8.0.R) merges
first; it is complete and green. PR #108 then rebases: its first commit (delete
`ImplicitDecoder`) is withdrawn by the ruling; its second (delete `two_stage`) survives and
has to reconcile with #107 — delete `TestTwoStageTranslatesWhatItsSiblingsRead`,
`TWO_STAGE_DROPPED` and the two-stage evidence test from `test_parameter_surface.py`, note
in `KNOWN_ISSUES` § History 30 that the model type it describes was removed in the same
release, and correct its own SCOPE §1 line — `deepsdf` reconstructs, measured. `Sine` stays
where it is; the re-homing was part of the withdrawn commit.

**Step 1 — this statement**, and the ruling on §2.

**Step 2 — MPA is buildable from the paper's config, before any layout work.** Small and
independent: `_get_implicit_params` reads the paper's vocabulary (`latent_size`,
`layer_dimensions`, `modulated`, `objects_per_decoder`, `final_activation`, `activation`),
defaults the output to `tanh`, and a test builds the paper's architecture from one of the
17 saved MPA configs and asserts it — 8 layers × 512, modulation on, sine synthesizer, ReLU
modulator, tanh, two outputs. Needs the config from the server. `SCOPE` §2.6 then says
"supported, as the paper's MPA baseline" instead of "unreachable". This is the
reproducibility fix and it does not wait on the layout.

**Step 3 — the layout, at the v0.4.0 boundary.** Constructor harmonization is Breaking, so
it belongs with the boundary §8.0.S already owns (signatures, `**kwargs` refusal,
`max_batch_size`, the `verbose` bridge). **Recommendation: this initiative absorbs S** — one
migration boundary instead of two, and S's item 4 (refuse unknown kwargs) is what makes the
constructor the schema. Commits, one concern each: the migration module and `read`; the
registry and `load_model` on `config["model"]`; the three constructor renames with the
paper's two configs and the three HuggingFace configs as the round-trip fixtures;
`save_model_params` writing sectioned; the generator and shipped config re-cut into
sections; the translators demoted; docs and CHANGELOG.

**Step 4 — kneepipeline's PR**, after v0.4.0 is on `main` and pulled into the working tree.

## 6. Verification per claim

| Claim | Verification |
|---|---|
| the three shipped models still load | `test_shipped_checkpoints` (regression harness) builds each of `231`/`551`/`647` from its flat `model_params_config.json` via `read` → registry, and `load_state_dict(strict=True)` passes — the same checkpoints, bit-identical outputs on the harness's fixed latent |
| the paper's MPA is buildable | the saved server config, through `load_model(model_type="mpa")`, yields a module asserted to have 8 blocks of width 512, a `ModulationNetwork`, `SirenLinear` blocks, `torch.tanh` as `final_activation`, `out_dim == 2` |
| the paper's MLP is buildable | same, for the implicit-MLP config: `Decoder` with two 8×512 heads and `latent_in` at layer 4 — the values the paper states; anything it does not state is read from the saved config, not guessed |
| deepsdf and MPA reconstruct | a tiny instance of each through `reconstruct_latent` on CPU, three iterations, two surfaces — the measurement of 2026-09-04, pinned, and the correction to #108's SCOPE line |
| no translator drift is possible | there is no translator to drift: a test asserts `loader.py` defines no `_get_*_params` outside the migration module |
| every sectioned key is read by its section's owner | §8.0.R's literal sweep, run per section: each key in `model` names a constructor parameter, each in `train` appears as a literal in `train/` or `utils.py`, each in `recon` in `reconstruct/` |
| a typo in `model` is refused | `read` of a config with `"paddding"` raises `TypeError` naming it, from the constructor, with no NSM code in the traceback's last frame |
| the flat→sectioned migration is total | every key of the flat shipped config lands in exactly one section, and `to_flat(to_sections(flat)) == flat` |
| kneepipeline moved nothing | §7.5a's five archived jobs, production env, both BScore variants within the measured 1e-4 run-to-run band |
| the suite still passes | the count on `main` at step 3's base commit is the baseline; every commit compared against it |

## 7. Not this plan

- **`grad_clip` reaching the latents** — needs a training experiment (`KNOWN_ISSUES` § Open).
- **A default config per model type** (`NSM_CODE_HEALTH_REFACTOR.md` §8.1) — becomes
  trivial once `model` is a section (a `model` block per type, the rest shared), and is
  done in step 3 as a side effect rather than as its own item.
- **The dataset being built from the config.** `dataset` stays a specification the user
  translates; making `MultiSurfaceSDFSamples(**config["dataset"])` work is the obvious
  follow-on and is a feature, not part of this.
