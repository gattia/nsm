# Adam + L-BFGS Hybrid Latent Optimizer — Findings Report

## Purpose of this document

Hand-off summary of the Adam+L-BFGS hybrid optimization parameter sweep run in late August 2025. Another agent picking this up cold should be able to:
1. Find where the optimizer code lives in this repo.
2. Find where the sweep experiments and configs live.
3. Use the parameters that were identified as "best" without re-running the sweep.
4. Understand what was explored *after* the chosen config but never concluded.

## 1. Where the code lives (this repo)

The hybrid Adam→L-BFGS path was added in commit `01c3fda Implement hybrid optimizer for latent reconstruction`.

**File:** `NSM/reconstruct/main.py`

Key spans in `reconstruct_latent(...)`:
- `main.py:252` — `project_latent(latent, latent_norm)` — **hard** hypersphere projection (legacy).
- `main.py:269` — `latent_norm_penalty(latent, target_norm, penalty_weight, penalty_type)` — **soft** norm penalty (quadratic / huber / barrier).
- `main.py:370–445` — function signature exposes:
  - `hybrid_optimizer=False`
  - `adam_iterations`, `lbfgs_iterations`
  - `lbfgs_lr=1.0`, `lbfgs_max_iter=20`, `lbfgs_history_size=100`
  - `use_soft_norm_constraint=True` (default), `norm_penalty_weight=1e-3`, `norm_penalty_type="quadratic"`
- `main.py:488–494` — Adam→LBFGS switchover logic.
- `main.py:686–696` — total_loss assembly: `recon_loss + latent_loss + eikonal_weight*eikonal_loss + norm_penalty_loss`.
- `main.py:699–711` — LBFGS closure (with optional in-closure hard projection when `use_soft_norm_constraint=False`).
- `main.py:730–734` — post-Adam-step hard projection (also gated by `use_soft_norm_constraint=False`).

## 2. Where the experiments live (separate dir)

**Directory:** `/dataNAS/people/aagatti/projects/nsm_recon_optimization_testing/`

This is a separate working directory, **not** part of the NSM repo. It contains:
- Many `test_lbfgs_optimizer_*.json` and `test_hybrid_*.json` configs.
- `baseline_config*.json` — Adam-only baselines.
- `run_baseline_benchmark.py`, `test_optimization/test_optimization.py` — runners.
- Markdown notes: `README.md`, `CURRENT_STATUS.md`, `BASELINE_CORRECTIONS.md`, `PARAMETER_MAPPING_CORRECTIONS.md`, `baseline_reconstruction_parameters.md`, `VALIDATION_TESTING_GUIDE.md`.
- `wandb/` directory with run data (project: `bone-modeling/nsm-optimization-testing`).

The validation set used: **139 femur cases** (bone + cart + med_men + lat_men) from OAI DESS, with a 14-case subset for fast iteration. The model under test is the femur model at `comak_gait_simulation/COMAK_SIMULATION_REQUIREMENTS/nsm_models/568_nsm_femur_bone_cart_men_v0.0.1/` (latent_size=1024, objects_per_decoder=4).

## 3. Chosen "best" config

The README explicitly designates **`test_hybrid_norm_10_3_full_dataset.json`** as the advanced hybrid config and runs it against all 139 validation cases as the alternative to the Adam baseline.

### Hybrid optimizer parameters

| Param | Value | Note |
|---|---|---|
| `hybrid_optimizer` | `true` | turns on Adam→LBFGS switching |
| `latent_optimizer_name` | `"lbfgs"` | optimizer kind (hybrid still expected) |
| `adam_iterations` | `10` | short warm-up |
| `lbfgs_iterations` | `50` | bulk of the work happens here |
| `lbfgs_max_iter` | `10` | inner iters per LBFGS step |
| `lbfgs_history_size` | `50` | |
| `lbfgs_lr` | `1.0` | |
| `lr` (Adam warm-up) | `0.01` | |
| `n_lr_updates` | `1` | |
| `lr_update_factor` | `1.1` | |
| `num_iterations` | `2000` | upper cap (early-stopping ends sooner) |
| `convergence` | `"recon_loss"` | |
| `convergence_patience` | `5` | tight |
| `min_rel_improve` | `1e-3` | |
| `grad_tol` | `1e-5` | |
| `param_change_tol` | `1e-3` | |
| `latent_norm` | `10.0` | target norm |
| `norm_penalty_weight` | `100` | penalty coefficient |
| `n_samples_latent_recon` | `1,000,000` | dense surface sampling |
| `batch_size` | `1,000,000` | |
| `loss_type` | `"l1"` | |
| `clamp_dist` | `0.1` | |
| `latent_init_std` / `latent_init_mean` | `0.01` / `0.0` | |
| `objects_per_decoder` | `4` | model-specific |

### ⚠️ Important caveat: norm is *soft*, not a hypersphere

The 10_3 config sets `latent_norm: 10.0` and `norm_penalty_weight: 100` but **does not** set `use_soft_norm_constraint`. The default in `reconstruct_latent(...)` is `True`, so the sweep used the **soft quadratic penalty** path (`latent_norm_penalty`), never the **hard hypersphere projection** (`project_latent`).

The hard-projection path exists in the code but was not exercised in this sweep. To test it, add `"use_soft_norm_constraint": false` to a copy of the config.

### Baseline it was compared against

`baseline_config_full_dataset.json` (Adam, 139 cases):
- `latent_optimizer_name: "adam"`, `lr: 0.005`, `num_iterations: 2000`
- `n_samples_latent_recon: 20000`, `batch_size: 300000`
- `n_lr_updates: 100`, `lr_update_factor: 1.1`, `convergence_patience: 50`
- `loss_type: "l1"`, `clamp_dist: 0.1`, `l2reg: false`, no norm constraint

These are the NSM model-config defaults derived from `568_nsm_femur_bone_cart_men_v0.0.1/model_params_config.json`, with the optimizer name corrected from training's `"AdamW"` to the NSM-supported `"adam"` (see `PARAMETER_MAPPING_CORRECTIONS.md`).

## 4. How they got there (sweep progression)

Earlier exploration progressively narrowed in on 10_3. All ran on 1 test case unless noted.

| Config | adam_iters | lbfgs_iters | lbfgs_max_iter | conv_patience | n_samples | batch | loss | Notes |
|---|---|---|---|---|---|---|---|---|
| `test_hybrid_norm_10_1.json` | 50 | 100 | 10 | 10 | 1M | 300k | l1 | First hybrid attempt — too long |
| `test_hybrid_norm_10_2.json` | 200 | 20 | 10 | 5 | 1M | 1M | l1 | Adam-heavy |
| **`test_hybrid_norm_10_3.json`** | **10** | **50** | **10** | **5** | **1M** | **1M** | **l1** | **Chosen recipe** |
| `test_hybrid_norm_10_3_full_dataset.json` | (same as 10_3) | | | | | | | Promoted to 139-case run |
| `test_hybrid_norm_10_3_14_examples.json` | (same as 10_3) | | | | | | | 14-case validation subset |

Pre-hybrid exploration (LBFGS only) lives in `test_lbfgs_optimizer_*.json` (about a dozen variants — `_latent_reg_*`, `_norm_10_config_all_points_*`, ending at `_v5`). These were superseded by the hybrid recipe.

## 5. Post-10_3 exploration — NOT concluded

After 10_3 was promoted to full-dataset, three L2-loss variants were created on Aug.31 but never promoted to full-dataset runs. **No memo declares these "best" — treat as ongoing exploration:**

- `test_hybrid_norm_10_3_l2.json` — `loss_type:"l2"`, `adam_iterations:50`, `lbfgs_iterations:100`, `lbfgs_max_iter:5`, `param_change_tol:0.01`
- `test_hybrid_norm_10_3_l2_1_maxiter.json` — `loss_type:"l2"`, `adam:10`, `lbfgs:100`, `lbfgs_max_iter:1`
- `test_hybrid_norm_10_3_l2_2_matix_small_batch.json` — `loss_type:"l2"`, `adam:10`, `lbfgs:100`, `lbfgs_max_iter:5`, `n_samples:20k`, `grad_tol:1e-6`, `min_rel_improve:1e-4`

Also: `baseline_config_l2.json` and `baseline_config_l2_big_batch.json` — Adam-only L2 baselines.

**Open question for future work:** does L2 beat L1 once tuned, and if so do the L-BFGS iteration shapes need to change? Single-case results may exist in `wandb/` but were not summarized.

## 6. Metrics tracked

Per `VALIDATION_TESTING_GUIDE.md`, runs log to wandb project `bone-modeling/nsm-optimization-testing` with tissue-specific ASSD:
- `mean_assd_bone`, `mean_assd_cartilage`, `mean_assd_med_meniscus`, `mean_assd_lat_meniscus`
- `mean_assd` (overall), `mean_reconstruction_time`
- `std_assd_*`, `median_assd_*`, `n_samples_assd_*`

Numeric ranking of configs is not written down in the markdown notes — would need to be pulled from wandb.

## 7. How to run

From `nsm_recon_optimization_testing/README.md`:

```bash
conda activate comak

# Baseline (Adam, 139 cases)
python test_optimization/test_optimization.py baseline_config_full_dataset.json

# Best hybrid (Adam + L-BFGS, 139 cases)
python test_optimization/test_optimization.py test_hybrid_norm_10_3_full_dataset.json
```

## 8. Action items for the next agent

1. **If you need a working production recipe today:** use `test_hybrid_norm_10_3_full_dataset.json` parameters as-is. They were the explicit recommendation as of Aug.29.2025.
2. **If you want to verify "best":** pull tissue-specific ASSD from wandb for the 10_3 full-dataset run vs the baseline full-dataset run. Numbers were not committed to a markdown summary.
3. **If you want to push further:**
   - Test the **hard hypersphere projection** path (`use_soft_norm_constraint: false`) — it was never tried.
   - Resolve the **L2 vs L1** question — three L2 hybrid variants exist but were never promoted to full-dataset.
   - Try varying `norm_penalty_weight` (fixed at 100 throughout) and `norm_penalty_type` (only `"quadratic"` tested; `"huber"` and `"barrier"` are implemented).
4. **Known gotcha:** the model config has `"optimizer": "AdamW"` for training, but NSM's `reconstruct_latent` expects `"adam"` (lowercase). See `PARAMETER_MAPPING_CORRECTIONS.md` and `BASELINE_CORRECTIONS.md`.

## 9. Key file references

Code (this repo):
- `NSM/reconstruct/main.py:252` — `project_latent`
- `NSM/reconstruct/main.py:269` — `latent_norm_penalty`
- `NSM/reconstruct/main.py:370` — `reconstruct_latent` signature with hybrid params
- `NSM/reconstruct/main.py:488` — Adam→LBFGS switchover
- `NSM/reconstruct/main.py:699` — LBFGS closure

Experiments (separate dir, `/dataNAS/people/aagatti/projects/nsm_recon_optimization_testing/`):
- `README.md`
- `test_hybrid_norm_10_3_full_dataset.json` — **chosen recipe, 139 cases**
- `baseline_config_full_dataset.json` — Adam baseline, 139 cases
- `baseline_reconstruction_parameters.md` — derivation from model config
- `PARAMETER_MAPPING_CORRECTIONS.md` — `AdamW` vs `adam` gotcha
- `VALIDATION_TESTING_GUIDE.md` — metric definitions and runner usage
