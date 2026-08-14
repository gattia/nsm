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

# Format code (Black, 100 char line length)
make format

# Check formatting without modifying
make format-check

# Lint with flake8
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

Neither of the two orderings this code once depended on is positional any more:

- **Param groups** carry `name` (`latent`, `model_0`, …); `adjust_learning_rate()` maps by
  name. Group order is never meaningful.
- **Schedule entries** carry `Target`; `get_learning_rate_schedules()` maps by target and
  returns canonical `[model, latent]`. That return order is an internal calling
  convention, not something the config controls.

Background: `docs/KNOWN_ISSUES_HISTORY.md` §1 — a positional-mapping bug swapped these two
schedules on every Adam/AdamW run from May 2023 to Aug 2026.

### Dependencies

Key libraries: PyTorch (core ML), mskt (pip name for pymskt - musculoskeletal toolkit), pymeshfix, pykeops (Sinkhorn/optimal transport), einops, wandb (experiment tracking)