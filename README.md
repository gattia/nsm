![Build Status](https://github.com/gattia/NSM/actions/workflows/build-test.yml/badge.svg?branch=main)<br>
|[Documentation](http://anthonygattiphd.com/NSM/)|



# Introduction

This pacakge is meant to develop generative deep learning models for creating human anatomy. The initial focus is on musculoskeletal tissues, particular of the knee. 

# Installation

## Standard Installation

```bash
# Create and activate conda environment
conda create -n nsm python=3.9
conda activate nsm

# Install PyTorch (ensure compatibility with your CUDA version if using GPU)
# See: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install NSM package
pip install -r requirements.txt
pip install .
```

## Development Installation
If you plan to contribute to the development of NSM, install it in editable mode. This means changes you make to the source code will be immediately reflected when you use the package.

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development setup instructions.

Quick setup:
```bash
# Clone the repository
git clone https://github.com/gattia/NSM
cd NSM

# Create and activate conda environment
conda create -n nsm-dev python=3.9
conda activate nsm-dev

# Install all dependencies and NSM in development mode
make install-dev
```

# Usage

## Logging

NSM uses Python's built-in `logging` module for debugging and monitoring. To enable logging output in your scripts:

```python
import logging

# Basic logging setup (adjust level as needed)
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# For more detailed output from specific modules
logging.getLogger('NSM.reconstruct.recon_evaluation').setLevel(logging.DEBUG)
```

### Logging Levels
- **DEBUG**: Detailed information for debugging (mesh processing details, loss values)
- **INFO**: General information about process flow  
- **WARNING**: Potential issues (missing meshes, NaN values)
- **ERROR**: Serious errors that may cause failures

### Example with File Output
```python
import logging

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nsm_processing.log'),
        logging.StreamHandler()  # Console output
    ]
)
```

## Model Loading

NSM provides a convenient model loader that simplifies loading pre-trained Neural Shape Models. For **real trained models**, you'll typically have:

- `experiment_dir/model_params_config.json` - Configuration saved during training
- `experiment_dir/model/2000.pth` - Model weights at epoch 2000

```python
import json
from NSM.models import load_model

# Load configuration from training
with open('experiment_dir/model_params_config.json', 'r') as f:
    config = json.load(f)

# Load trained model
model = load_model(
    config=config,
    path_model_state='experiment_dir/model/2000.pth',
    model_type='triplanar'  # or 'deepsdf'
)

# Ready for inference!
model.eval()
```

### Supported Model Types

- `'triplanar'` - TriplanarDecoder for triplanar neural representations
- `'deepsdf'` - Standard DeepSDF decoder

### Configuration Templates

Get template configurations with sensible defaults:

```python
from NSM.models import get_model_config_template, list_supported_models

# See all supported model types
print(list_supported_models())
# ['triplanar', 'deepsdf']

# Get configuration template for any model type
config = get_model_config_template('deepsdf')
# Modify parameters as needed
config['latent_size'] = 512
config['layer_dimensions'] = [512, 512, 512, 256, 128]
```

## Examples

### Loading a Trained Model

See [`examples/load_trained_model.py`](examples/load_trained_model.py) for a complete example:

```bash
# Run the example with your trained model
python examples/load_trained_model.py /path/to/experiment_dir 2000 --model-type triplanar

# See all options
python examples/load_trained_model.py --help
```

# Development / Contributing

## Quick Development Commands

The project includes a Makefile for common development tasks:

```bash
# Run all tests
make test

# Run only model loader tests
make test-loader

# Run tests with coverage report
make test-coverage

# Format code with black
make format

# Check code style with flake8
make lint

# Clean up temporary files
make clean
```

## Tests
Run tests with pytest:

```bash
# Run all tests
pytest

# Run specific test modules
pytest testing/NSM/models/                     # Model loader tests
pytest testing/NSM/datasets/                   # Dataset tests (including multi-surface registration)

# Run tests with verbose output
pytest -v

# Use Makefile shortcuts
make test                                       # Run all tests
make test-loader                               # Run only model loader tests
```

## Coverage
Generate test coverage reports:
```bash
make test-coverage              # HTML + terminal report
```

## Contributing
If you want to contribute, please read the documentation in `CONTRIBUTING.md` and see `DEVELOPMENT.md` for detailed development setup instructions.

## Documentation

Additional documentation can be found in the [`docs/`](docs/) folder:

- [`docs/SCOPE.md`](docs/SCOPE.md) - What NSM supports, module status rulings, the public API contract
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Structural map: invariants, traps, dependency graph
- [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) - **If you have training runs on disk, read this** — which past bugs affect which results, with date ranges
- [`docs/MULTI_SURFACE_REGISTRATION.md`](docs/MULTI_SURFACE_REGISTRATION.md) - Multi-surface registration functionality

An API reference builds locally with `make docs` (pdoc, rendered into `site/`). The
publishing workflow (`.github/workflows/docs.yml`) is written but deliberately not wired
to `push` yet: it publishes when the Phase 2 docstring-accuracy pass lands, so the first
published reference is not wrong on arrival. Until then it can be run by hand from the
Actions tab.

## TODO

- [ ] **Add logging throughout the codebase**: Extend logging support to all major modules (training, model loading, mesh processing, etc.) following the pattern established in `NSM.reconstruct.recon_evaluation`


# License

This project is licensed under the terms of the license specified in the [LICENSE](LICENSE) file.