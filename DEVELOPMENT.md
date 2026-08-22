# NSM Development Environment Setup

This document provides instructions for setting up a development environment for the Neural Shape Models (NSM) library, including testing, linting, and development dependencies.

## Quick Setup

### Option 1: Using conda (Recommended)

```bash
# Create a new conda environment
conda create -n nsm-dev python=3.9 -y
conda activate nsm-dev

# Install PyTorch (adjust for your CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install runtime + development dependencies and NSM in development mode
# (pyproject.toml declares no dependencies, so requirements.txt is NOT optional:
# skipping it leaves every subpackage except NSM.utils unimportable)
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
# ...or, equivalently, all three in one step:
make install-dev
```

### Option 2: Using pip with virtual environment

```bash
# Create virtual environment
python -m venv nsm-dev
source nsm-dev/bin/activate  # On Windows: nsm-dev\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version - adjust for GPU if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install runtime + development dependencies and NSM in development mode
# (requirements.txt is not optional; see Option 1)
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Development Dependencies

The development environment includes:

- **Testing**: `pytest`, `pytest-cov` for test coverage
- **Linting**: `flake8`, `black` for code formatting
- **Type checking**: `mypy` for static type analysis
- **Documentation**: `sphinx` for documentation generation
- **Jupyter**: `jupyter` for notebook development
- **Scientific**: `numpy`, `scipy`, `matplotlib` for data handling and visualization

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with coverage
```bash
pytest --cov=NSM --cov-report=html
```

### Run specific test files
```bash
# Test model loader
pytest testing/NSM/models/test_loader.py -v

# Test triplanar model
pytest testing/NSM/models/test_triplanar.py -v
```

### Run tests in parallel (faster)
```bash
pytest -n auto
```

## Code Quality

### Linting
```bash
# Check code style
flake8 NSM/

# Format code
black NSM/ testing/
```

### Type checking
```bash
mypy NSM/
```

## Development Workflow

1. **Before making changes**: Create a new branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **During development**: Run tests frequently
   ```bash
   pytest testing/ -v
   ```

3. **Before committing**: Run the full test suite and linting
   ```bash
   pytest
   flake8 NSM/
   black NSM/ testing/ --check
   ```

4. **Adding new features**: 
   - Add tests in the appropriate `testing/NSM/` subdirectory
   - Update documentation if needed
   - Ensure all tests pass

## Project Structure

```
NSM/
├── NSM/                    # Main package
│   ├── configs/           # default_config.json and its generator
│   ├── datasets/          # Dataset handling
│   ├── mesh/              # Marching cubes, refinement, interpolation
│   ├── models/            # Model definitions and loader
│   ├── reconstruct/       # Reconstruction utilities
│   ├── train/             # Training utilities
│   ├── losses.py          # SDF loss functions
│   └── utils.py           # Utility functions
├── testing/               # Test suite
│   └── NSM/              # Tests mirroring package structure
├── examples/              # Usage examples
├── docs/                  # Engineering docs (SCOPE, ARCHITECTURE, KNOWN_ISSUES)
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Development dependencies
└── pyproject.toml         # Package configuration
```

## Adding New Tests

When adding new functionality:

1. Create test files in `testing/NSM/` following the package structure
2. Use descriptive test class and method names
3. Include docstrings explaining what each test validates
4. Test both success and failure cases
5. Use fixtures for common test data

Example test structure:
```python
import unittest
import torch
from NSM.models import YourNewModel

class TestYourNewModel(unittest.TestCase):
    """Test suite for YourNewModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {...}
    
    def test_initialization(self):
        """Test model initialization."""
        model = YourNewModel(**self.config)
        self.assertIsInstance(model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        # Test implementation
        pass
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch sizes in tests or use CPU-only testing
2. **Import errors**: Ensure NSM is installed in development mode (`pip install -e .`)
3. **PyTorch version conflicts**: Check compatibility between PyTorch and CUDA versions

### Getting Help

- Check existing issues in the repository
- Run tests with verbose output: `pytest -v -s`
- Use debugging: `pytest --pdb` to drop into debugger on failures

## Contributing

1. Follow the development workflow above
2. Ensure all tests pass: `pytest`
3. Follow code style: `black NSM/ testing/`
4. Add appropriate tests for new features
5. Update documentation as needed

## Performance Testing

For performance-critical changes:

```bash
# Timing scripts live in testing/testing_sdf_calculation_times/ (scratch scripts,
# run directly rather than through pytest)
python testing/testing_sdf_calculation_times/time.py

# Profile code
python -m cProfile -s cumtime your_script.py
```
