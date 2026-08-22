# NSM Development Makefile
# Simplifies common development tasks

.PHONY: help install install-dev test test-loader test-coverage lint autoformat docs clean env-setup quick-test
# docs/ and build/ are real directories, so without .PHONY make would treat
# `make docs` as an up-to-date file target and do nothing.

# Default target
help:
	@echo "NSM Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install NSM package in current environment"
	@echo "  install-dev      Install NSM in development mode with dev dependencies"
	@echo "  test             Run all tests with pytest"
	@echo "  test-loader      Run only model loader tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  lint             Check formatting and lint (isort, black, flake8)"
	@echo "  autoformat       Apply isort and black"
	@echo "  docs             Build the API reference into site/"
	@echo "  clean            Clean up temporary files and caches"
	@echo "  env-setup        Setup conda development environment"
	@echo "  quick-test       Quick dev cycle: autoformat + loader tests"

# Installation targets
install:
	pip install .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

# Testing targets
test:
	pytest testing/ -v

test-loader:
	pytest testing/NSM/models/test_loader.py -v

test-coverage:
	pytest testing/ --cov=NSM --cov-report=html --cov-report=term-missing

# TODO: Add parallel testing once pytest-xdist is fully configured
# test-parallel:
#	pytest testing/ -n auto -v

# Code quality targets -- same names as gattia/pymskt, so `make lint` and
# `make autoformat` mean the same thing in both repos.
lint:
	set -e
	isort -c NSM/ testing/
	black --check --config pyproject.toml NSM/ testing/
	flake8 NSM/ testing/

autoformat:
	set -e
	isort NSM/ testing/
	black --config pyproject.toml NSM/ testing/

# Cleanup targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Environment setup
env-setup:
	conda create -n nsm-dev python=3.9 -y
	@echo "Environment created. Activate it with: conda activate nsm-dev"
	@echo "Then run: make install-dev"

# Documentation. pdoc renders an API reference from docstrings into site/.
#
# site/, NOT docs/: docs/ holds hand-written engineering documents (SCOPE,
# ARCHITECTURE, KNOWN_ISSUES) that pdoc would overwrite. pymskt writes into its
# docs/ because that directory is generated output; ours is not.
#
# (PDOC_ALLOW_EXEC=1 was needed while the vendored NSM.dependencies pulled in
# pykeops, which compiles a probe binary at import; both left with PR #64 and
# the build was re-verified without it.)
docs:
	pdoc -o site/ NSM

# TODO: CI/CD targets - set these up once CI is configured
# ci-test: install-dev test-coverage lint
#	@echo "CI tests completed!"

# Quick development helpers
quick-test: autoformat test-loader
	@echo "Quick development cycle completed!"