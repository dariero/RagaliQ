.PHONY: install install-dev test lint format typecheck clean all venv

# Virtual environment
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PRE_COMMIT := $(VENV)/bin/pre-commit

# Default target
all: install-dev lint typecheck test

# Create virtual environment
venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

# Install package in production mode
install: venv
	$(PIP) install .

# Install package with dev dependencies
install-dev: venv
	$(PIP) install ".[dev]"

# Run tests
test:
	$(PYTEST) tests/ -v --cov=ragaliq --cov-report=term-missing

# Run tests without coverage (faster)
test-fast:
	$(PYTEST) tests/ -v

# Run only unit tests
test-unit:
	$(PYTEST) tests/unit/ -v

# Run only integration tests
test-integration:
	$(PYTEST) tests/integration/ -v

# Lint code
lint:
	$(RUFF) check src/ tests/

# Format code
format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

# Type checking
typecheck:
	$(MYPY) src/

# Setup pre-commit hooks
pre-commit-install:
	$(PRE_COMMIT) install

# Run pre-commit on all files
pre-commit-run:
	$(PRE_COMMIT) run --all-files

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Clean everything including venv
clean-all: clean
	rm -rf $(VENV)

# Build package
build: clean
	$(PYTHON) -m build

# Publish to PyPI (test)
publish-test: build
	twine upload --repository testpypi dist/*

# Publish to PyPI (production)
publish: build
	twine upload dist/*

# Run example
example:
	$(PYTHON) examples/basic_usage.py

# Generate HTML coverage report
coverage-html:
	$(PYTEST) tests/ --cov=ragaliq --cov-report=html
	open htmlcov/index.html

# Help
help:
	@echo "Available targets:"
	@echo "  venv         - Create virtual environment"
	@echo "  install      - Install package"
	@echo "  install-dev  - Install package with dev dependencies"
	@echo "  test         - Run all tests with coverage"
	@echo "  test-fast    - Run tests without coverage"
	@echo "  test-unit    - Run unit tests only"
	@echo "  lint         - Check code style"
	@echo "  format       - Format code"
	@echo "  typecheck    - Run mypy"
	@echo "  pre-commit-install - Setup pre-commit hooks"
	@echo "  pre-commit-run     - Run pre-commit on all files"
	@echo "  clean        - Remove build artifacts"
	@echo "  clean-all    - Remove build artifacts and venv"
	@echo "  build        - Build package"
