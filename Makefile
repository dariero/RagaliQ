.PHONY: install install-dev test lint format typecheck clean all

# Default target
all: install-dev lint typecheck test

# Install package in production mode
install:
	pip install -e .

# Install package with dev dependencies
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --cov=ragaliq --cov-report=term-missing

# Run tests without coverage (faster)
test-fast:
	pytest tests/ -v

# Run only unit tests
test-unit:
	pytest tests/unit/ -v

# Run only integration tests
test-integration:
	pytest tests/integration/ -v

# Lint code
lint:
	ruff check src/ tests/

# Format code
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Type checking
typecheck:
	mypy src/

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

# Build package
build: clean
	python -m build

# Publish to PyPI (test)
publish-test: build
	twine upload --repository testpypi dist/*

# Publish to PyPI (production)
publish: build
	twine upload dist/*

# Run example
example:
	python examples/basic_usage.py

# Generate HTML coverage report
coverage-html:
	pytest tests/ --cov=ragaliq --cov-report=html
	open htmlcov/index.html

# Help
help:
	@echo "Available targets:"
	@echo "  install      - Install package"
	@echo "  install-dev  - Install package with dev dependencies"
	@echo "  test         - Run all tests with coverage"
	@echo "  test-fast    - Run tests without coverage"
	@echo "  test-unit    - Run unit tests only"
	@echo "  lint         - Check code style"
	@echo "  format       - Format code"
	@echo "  typecheck    - Run mypy"
	@echo "  clean        - Remove build artifacts"
	@echo "  build        - Build package"
