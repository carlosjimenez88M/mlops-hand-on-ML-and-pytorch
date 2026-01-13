.PHONY: help install install-uv sync test test-cov run-pipeline run-download run-preprocessing lint format clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install-uv      - Install uv package manager"
	@echo "  make install         - Install all dependencies with uv"
	@echo "  make sync            - Sync dependencies with uv"
	@echo "  make test            - Run unit tests"
	@echo "  make test-cov        - Run tests with coverage report"
	@echo "  make run-pipeline    - Run complete MLOps pipeline"
	@echo "  make run-download    - Run only download step"
	@echo "  make run-preprocessing - Run only preprocessing step"
	@echo "  make lint            - Run code linters"
	@echo "  make format          - Format code"
	@echo "  make clean           - Clean cache and temp files"

# Install uv if not already installed
install-uv:
	@echo "Installing uv package manager..."
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
install: install-uv
	@echo "Creating virtual environment and installing dependencies with uv..."
	@test -d .venv || uv venv .venv
	@uv pip install -e .

# Sync dependencies (faster than install)
sync: install-uv
	@echo "Syncing dependencies with uv..."
	uv pip sync

# Run unit tests
test:
	@echo "Running unit tests..."
	@.venv/bin/pytest tests/ -v

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	@.venv/bin/pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run complete pipeline
run-pipeline: install
	@echo "Running complete MLOps pipeline..."
	@.venv/bin/python main.py

# Run only download step
run-download: install
	@echo "Running download step..."
	@.venv/bin/python main.py main.execute_steps=[01_download_data]

# Run only preprocessing step
run-preprocessing: install
	@echo "Running preprocessing step..."
	@.venv/bin/python main.py main.execute_steps=[02_preprocessing_and_imputation]

# Lint code
lint:
	@echo "Running linters..."
	@.venv/bin/ruff check src/ tests/ || echo "Ruff not installed, skipping..."

# Format code
format:
	@echo "Formatting code..."
	@.venv/bin/ruff format src/ tests/ || echo "Ruff not installed, skipping..."

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	@echo "Clean complete!"

# Test configuration
test-config:
	@echo "Testing pipeline configuration..."
	@.venv/bin/python test_pipeline.py
