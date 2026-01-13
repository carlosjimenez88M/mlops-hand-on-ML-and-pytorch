.PHONY: help install install-uv sync run-tests test-cov run-pipeline test-config lint format clean docker-build docker-run api-install api-local api-test api-test-cov api-build compose-up compose-down

# Default target
help:
	@echo "==================================================================="
	@echo "  Housing Price Prediction - MLOps Pipeline"
	@echo "==================================================================="
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make install         - Install all dependencies with uv"
	@echo "  make install-uv      - Install uv package manager"
	@echo ""
	@echo "🚀 Pipeline Execution:"
	@echo "  make run-pipeline    - Run complete MLOps pipeline (Steps 1-5)"
	@echo "  make test-config     - Test pipeline configuration"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make run-tests       - Run unit tests"
	@echo "  make test-cov        - Run tests with coverage report"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make docker-build    - Build pipeline Docker image"
	@echo "  make docker-run      - Run pipeline in Docker container"
	@echo "  make compose-up      - Start API with docker-compose"
	@echo "  make compose-down    - Stop docker-compose services"
	@echo ""
	@echo "🌐 API Commands:"
	@echo "  make api-install     - Install API dependencies"
	@echo "  make api-local       - Run API locally (port 8080)"
	@echo "  make api-test        - Run API tests"
	@echo "  make api-test-cov    - Run API tests with coverage"
	@echo "  make api-build       - Build API Docker image"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  make lint            - Run code linters"
	@echo "  make format          - Format code with ruff"
	@echo "  make clean           - Clean cache and temp files"
	@echo ""
	@echo "==================================================================="

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
run-tests:
	@echo "Running unit tests..."
	@.venv/bin/pytest tests/ -v

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	@.venv/bin/pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

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

# ===================================================================
# Docker Commands
# ===================================================================

# Build pipeline Docker image
docker-build:
	@echo "Building pipeline Docker image..."
	docker build -t housing-pipeline:latest .

# Run pipeline in Docker
docker-run:
	@echo "Running pipeline in Docker..."
	docker run --env-file .env housing-pipeline:latest

# Build API Docker image
api-build:
	@echo "Building API Docker image..."
	cd api && docker build -t housing-api:latest .

# Start services with docker-compose
compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up -d
	@echo "API available at http://localhost:8080"
	@echo "Health check: curl http://localhost:8080/health"

# Stop docker-compose services
compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

# ===================================================================
# API Commands
# ===================================================================

# Install API dependencies
api-install:
	@echo "Installing API dependencies..."
	cd api && pip install -r requirements.txt

# Run API locally
api-local: api-install
	@echo "Starting API locally on port 8080..."
	@echo "API will be available at http://localhost:8080"
	@echo "API docs at http://localhost:8080/docs"
	cd api && uvicorn app.main:app --reload --port 8080

# Test API
api-test: api-install
	@echo "Running API tests..."
	cd api && pytest tests/ -v

# Test API with coverage
api-test-cov: api-install
	@echo "Running API tests with coverage..."
	cd api && pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
