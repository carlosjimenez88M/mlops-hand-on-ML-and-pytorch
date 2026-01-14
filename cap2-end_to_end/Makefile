.PHONY: help install install-uv sync run-tests test-cov run-pipeline test-config lint format clean docker-build docker-run api-install api-local api-test api-test-cov api-build streamlit-install streamlit-local streamlit-test streamlit-build compose-up compose-down compose-logs

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
	@echo "  make run-sweep       - Run W&B Sweep for hyperparameter optimization"
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
	@echo "🎨 Streamlit Commands:"
	@echo "  make streamlit-install  - Install Streamlit dependencies"
	@echo "  make streamlit-local    - Run Streamlit locally (port 8501)"
	@echo "  make streamlit-build    - Build Streamlit Docker image"
	@echo "  make run-full-stack     - Run API + Streamlit locally"
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

# Run W&B Sweep for hyperparameter optimization
run-sweep: install
	@echo "Running W&B Sweep for hyperparameter optimization..."
	@echo "This will test multiple combinations of n_clusters and gamma"
	@.venv/bin/python src/data/03_feature_engineering/run_sweep.py --count 20

# Run single sweep agent (for parallel execution)
run-sweep-agent: install
	@echo "Running sweep agent (use SWEEP_ID environment variable)..."
	@.venv/bin/python -c "import wandb; wandb.agent('$$SWEEP_ID', function=lambda: __import__('src.data.03_feature_engineering.run_sweep').train_with_sweep())"

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
	@echo ""
	@echo "==================================================================="
	@echo "  Services are now running!"
	@echo "==================================================================="
	@echo ""
	@echo "🌐 API:        http://localhost:8080"
	@echo "   - Docs:     http://localhost:8080/docs"
	@echo "   - Health:   curl http://localhost:8080/health"
	@echo ""
	@echo "🎨 Streamlit:  http://localhost:8501"
	@echo ""
	@echo "To view logs: make compose-logs"
	@echo "To stop:      make compose-down"
	@echo "==================================================================="

# Stop docker-compose services
compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

# View logs from all services
compose-logs:
	@echo "Showing logs from all services (Ctrl+C to exit)..."
	docker-compose logs -f

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

# ===================================================================
# Streamlit Commands
# ===================================================================

# Install Streamlit dependencies
streamlit-install:
	@echo "Installing Streamlit dependencies..."
	cd streamlit_app && pip install -r requirements.txt

# Run Streamlit locally
streamlit-local: streamlit-install
	@echo "Starting Streamlit locally on port 8501..."
	@echo "Streamlit will be available at http://localhost:8501"
	@echo "Make sure the API is running at http://localhost:8080"
	cd streamlit_app && API_URL=http://localhost:8080 streamlit run app.py --server.port 8501

# Build Streamlit Docker image
streamlit-build:
	@echo "Building Streamlit Docker image..."
	cd streamlit_app && docker build -t housing-streamlit:latest .

# Run full stack locally (API + Streamlit in parallel)
run-full-stack:
	@echo "Starting full stack (API + Streamlit)..."
	@echo "This will run both services in parallel"
	@echo ""
	@echo "API will be available at:       http://localhost:8080"
	@echo "Streamlit will be available at: http://localhost:8501"
	@echo ""
	@echo "Press Ctrl+C to stop both services"
	@echo ""
	@make -j2 api-local streamlit-local
