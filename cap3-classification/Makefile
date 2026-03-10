.PHONY: install sync lint format test test-cov api-test api-test-cov run-pipeline run-competition run-sweep register monitor-drift api-local docker-api-build docker-api-test docker-api-run smoke

UV ?= uv
PROJECT_DIR := $(CURDIR)
export UV_CACHE_DIR ?= $(PROJECT_DIR)/.uv-cache

help:
	@echo "cap3-classification"
	@echo "  make install           Install dependencies with uv"
	@echo "  make test              Run unit tests"
	@echo "  make test-cov          Run tests with coverage"
	@echo "  make run-pipeline      Run the full pipeline through mlflow run ."
	@echo "  make run-competition   Run through model competition only"
	@echo "  make run-sweep         Run model competition and sweep"
	@echo "  make register          Run sweep and registration"
	@echo "  make monitor-drift     Generate drift report"
	@echo "  make api-local         Serve the registered model with FastAPI"
	@echo "  make docker-api-build  Build the API image"
	@echo "  make docker-api-test   Run tests and coverage inside Docker"
	@echo "  make docker-api-run    Run the API container"

install:
	$(UV) sync --extra dev

sync:
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check src tests api

format:
	$(UV) run ruff format src tests api

test:
	$(UV) run pytest tests api/tests

test-cov:
	$(UV) run pytest tests api/tests --cov=src/cap3_classification --cov=api/app --cov-report=term-missing

run-pipeline:
	$(UV) run mlflow run . --env-manager local

run-competition:
	$(UV) run mlflow run . --env-manager local -P pipeline_overrides='main.execute_steps=["01_ingest","02_validate","03_feature_engineering","04_split","05_model_competition"]'

run-sweep:
	$(UV) run mlflow run . --env-manager local -P pipeline_overrides='main.execute_steps=["01_ingest","02_validate","03_feature_engineering","04_split","05_model_competition","06_sweep"]'

register:
	$(UV) run mlflow run . --env-manager local -P pipeline_overrides='main.execute_steps=["01_ingest","02_validate","03_feature_engineering","04_split","05_model_competition","06_sweep","07_register"]'

monitor-drift:
	$(UV) run python scripts/monitor_drift.py

api-local:
	$(UV) run uvicorn api.app.main:app --host 0.0.0.0 --port 8080

docker-api-build:
	docker build -f api/Dockerfile -t cap3-classification-api:latest .

docker-api-test:
	docker build -f api/Dockerfile --target test -t cap3-classification-api:test .
	docker run --rm cap3-classification-api:test

docker-api-run:
	docker run --rm -p 8080:8080 \
		-e MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
		-e MLFLOW_MODEL_NAME=chapter3_digit_classifier \
		-e MLFLOW_MODEL_ALIAS=staging \
		cap3-classification-api:latest

smoke:
	$(UV) run mlflow run . --env-manager local -P pipeline_overrides='main.execute_steps=["01_ingest","02_validate","03_feature_engineering","04_split","05_model_competition","06_sweep","07_register"] dataset.fetch_strategy=fallback_only dataset.primary_source=digits wandb.mode=disabled sweep.offline_runs=4 competition.candidate_models=["dummy_most_frequent","sgd_classifier","random_forest","torch_mlp"]'
