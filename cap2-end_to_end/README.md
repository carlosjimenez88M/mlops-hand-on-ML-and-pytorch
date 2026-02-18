# MLOps Pipeline - Housing Price Prediction

**Status**:  Production Ready | **Model Version**: 5 | **MAPE**: 20.40%

Complete end-to-end MLOps pipeline for California housing price prediction using Random Forest, with W&B Sweep optimization and MLflow Model Registry integration.

---

##  Quick Start

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/carlosjimenez88M/mlops-hand-on-ML-and-pytorch.git
cd mlops-hand-on-ML-and-pytorch/cap2-end_to_end

# Install dependencies
uv sync --extra dev --extra api --extra streamlit

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run complete pipeline
uv run python main.py

# Run through MLflow entrypoint
uv run mlflow run . --env-manager local
```

---

##  Pipeline Results

### Best Model Performance
```yaml
Algorithm: Random Forest Regressor
MAPE: 20.40%              # Mean prediction error
R²: 0.7834                 # Explains 78% of variance
RMSE: 53,277              # Root mean squared error
Within 10%: 36.2%         # Predictions within ±10%
Dataset: 20,640 samples (16,512 train / 4,128 test)
Features: 14 (8 numerical + 5 categorical + 1 engineered)
```

### Pipeline Execution
- **Total Time**: 8.6 minutes (515.92 seconds)
- **All Steps**:  Completed successfully
- **Optimization**: Bayesian (5 runs, not GridSearch)
- **Tracking**: MLflow + Weights & Biases

---

##  Architecture

### Pipeline Steps

```
01. Download Data         → Fetch from GCS bucket
02. Preprocessing         → Handle missing values, outliers
03. Feature Engineering   → Create cluster labels, encode categories
04. Data Segregation      → 80/20 train/test split
05. Model Selection       → Compare 5 algorithms
06. Hyperparameter Sweep  → W&B Bayesian optimization
07. Model Registration    → MLflow Model Registry + local save
```

### Project Structure
```
cap2-end_to_end/
 main.py                      # Pipeline orchestrator
 config.yaml                  # Central configuration
 configs/
    model_config.yaml       # Generated model metadata
 src/
    data/                   # Data processing steps (01-04)
    model/                  # Model steps (05-07)
 api/                        # FastAPI prediction service
 models/trained/             # Saved models
 tests/                      # Unit and integration tests
 .github/workflows/          # CI/CD pipelines
```

---

##  Configuration

### Environment Variables (.env)

```bash
# Google Cloud Platform
GCS_BUCKET_NAME=your-gcs-bucket
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Weights & Biases
WANDB_PROJECT=housing-mlops-gcp
WANDB_ENTITY=your-wandb-username
WANDB_API_KEY=your-wandb-api-key-here

# MLflow
MLFLOW_TRACKING_URI=./mlruns
```

** Security**: Never commit `.env` files to Git!

### Model Configuration (configs/model_config.yaml)

Auto-generated after training with:
- Model name, version, stage
- All 14 feature columns
- Complete hyperparameters
- Performance metrics
- MLflow Run ID and Model URI
- Sweep ID for traceability

---

##  Usage

### Run Complete Pipeline
```bash
# All 7 steps with 5 sweep runs
uv run python main.py

# Specific steps only
uv run python main.py main.execute_steps='["06_sweep","07_registration"]'

# More sweep runs for better optimization
uv run python main.py main.execute_steps='["06_sweep"]' sweep.sweep_count=20
```

### Use Trained Model

```python
import mlflow
import joblib

# Option 1: Load from MLflow Registry
model = mlflow.pyfunc.load_model("models:/housing_price_model/5")

# Option 2: Load from local file
model = joblib.load("src/model/07_registration/models/trained/housing_price_model.pkl")

# Make predictions
predictions = model.predict(X_new)
```

### Start API Server

```bash
cd api
uv run --extra api python main.py

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"longitude":-122.23,"latitude":37.88,"housing_median_age":41,...}'
```

---

##  Features

### Data Processing
-  Missing value imputation (median strategy)
-  Outlier detection and handling
-  Feature scaling and normalization
-  One-hot encoding for categorical variables
-  K-Means clustering for location features

### Model Training
-  5 algorithm comparison (Random Forest, Gradient Boosting, etc.)
-  Bayesian hyperparameter optimization with W&B Sweep
-  Business metrics (MAPE, Within-X%)
-  MLflow experiment tracking
-  Automatic model versioning

### MLOps Best Practices
-  Reproducible pipelines (Hydra configuration)
-  Experiment tracking (MLflow + W&B)
-  Model registry (MLflow)
-  CI/CD with GitHub Actions
-  FastAPI REST API
-  Docker containerization
-  Cloud deployment ready (GCP Cloud Run/Cloud Functions)

---

##  Hyperparameter Optimization

### W&B Sweep Configuration
```yaml
Method: Bayesian Optimization
Metric: MAPE (minimize)
Early Termination: Hyperband

Parameters:
  n_estimators: [50-500]
  max_depth: [5-30]
  min_samples_split: [2-20]
  min_samples_leaf: [1-10]
  max_features: ['sqrt', 'log2']
```

### Why Fast Training is Expected

**User Question**: "¿El modelo usa todos los datos? Se ejecuta muy rápido."

**Answer**:  YES, uses ALL data (20,640 samples, 14 features). Fast because:

| Factor | Explanation |
|--------|-------------|
| **Bayesian Optimization** | Smart sampling (not exhaustive GridSearch) |
| **Early Termination** | Stops unpromising runs automatically |
| **Single Training** | One model per combination (no cross-validation per run) |
| **Data Caching** | Loads once, reuses across runs |

**Comparison**:
- GridSearch: 50 combinations × 5-fold CV = 250 models (~2 hours)
- W&B Bayesian: 5 intelligent runs = 5 models (~2-3 minutes)

---

##  Model Details

### Feature Engineering
```python
Numerical Features (8):
- longitude, latitude
- housing_median_age
- total_rooms, total_bedrooms
- population, households
- median_income

Categorical Features (5):
- ocean_proximity_<1H OCEAN
- ocean_proximity_INLAND
- ocean_proximity_ISLAND
- ocean_proximity_NEAR BAY
- ocean_proximity_NEAR OCEAN

Engineered Features (1):
- cluster_label (K-Means with n_clusters=2)
```

### Best Hyperparameters
```yaml
n_estimators: 128
max_depth: 23
min_samples_split: 9
min_samples_leaf: 9
max_features: log2
random_state: 42
```

---

##  Deployment

### Docker
```bash
# Build image
docker build -t housing-price-api -f api/Dockerfile .

# Run container
docker run -p 8000:8000 \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e MLFLOW_TRACKING_URI=./mlruns \
  housing-price-api
```

### Google Cloud Platform

#### Cloud Run (Recommended)
```bash
# Deploy to Cloud Run
gcloud run deploy housing-price-api \
  --source api/ \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "WANDB_API_KEY=$WANDB_API_KEY"
```

#### Cloud Functions
```bash
# Deploy prediction function
gcloud functions deploy predict_housing_price \
  --runtime python312 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point predict \
  --source api/
```

See `api/DEPLOYMENT_CLOUD_FUNCTIONS.md` for detailed instructions.

---

##  Monitoring & Tracking

### MLflow UI
```bash
# Start MLflow server
mlflow ui

# Navigate to: http://localhost:5000
# View: Experiments, runs, models, metrics
```

### Weights & Biases
- **Project**: https://wandb.ai/<your-entity>/housing-mlops-gcp
- **Sweeps**: View hyperparameter optimization results
- **Artifacts**: Training/test data, model metadata

### Model Registry
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model = client.get_registered_model("housing_price_model")
print(f"Latest version: {model.latest_versions[0].version}")
print(f"Stage: {model.latest_versions[0].current_stage}")
```

---

##  Testing

### Run Tests
```bash
# Install test dependencies
uv sync --extra dev --extra api

# Run all tests with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Run API tests
uv run --extra api pytest api/tests/ -v
```

### CI/CD
GitHub Actions workflows in `.github/workflows/`:
- `mlops-complete-pipeline.yml` - Quality gate + pipeline manual 01-07 + promotion opcional
- `mlops-model-monitoring.yml` - Drift monitoring semanal/manual (PSI)
- Actions modernas (`checkout@v6`, `setup-python@v6`, `setup-uv@v7`, `auth@v3`)

---

##  Security

### Exposed API Key Incident (January 13, 2026)

** CRITICAL**: A W&B API key was exposed in Git history. Actions taken:

 **Completed**:
1. Key removed from all files
2. Git history rewritten (all commits cleaned)
3. Comprehensive .gitignore created
4. Local repository verified clean

⏳ **REQUIRED - DO NOW**:
1. **Revoke old key**: https://wandb.ai/settings → API keys → Revoke `d9eeb1a...`
2. **Generate new key**: In same settings page → Generate new key
3. **Force push to GitHub**:
   ```bash
   cd /Users/carlosdaniel/Documents/Projects/Personal_Projects/mlops-hand-on-ML-and-pytorch
   git push origin --force --all
   git push origin --force --tags
   ```
4. **Update GitHub Secret**: Settings → Secrets → Update `WANDB_API_KEY`
5. **Update local .env**: Replace old key with new one

**Why urgent**: Old key is still active and can be used until revoked!

### Prevention
-  `.gitignore` blocks `.env`, credentials, API keys
-  GitHub Actions uses secrets (not hardcoded)
-  GitGuardian monitors for exposed secrets
-  Consider adding `git-secrets` pre-commit hook

---

##  Documentation

### Key Files
- `README.md` (this file) - Complete project documentation
- `config.yaml` - Pipeline configuration
- `configs/model_config.yaml` - Auto-generated model metadata
- `scripts/promote_model.py` - Guarded MLflow stage promotion (Staging -> Production)
- `scripts/monitor_drift.py` - PSI-based drift report from GCS datasets
- `api/README.md` - API usage guide
- `.env.example` - Environment template

### External Resources
- **W&B Docs**: https://docs.wandb.ai/
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **Hydra Config**: https://hydra.cc/
- **FastAPI**: https://fastapi.tiangolo.com/

---

##  Contributing

### Development Setup
```bash
# Install dependencies (dev extras)
uv sync --extra dev --extra api

# Install pre-commit hooks (recommended)
pre-commit install

# Run lint + format check
uv run ruff check src tests api streamlit_app
uv run ruff format --check src tests api streamlit_app

# Run type checking
uv run mypy src/
```

### Branch Strategy
- `master` - Production-ready code
- `cap2-end_to_end` - Current development
- Feature branches - `feature/your-feature-name`

---

##  Troubleshooting

### W&B Web UI Error
**Issue**: `Cannot query field "codePathLocal" on type "RunInfo"`

**Solution**: This is a W&B platform GraphQL bug, NOT your pipeline:
-  Your data is logged successfully
-  Use MLflow UI instead: `mlflow ui`
-  Or clear browser cache and retry
-  See: https://github.com/wandb/wandb/issues

### Model Not Loading
```python
# If MLflow model fails, use local file
import joblib
model_path = "src/model/07_registration/models/trained/housing_price_model.pkl"
model = joblib.load(model_path)
```

### GCS Permission Denied
```bash
# Authenticate with service account
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
```

---

##  Support

- **Issues**: https://github.com/carlosjimenez88M/mlops-hand-on-ML-and-pytorch/issues
- **Email**: danieljimenez88m@gmail.com
- **W&B Support**: support@wandb.ai

---

##  License

This project is licensed under the MIT License.

---

##  Acknowledgments

- **Dataset**: California Housing Prices (Scikit-learn)
- **Tools**: MLflow, Weights & Biases, Hydra, FastAPI
- **Cloud**: Google Cloud Platform
- **CI/CD**: GitHub Actions

---

**Built with  using MLOps best practices**

**Last Updated**: January 13, 2026 | **Model Version**: 5 | **Status**: Production Ready 
