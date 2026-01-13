# Housing Price Prediction - MLOps Pipeline

End-to-end MLOps pipeline for predicting California housing prices with automated training, evaluation, and deployment.

## Features

- Complete MLOps pipeline with 5 orchestrated steps
- Experiment tracking with Weights & Biases
- Model versioning and artifact management with MLflow
- Business-focused evaluation metrics (MAPE, accuracy thresholds)
- FastAPI service for model predictions
- Cloud Run deployment for production
- Comprehensive testing and CI/CD with GitHub Actions
- Docker support for containerized execution

## Quick Start

### Prerequisites

- Python 3.12+
- `uv` package manager (or pip)
- Google Cloud account (for GCS and Cloud Run)
- Weights & Biases account

### Installation

```bash
# Clone repository
git clone <repository-url>
cd cap2-end_to_end

# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Run Pipeline

```bash
# Run complete pipeline (all 5 steps)
make run-pipeline

# Run specific steps
make run-download
make run-preprocessing

# Run tests
make run-tests
make test-cov
```

### Run API

```bash
# Run locally
make api-local

# Or with Docker Compose
make compose-up

# Test API
make api-test
```

## Pipeline Architecture

The pipeline consists of 5 sequential steps:

### Step 1: Data Download
**Location**: `src/data/01_download_data/`

- Downloads California housing dataset
- Uploads to Google Cloud Storage
- Logs artifacts to W&B

### Step 2: Preprocessing & Imputation
**Location**: `src/data/02_preprocessing_and_imputation/`

- Handles missing values using median imputation
- Preprocesses features
- Validates data quality
- Tracks preprocessing statistics

**Output**: Clean dataset ready for feature engineering

### Step 3: Feature Engineering
**Location**: `src/data/03_feature_engineering/`

- Creates derived features:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`
- Adds cluster-based features using DBSCAN
- Performs hyperparameter optimization (optional)
- Logs feature importance to W&B

**Key Parameters**:
- `optimize_hyperparams`: Enable/disable optimization
- `n_clusters`: Number of clusters for feature engineering
- `gamma`: DBSCAN gamma parameter

### Step 4: Data Segregation
**Location**: `src/data/04_segregation/`

- Stratified train/test split
- Preserves target distribution
- Uploads splits to GCS
- Logs dataset statistics

**Output**:
- Training set: `data/04-segregated/train.csv`
- Test set: `data/04-segregated/test.csv`

### Step 5: Model Selection
**Location**: `src/model/05_model_selection/`

- Trains multiple algorithms with GridSearchCV:
  - Random Forest
  - Gradient Boosting
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
- Selects best model based on MAPE
- Evaluates with business metrics
- Registers best model to MLflow and GCS

**Business Metrics**:
- **MAPE** (Mean Absolute Percentage Error): Primary metric
  - Interpretation: "5% MAPE = predictions are off by 5% on average"
- **Median APE**: Robust to outliers
- **Within-X% Thresholds**: % of predictions within 5%, 10%, 15% of actual
- **Traditional Metrics**: MAE, RMSE, R²

## Project Structure

```
cap2-end_to_end/
├── src/
│   ├── data/                   # Data pipeline steps
│   │   ├── 01_download_data/
│   │   ├── 02_preprocessing_and_imputation/
│   │   ├── 03_feature_engineering/
│   │   └── 04_segregation/
│   └── model/                  # Model training
│       └── 05_model_selection/
├── api/                        # FastAPI service
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   ├── models/
│   │   └── routers/
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── deploy_to_cloudrun.sh
├── tests/                      # Pipeline tests
├── .github/
│   └── workflows/
│       └── mlops-pipeline-manual.yml
├── config.yaml                 # Pipeline configuration
├── main.py                     # Pipeline orchestrator
├── Makefile                    # Command shortcuts
├── Dockerfile                  # Pipeline container
└── docker-compose.yaml         # API development
```

## Configuration

### Pipeline Configuration

Edit `config.yaml` to customize pipeline behavior:

```yaml
main:
  execute_steps:
    - "01_download_data"
    - "02_preprocessing_and_imputation"
    - "03_feature_engineering"
    - "04_segregation"
    - "05_model_selection"

feature_engineering:
  optimize_hyperparams: true
  n_clusters: 15
  gamma: 1.0

model_selection:
  target_column: "median_house_value"
  random_state: 42
```

### Environment Variables

Create `.env` file:

```bash
# Google Cloud
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Weights & Biases
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=housing-mlops-gcp

# API Configuration (optional)
MODEL_PATH=models/best_model.pkl
```

## GitHub Actions

### Manual Trigger Workflow

**File**: `.github/workflows/mlops-pipeline-manual.yml`

**Purpose**: Execute complete pipeline with manual control

**Features**:
- Manual trigger only (`workflow_dispatch`)
- Configurable options:
  - `steps_to_run`: all, data_only, model_only, test_only
  - `optimize_hyperparams`: true/false
- Three sequential jobs:
  1. Unit tests with coverage
  2. Data pipeline (Steps 1-4)
  3. Model training (Step 5)
- Artifact uploads for logs and models

**Usage**:

1. Go to GitHub Actions tab
2. Select "MLOps Pipeline - Training & Registration"
3. Click "Run workflow"
4. Choose options and run

**Benefits**:
- Prevents unwanted executions on push
- Ideal for multi-project repositories
- Full control over execution

## Makefile Commands

### Setup & Installation
```bash
make install         # Install all dependencies
make install-uv      # Install uv package manager
```

### Pipeline Execution
```bash
make run-pipeline    # Run complete pipeline (Steps 1-5)
make run-download    # Run only download step
make run-preprocessing  # Run only preprocessing step
```

### Testing
```bash
make run-tests       # Run unit tests
make test-cov        # Run tests with coverage
```

### Docker Commands
```bash
make docker-build    # Build pipeline Docker image
make docker-run      # Run pipeline in Docker
make compose-up      # Start API with docker-compose
make compose-down    # Stop docker-compose services
```

### API Commands
```bash
make api-install     # Install API dependencies
make api-local       # Run API locally (port 8080)
make api-test        # Run API tests
make api-test-cov    # Run API tests with coverage
make api-build       # Build API Docker image
```

### Development
```bash
make lint            # Run code linters
make format          # Format code with ruff
make clean           # Clean cache and temp files
```

## API Usage

See detailed API documentation in [`api/README.md`](api/README.md)

### Quick Example

```bash
# Start API
make api-local

# Make prediction
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "longitude": -122.23,
      "latitude": 37.88,
      "housing_median_age": 41.0,
      "total_rooms": 880.0,
      "total_bedrooms": 129.0,
      "population": 322.0,
      "households": 126.0,
      "median_income": 8.3252,
      "ocean_proximity": "NEAR BAY"
    }]
  }'
```

## Cloud Deployment

### Deploy API to Cloud Run

```bash
cd api
export GCP_PROJECT_ID=your-project-id
export GCS_BUCKET_NAME=your-bucket-name
./deploy_to_cloudrun.sh
```

See [`api/README.md`](api/README.md) for detailed deployment instructions.

## Business Metrics Explained

### Mean Absolute Percentage Error (MAPE)

**Primary metric for model selection**

- Measures average % difference between predictions and actual values
- Easy to interpret: "5% MAPE = predictions off by 5% on average"
- Business-friendly: Stakeholders understand percentages
- Scale-independent: Works across different price ranges

**Formula**: `MAPE = (1/n) * Σ|actual - predicted| / actual * 100`

### Accuracy Thresholds

**Complementary metrics for business context**

- **Within 5%**: High precision predictions
- **Within 10%**: Acceptable predictions
- **Within 15%**: Moderate predictions

**Example**: If 80% of predictions are within 10%, the model is highly accurate for most cases.

### Why MAPE over RMSE?

1. **Interpretability**: "5% error" vs "$50,000 error"
2. **Scale independence**: Works for $100K and $1M homes equally
3. **Business alignment**: Stakeholders think in percentages
4. **Decision making**: Easy to set acceptable thresholds

## Testing

### Pipeline Tests

```bash
# Run all tests
make run-tests

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### API Tests

```bash
# Run API tests
make api-test

# Run with coverage
make api-test-cov
```

## Troubleshooting

### Pipeline Issues

**Issue**: `WANDB_API_KEY not found`

**Solution**: Set environment variable:
```bash
export WANDB_API_KEY=your-key
```

**Issue**: `GCS bucket access denied`

**Solution**: Verify service account permissions:
```bash
gcloud auth application-default login
```

### API Issues

**Issue**: `Model file not found`

**Solution**: Ensure model exists at configured path or run pipeline first:
```bash
make run-pipeline
```

**Issue**: `Port 8080 already in use`

**Solution**: Use different port:
```bash
cd api && uvicorn app.main:app --port 8081
```

## Production Recommendations

### Security
1. Enable Cloud Run authentication
2. Implement API key validation
3. Use Secret Manager for credentials
4. Configure CORS appropriately
5. Enable HTTPS only

### Performance
1. Set min instances to reduce cold starts
2. Implement response caching
3. Use model optimization (quantization, ONNX)
4. Monitor and optimize resource allocation
5. Use Cloud CDN for static assets

### Monitoring
1. Set up Cloud Monitoring alerts
2. Track prediction metrics
3. Monitor model drift
4. Log errors and exceptions
5. Create custom dashboards

### Cost Optimization
1. Use CPU allocation only during requests
2. Set appropriate max instances
3. Use min-instances=0 for low traffic
4. Monitor and optimize memory/CPU
5. Use preemptible instances for training

## CI/CD Workflow

1. **Development**: Make changes locally, test with `make run-tests`
2. **Commit**: Push changes to GitHub
3. **Manual Trigger**: Run GitHub Action workflow
4. **Pipeline Execution**: Data pipeline → Model training
5. **Artifacts**: Download trained models from artifacts
6. **Deployment**: Deploy API to Cloud Run with `./deploy_to_cloudrun.sh`
7. **Monitoring**: Track metrics in W&B and Cloud Monitoring

## Contributing

1. Create feature branch
2. Make changes
3. Add tests for new features
4. Run `make run-tests` and `make api-test`
5. Update documentation
6. Create pull request

## License

This project is part of the MLOps hands-on learning series.

## Contact

**Author**: Carlos Daniel Jiménez
**Email**: danieljimenez88m@gmail.com

## Acknowledgments

- California Housing Dataset from scikit-learn
- MLOps best practices from the community
- Google Cloud Platform for infrastructure
- Weights & Biases for experiment tracking
