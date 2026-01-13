# Deploying API to Google Cloud Functions (2nd Generation)

This guide explains how to deploy the Housing Price Prediction API to Google Cloud Functions.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Environment Variables](#environment-variables)
6. [MLflow Model Integration](#mlflow-model-integration)
7. [Testing](#testing)
8. [Monitoring](#monitoring)
9. [Cost Optimization](#cost-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **Required APIs** enabled:
   - Cloud Functions API
   - Cloud Build API
   - Cloud Run API (used by Cloud Functions 2nd gen)
   - Cloud Storage API
   - Artifact Registry API

3. **IAM Permissions**:
   - Cloud Functions Developer
   - Service Account User
   - Cloud Storage Object Viewer (for model access)

4. **Local Tools**:
   ```bash
   # Install gcloud CLI
   # https://cloud.google.com/sdk/docs/install

   # Authenticate
   gcloud auth login

   # Set project
   gcloud config set project YOUR_PROJECT_ID
   ```

---

## Architecture Overview

```
User Request
    ↓
Cloud Functions (FastAPI + Uvicorn)
    ↓
├─→ MLflow Model Registry (Priority)
├─→ GCS Bucket (Fallback)
└─→ Local Model (Fallback)
    ↓
├─→ Weights & Biases (Logging)
└─→ Response
```

**Why Cloud Functions 2nd Gen?**
- Built on Cloud Run
- Better scaling (0 to many instances)
- Longer timeouts (60 min vs 9 min)
- More memory (up to 16GB)
- Better for ML APIs

---

## Configuration

### 1. Create Function Entry Point

Cloud Functions requires a specific entry point. We'll use the existing FastAPI app:

**`api/main_cloud_function.py`** (to be created):
```python
"""
Cloud Functions entry point for FastAPI application.
"""
from app.main import app

# Cloud Functions will call this
def handler(request):
    """Entry point for Cloud Functions."""
    import asgi_adapter
    return asgi_adapter.handle_request(app, request)
```

### 2. Update requirements.txt

Create `api/requirements.txt`:
```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0
scikit-learn>=1.0.0
pandas>=2.0.0
numpy>=1.20.0
google-cloud-storage>=2.0.0
mlflow>=2.0.0
wandb>=0.15.0
functions-framework==3.*
```

### 3. Create Cloud Function Configuration

**`api/cloudfunctions.yaml`**:
```yaml
runtime: python312
entry_point: handler
memory: 2048Mi
timeout: 540s  # 9 minutes
min_instances: 0
max_instances: 10
environment_variables:
  MLFLOW_MODEL_NAME: "housing_price_model"
  MLFLOW_MODEL_STAGE: "Production"
  WANDB_PROJECT: "housing-mlops-api"
  LOCAL_MODEL_PATH: ""
  GCS_BUCKET: ""
  GCS_MODEL_PATH: ""
```

---

## Deployment Options

### Option 1: Deploy with gcloud CLI (Recommended)

```bash
cd api

# Deploy function
gcloud functions deploy housing-price-api \
  --gen2 \
  --runtime=python312 \
  --region=us-central1 \
  --source=. \
  --entry-point=handler \
  --trigger-http \
  --allow-unauthenticated \
  --memory=2048Mi \
  --timeout=540s \
  --min-instances=0 \
  --max-instances=10 \
  --set-env-vars="MLFLOW_MODEL_NAME=housing_price_model,MLFLOW_MODEL_STAGE=Production,WANDB_PROJECT=housing-mlops-api" \
  --set-secrets="WANDB_API_KEY=WANDB_API_KEY:latest,MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI:latest"
```

### Option 2: Deploy from Source Repository

```bash
# Build and deploy from Git
gcloud functions deploy housing-price-api \
  --gen2 \
  --runtime=python312 \
  --region=us-central1 \
  --source=https://github.com/YOUR_USERNAME/YOUR_REPO \
  --source-path=api \
  --entry-point=handler \
  --trigger-http \
  --allow-unauthenticated
```

### Option 3: Use Cloud Build (CI/CD)

**`.github/workflows/deploy-api-cloud-functions.yml`**:
```yaml
name: Deploy API to Cloud Functions

on:
  push:
    branches:
      - main
    paths:
      - 'api/**'
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}

      - name: Deploy to Cloud Functions
        run: |
          gcloud functions deploy housing-price-api \
            --gen2 \
            --runtime=python312 \
            --region=us-central1 \
            --source=./api \
            --entry-point=handler \
            --trigger-http \
            --allow-unauthenticated \
            --memory=2048Mi \
            --timeout=540s \
            --set-env-vars="MLFLOW_MODEL_NAME=housing_price_model,MLFLOW_MODEL_STAGE=Production" \
            --set-secrets="WANDB_API_KEY=WANDB_API_KEY:latest"
```

---

## Environment Variables

Set these as environment variables or secrets:

### Required
- `MLFLOW_MODEL_NAME`: Name of registered model (e.g., "housing_price_model")
- `MLFLOW_MODEL_STAGE`: Model stage to load ("Production", "Staging", or "" for latest)

### Optional (with defaults)
- `MLFLOW_TRACKING_URI`: MLflow tracking server URL (leave empty for local)
- `LOCAL_MODEL_PATH`: Path to local model file (fallback)
- `GCS_BUCKET`: GCS bucket for model (fallback)
- `GCS_MODEL_PATH`: Path to model in GCS (fallback)
- `WANDB_API_KEY`: W&B API key for logging
- `WANDB_PROJECT`: W&B project name

### Using Secret Manager

```bash
# Create secrets
echo -n "your-wandb-api-key" | gcloud secrets create WANDB_API_KEY --data-file=-
echo -n "your-mlflow-uri" | gcloud secrets create MLFLOW_TRACKING_URI --data-file=-

# Grant access to function's service account
gcloud secrets add-iam-policy-binding WANDB_API_KEY \
  --member="serviceAccount:YOUR_PROJECT_ID@appspot.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## MLflow Model Integration

### Setup 1: Use MLflow Model Registry (Recommended)

The API will automatically load the latest Production/Staging model:

```bash
# Environment variables needed
MLFLOW_MODEL_NAME=housing_price_model
MLFLOW_MODEL_STAGE=Production
MLFLOW_TRACKING_URI=https://your-mlflow-server.com  # Optional if using local
```

**Benefits**:
- Automatic model versioning
- A/B testing support
- Model lineage tracking
- Easy rollback

### Setup 2: Use GCS Bucket (Fallback)

Store model in GCS and configure:

```bash
# Upload model to GCS
gsutil cp models/trained/housing_price_model.pkl gs://YOUR_BUCKET/models/trained/

# Environment variables
GCS_BUCKET=YOUR_BUCKET
GCS_MODEL_PATH=models/trained/housing_price_model.pkl
```

### Setup 3: Package Model in Function (Fallback)

Include model file in deployment:

```bash
# Add model to api/ directory
cp models/trained/housing_price_model.pkl api/models/

# Environment variable
LOCAL_MODEL_PATH=models/housing_price_model.pkl
```

**Note**: Model is loaded on cold start (first request). Subsequent requests use cached model.

---

## Testing

### 1. Test Locally

```bash
cd api
uvicorn app.main:app --reload --port 8080
```

Test endpoints:
```bash
# Health check
curl http://localhost:8080/health

# Prediction
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252
  }'
```

### 2. Test Deployed Function

```bash
# Get function URL
FUNCTION_URL=$(gcloud functions describe housing-price-api \
  --gen2 \
  --region=us-central1 \
  --format='value(serviceConfig.uri)')

# Test health
curl $FUNCTION_URL/health

# Test prediction
curl -X POST $FUNCTION_URL/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252
  }'
```

### 3. Load Testing

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Run load test
hey -n 1000 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"longitude":-122.23,"latitude":37.88,"housing_median_age":41.0,"total_rooms":880.0,"total_bedrooms":129.0,"population":322.0,"households":126.0,"median_income":8.3252}' \
  $FUNCTION_URL/api/v1/predict
```

---

## Monitoring

### 1. Cloud Functions Dashboard

View metrics in Google Cloud Console:
- Invocations
- Execution time
- Memory usage
- Active instances
- Errors

### 2. Cloud Logging

```bash
# View logs
gcloud functions logs read housing-price-api --gen2 --limit=50

# Stream logs
gcloud functions logs read housing-price-api --gen2 --follow
```

### 3. Weights & Biases

Monitor predictions in W&B:
- Go to https://wandb.ai/YOUR_ENTITY/housing-mlops-api
- View prediction counts, response times, model versions

### 4. Set up Alerts

```bash
# Create alert for high error rate
gcloud alpha monitoring policies create \
  --notification-channels=YOUR_CHANNEL_ID \
  --display-name="API Error Rate Alert" \
  --condition-display-name="High error rate" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s
```

---

## Cost Optimization

### Current Costs (Estimates)

**Cloud Functions 2nd Gen Pricing** (us-central1):
- Invocations: $0.40 per million
- Compute time: $0.000024 per GB-second
- Memory: $0.0000025 per GB-second
- Network egress: $0.12 per GB

**Example**:
- 100,000 requests/month
- 2GB memory
- 2 second avg response time
- **Cost**: ~$10-15/month

### Optimization Tips

1. **Use min_instances=0** for low traffic (default)
2. **Set max_instances** to control costs
3. **Use appropriate memory** (2GB for ML models)
4. **Cache model** (done automatically)
5. **Use regional deployment** (cheaper than multi-region)
6. **Monitor cold starts** (first request is slower)

### Scaling Configuration

```bash
# Low traffic (development)
--min-instances=0 --max-instances=3 --memory=2048Mi

# Medium traffic (production)
--min-instances=1 --max-instances=10 --memory=2048Mi

# High traffic (scale)
--min-instances=3 --max-instances=50 --memory=4096Mi
```

---

## Troubleshooting

### Issue 1: Model Not Loading

**Error**: `No model path configured`

**Solution**: Check environment variables
```bash
gcloud functions describe housing-price-api --gen2 --region=us-central1 --format=json | jq .serviceConfig.environmentVariables
```

### Issue 2: Timeout Errors

**Error**: `Function execution took too long`

**Solution**: Increase timeout
```bash
gcloud functions deploy housing-price-api --timeout=540s
```

### Issue 3: Out of Memory

**Error**: `Memory limit exceeded`

**Solution**: Increase memory
```bash
gcloud functions deploy housing-price-api --memory=4096Mi
```

### Issue 4: Cold Start Latency

**Problem**: First request takes 10+ seconds

**Solutions**:
1. Use `--min-instances=1` to keep 1 instance warm
2. Implement health check pinging
3. Optimize model loading (use smaller models)

### Issue 5: Authentication Errors

**Error**: `Could not load credentials`

**Solution**: Use service account with proper permissions
```bash
gcloud functions deploy housing-price-api \
  --service-account=YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com
```

---

## Next Steps

1. **Set up CI/CD**: Automate deployments with GitHub Actions
2. **Enable Authentication**: Add API keys or OAuth
3. **Add Rate Limiting**: Use Cloud Endpoints or API Gateway
4. **Set up Custom Domain**: Use Cloud Load Balancer
5. **Implement Caching**: Use Cloud CDN for responses
6. **Add Monitoring**: Set up alerts and dashboards
7. **A/B Testing**: Deploy multiple versions
8. **Cost Monitoring**: Set up billing alerts

---

## Alternative: Cloud Run (Recommended for Production)

For production workloads, consider **Cloud Run** instead:

```bash
# Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/housing-api:latest ./api

# Deploy to Cloud Run
gcloud run deploy housing-api \
  --image gcr.io/YOUR_PROJECT_ID/housing-api:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --allow-unauthenticated \
  --set-env-vars="MLFLOW_MODEL_NAME=housing_price_model,MLFLOW_MODEL_STAGE=Production"
```

**Benefits over Cloud Functions**:
- More control over runtime
- Better for containers
- More CPU options
- Better for ML workloads
- Similar pricing

---

## Resources

- [Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [FastAPI on Cloud Functions](https://cloud.google.com/functions/docs/samples/functions-http-fastapi)
- [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator)
