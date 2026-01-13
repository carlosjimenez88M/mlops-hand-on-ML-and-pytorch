# Cloud Run Deployment Guide - Housing Price Prediction API

**Comprehensive Step-by-Step Guide for Deploying MLOps API to GCP Cloud Run**

Version: 1.0
Last Updated: 2026-01-13
Target Environment: Production

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Architecture on Cloud Run](#project-architecture-on-cloud-run)
3. [Initial GCP Setup](#initial-gcp-setup)
4. [Secrets Management](#secrets-management)
5. [Container Image Build & Registry](#container-image-build--registry)
6. [Cloud Run Deployment](#cloud-run-deployment)
7. [Domain & SSL Configuration](#domain--ssl-configuration)
8. [Monitoring & Logging](#monitoring--logging)
9. [CI/CD Automation](#cicd-automation)
10. [Scaling & Performance](#scaling--performance)
11. [Security Hardening](#security-hardening)
12. [Cost Optimization](#cost-optimization)
13. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

Install these tools on your local machine:

```bash
# 1. Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud --version

# 2. Docker (for local testing)
# macOS: brew install docker
# Linux: sudo apt-get install docker.io
docker --version

# 3. Authentication helper for Docker
gcloud auth configure-docker

# 4. (Optional) Cloud Run CLI
gcloud components install beta
```

### Required GCP APIs

Enable the following APIs in your GCP project:

```bash
export PROJECT_ID="mlops-practices-wb"
export REGION="us-central1"

gcloud config set project ${PROJECT_ID}

gcloud services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  cloudtrace.googleapis.com
```

### Service Accounts

Create a dedicated service account for Cloud Run:

```bash
# Create service account
gcloud iam service-accounts create cloudrun-mlops-api \
  --display-name="Cloud Run MLOps API Service Account" \
  --description="Service account for housing prediction API"

export SERVICE_ACCOUNT="cloudrun-mlops-api@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant necessary permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/cloudtrace.agent"
```

---

## Project Architecture on Cloud Run

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      CLOUD RUN SERVICE                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  FastAPI Application (Uvicorn)                        │  │
│  │  - /health (Health check endpoint)                    │  │
│  │  - /api/v1/predict (Prediction endpoint)              │  │
│  │  - /docs (API documentation)                          │  │
│  └────────────────────┬──────────────────────────────────┘  │
│                       │                                      │
│  ┌────────────────────▼──────────────────────────────────┐  │
│  │  Model Loader                                         │  │
│  │  - Loads from MLflow Model Registry                   │  │
│  │  - Falls back to local model file                     │  │
│  │  - Caches model in memory                             │  │
│  └────────────────────┬──────────────────────────────────┘  │
└─────────────────────────┼────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌───────────────┐ ┌──────────┐ ┌──────────────┐
│  GCS Bucket   │ │ Secret   │ │   MLflow     │
│  (Models)     │ │ Manager  │ │   Registry   │
└───────────────┘ └──────────┘ └──────────────┘
```

### Resource Requirements

| Resource | Minimum | Recommended | Maximum |
|----------|---------|-------------|---------|
| CPU | 1 vCPU | 2 vCPU | 4 vCPU |
| Memory | 512 MiB | 2 GiB | 8 GiB |
| Timeout | 60s | 300s | 900s |
| Concurrency | 10 | 80 | 1000 |

---

## Initial GCP Setup

### Step 1: Configure Project Settings

```bash
# Set project and region
export PROJECT_ID="mlops-practices-wb"
export REGION="us-central1"
export SERVICE_NAME="housing-prediction-api"
export IMAGE_NAME="housing-mlops-api"

gcloud config set project ${PROJECT_ID}
gcloud config set run/region ${REGION}
```

### Step 2: Create Artifact Registry Repository

Cloud Run now recommends Artifact Registry over Container Registry:

```bash
# Create repository
gcloud artifacts repositories create ${IMAGE_NAME}-repo \
  --repository-format=docker \
  --location=${REGION} \
  --description="Docker repository for Housing Prediction API"

# Configure Docker authentication
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### Step 3: Set Up GCS Bucket (if not exists)

```bash
export BUCKET_NAME="mlops-practices-wb-cap2-end_to_end"

# Create bucket (skip if already exists)
gcloud storage buckets create gs://${BUCKET_NAME} \
  --location=${REGION} \
  --uniform-bucket-level-access

# Create models directory
gcloud storage buckets create gs://${BUCKET_NAME}/models/
```

---

## Secrets Management

### Step 1: Store W&B API Key in Secret Manager

**CRITICAL: NEVER commit API keys to Git. Always use Secret Manager.**

```bash
# Create secret for W&B API key
echo -n "YOUR_ACTUAL_WANDB_API_KEY" | \
gcloud secrets create wandb-api-key \
  --replication-policy="automatic" \
  --data-file=-

# Grant access to Cloud Run service account
gcloud secrets add-iam-policy-binding wandb-api-key \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

# Verify secret was created
gcloud secrets versions access latest --secret="wandb-api-key"
```

### Step 2: Store MLflow Tracking URI (if using remote MLflow)

```bash
# If using a remote MLflow server
echo -n "https://your-mlflow-server.com" | \
gcloud secrets create mlflow-tracking-uri \
  --replication-policy="automatic" \
  --data-file=-

gcloud secrets add-iam-policy-binding mlflow-tracking-uri \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

### Step 3: Create GCS Service Account Key (if needed)

```bash
# Create key for GCS access
gcloud iam service-accounts keys create gcs-key.json \
  --iam-account=${SERVICE_ACCOUNT}

# Store in Secret Manager
gcloud secrets create gcs-service-account-key \
  --replication-policy="automatic" \
  --data-file=gcs-key.json

# IMPORTANT: Delete local key file after upload
rm gcs-key.json

# Grant access
gcloud secrets add-iam-policy-binding gcs-service-account-key \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Container Image Build & Registry

### Step 1: Review Dockerfile

Ensure your Dockerfile is optimized for Cloud Run:

```dockerfile
# api/Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose port (Cloud Run uses PORT env variable)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Step 2: Build Docker Image Locally (Test First)

```bash
cd /Users/carlosdaniel/Documents/Projects/Personal_Projects/mlops-hand-on-ML-and-pytorch/cap2-end_to_end/api

# Build image
docker build -t ${IMAGE_NAME}:latest .

# Test locally
docker run -p 8080:8080 \
  -e GCP_PROJECT_ID=${PROJECT_ID} \
  -e GCS_BUCKET_NAME=${BUCKET_NAME} \
  -e WANDB_API_KEY="test-key" \
  ${IMAGE_NAME}:latest

# Test health endpoint
curl http://localhost:8080/health

# Test prediction endpoint
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'

# Stop container
docker stop $(docker ps -q --filter ancestor=${IMAGE_NAME}:latest)
```

### Step 3: Build and Push to Artifact Registry

```bash
# Define image URL
export IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}-repo/${IMAGE_NAME}:latest"

# Build for Cloud Run
docker build -t ${IMAGE_URL} ./api

# Push to Artifact Registry
docker push ${IMAGE_URL}

# Verify image was pushed
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}-repo
```

### Step 4: Alternative - Use Cloud Build (Recommended for Production)

```bash
# Submit build to Cloud Build (builds in GCP)
gcloud builds submit ./api \
  --tag=${IMAGE_URL} \
  --timeout=20m

# This is better because:
# - No local Docker required
# - Faster upload (builds in GCP network)
# - Automatic vulnerability scanning
# - Build logs stored in Cloud Logging
```

---

## Cloud Run Deployment

### Step 1: Deploy Service (Initial Deployment)

```bash
gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_URL} \
  --platform=managed \
  --region=${REGION} \
  --service-account=${SERVICE_ACCOUNT} \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300s \
  --concurrency=80 \
  --min-instances=1 \
  --max-instances=10 \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET_NAME=${BUCKET_NAME},WANDB_PROJECT=housing-mlops-gcp,WANDB_ENTITY=danieljimenez88m-carlosdanieljimenez-com" \
  --set-secrets="WANDB_API_KEY=wandb-api-key:latest" \
  --allow-unauthenticated \
  --ingress=all \
  --port=8080
```

**Explanation of Flags:**

- `--memory=2Gi`: Allocate 2GB RAM (adjust based on model size)
- `--cpu=2`: Use 2 vCPUs for parallel request handling
- `--timeout=300s`: Allow up to 5 minutes per request (adjust for batch predictions)
- `--concurrency=80`: Handle 80 requests per container instance
- `--min-instances=1`: Keep 1 instance warm (prevents cold starts)
- `--max-instances=10`: Scale up to 10 instances under load
- `--set-secrets`: Mount secrets as environment variables from Secret Manager
- `--allow-unauthenticated`: Allow public access (for now; secure later)
- `--ingress=all`: Allow traffic from all sources

### Step 2: Verify Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --region=${REGION} \
  --format='value(status.url)')

echo "Service URL: ${SERVICE_URL}"

# Test health endpoint
curl ${SERVICE_URL}/health

# Test prediction endpoint
curl -X POST ${SERVICE_URL}/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'

# Check API documentation
open ${SERVICE_URL}/docs
```

### Step 3: Update Deployment (After Code Changes)

```bash
# Rebuild and push new image
docker build -t ${IMAGE_URL} ./api
docker push ${IMAGE_URL}

# Deploy new revision
gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_URL} \
  --region=${REGION}

# Cloud Run will:
# 1. Create new revision
# 2. Gradually shift traffic to new revision
# 3. Keep old revision as fallback
```

### Step 4: Traffic Management (Blue/Green Deployment)

```bash
# Deploy new version without shifting traffic
gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_URL} \
  --region=${REGION} \
  --no-traffic

# List revisions
gcloud run revisions list --service=${SERVICE_NAME} --region=${REGION}

# Gradually shift traffic (50% to new, 50% to old)
gcloud run services update-traffic ${SERVICE_NAME} \
  --region=${REGION} \
  --to-revisions=housing-prediction-api-00002-abc=50,housing-prediction-api-00001-xyz=50

# If new version is stable, shift 100% traffic
gcloud run services update-traffic ${SERVICE_NAME} \
  --region=${REGION} \
  --to-latest
```

---

## Domain & SSL Configuration

### Step 1: Map Custom Domain

```bash
# Add domain mapping
gcloud run domain-mappings create \
  --service=${SERVICE_NAME} \
  --domain=api.yourdomain.com \
  --region=${REGION}

# Cloud Run will provide DNS records to add to your domain registrar
```

### Step 2: Add DNS Records

Add these records to your domain provider (e.g., Cloudflare, GoDaddy):

```
Type: CNAME
Name: api.yourdomain.com
Value: ghs.googlehosted.com
```

### Step 3: Verify SSL Certificate

```bash
# Cloud Run automatically provisions SSL certificate
# Check certificate status
gcloud run domain-mappings describe \
  --domain=api.yourdomain.com \
  --region=${REGION}

# Wait for "ACTIVE" status (can take 15-60 minutes)
```

---

## Monitoring & Logging

### Step 1: Enable Cloud Monitoring

```bash
# Cloud Run automatically sends metrics to Cloud Monitoring
# View metrics in console:
# https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics

# Key metrics to monitor:
# - Request count
# - Request latency (p50, p95, p99)
# - Error rate
# - Container instance count
# - CPU utilization
# - Memory utilization
```

### Step 2: Set Up Log-Based Alerts

```bash
# Create alert for high error rate
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Cloud Run High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=5 \
  --condition-threshold-duration=300s \
  --condition-filter='resource.type="cloud_run_revision" AND severity="ERROR"'
```

### Step 3: View Logs

```bash
# Stream logs in real-time
gcloud run services logs tail ${SERVICE_NAME} --region=${REGION}

# Filter logs by severity
gcloud run services logs read ${SERVICE_NAME} \
  --region=${REGION} \
  --filter='severity="ERROR"' \
  --limit=50

# View logs in console
open "https://console.cloud.google.com/logs/query?project=${PROJECT_ID}"
```

### Step 4: Enable Cloud Trace (Request Tracing)

Add to your FastAPI app:

```python
# api/app/main.py
from google.cloud import trace_v1

# Initialize tracer
tracer = trace_v1.TraceServiceClient()

@app.middleware("http")
async def trace_requests(request: Request, call_next):
    trace_id = request.headers.get("X-Cloud-Trace-Context", "").split("/")[0]
    with tracer.span(name=f"{request.method} {request.url.path}", trace_id=trace_id):
        response = await call_next(request)
    return response
```

---

## CI/CD Automation

### Step 1: Create GitHub Actions Workflow

Create `.github/workflows/deploy-to-cloud-run.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches:
      - main
    paths:
      - 'cap2-end_to_end/api/**'
  workflow_dispatch:

env:
  PROJECT_ID: mlops-practices-wb
  REGION: us-central1
  SERVICE_NAME: housing-prediction-api
  IMAGE_NAME: housing-mlops-api

jobs:
  deploy:
    name: Deploy to Cloud Run
    runs-on: ubuntu-latest

    permissions:
      contents: read
      id-token: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker for Artifact Registry
        run: |
          gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

      - name: Build Docker image
        working-directory: cap2-end_to_end/api
        run: |
          docker build -t ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.IMAGE_NAME }}-repo/${{ env.IMAGE_NAME }}:${{ github.sha }} .
          docker tag ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.IMAGE_NAME }}-repo/${{ env.IMAGE_NAME }}:${{ github.sha }} \
                     ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.IMAGE_NAME }}-repo/${{ env.IMAGE_NAME }}:latest

      - name: Push Docker image
        run: |
          docker push ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.IMAGE_NAME }}-repo/${{ env.IMAGE_NAME }}:${{ github.sha }}
          docker push ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.IMAGE_NAME }}-repo/${{ env.IMAGE_NAME }}:latest

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ${{ env.SERVICE_NAME }} \
            --image=${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.IMAGE_NAME }}-repo/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            --region=${{ env.REGION }} \
            --platform=managed \
            --quiet

      - name: Test deployment
        run: |
          SERVICE_URL=$(gcloud run services describe ${{ env.SERVICE_NAME }} \
            --region=${{ env.REGION }} \
            --format='value(status.url)')
          curl -f ${SERVICE_URL}/health || exit 1
```

### Step 2: Set Up GitHub Secrets

```bash
# Create service account for GitHub Actions
gcloud iam service-accounts create github-actions-deployer \
  --display-name="GitHub Actions Deployer"

# Grant permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Create key
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com

# Add to GitHub Secrets:
# 1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions
# 2. Add secret: GCP_SA_KEY = <paste contents of github-actions-key.json>

# Delete local key
rm github-actions-key.json
```

---

## Scaling & Performance

### Step 1: Configure Autoscaling

```bash
# Update service with autoscaling configuration
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --min-instances=1 \
  --max-instances=100 \
  --cpu-throttling \
  --cpu-boost
```

**Scaling Parameters:**

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--min-instances` | Minimum instances (prevents cold starts) | 1-3 |
| `--max-instances` | Maximum instances (cost control) | 10-100 |
| `--concurrency` | Requests per instance | 80 |
| `--cpu` | vCPUs per instance | 2 |
| `--memory` | Memory per instance | 2Gi |

### Step 2: Optimize Cold Start Time

```bash
# Enable CPU boost (allocates CPU during startup)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --cpu-boost

# This reduces cold start from ~5s to ~2s
```

### Step 3: Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -T 'application/json' \
  -p payload.json \
  ${SERVICE_URL}/api/v1/predict

# Create payload.json:
cat > payload.json << 'EOF'
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
EOF
```

---

## Security Hardening

### Step 1: Enable IAM Authentication

```bash
# Disable public access
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --no-allow-unauthenticated

# Now requires authentication token to access
```

### Step 2: Generate Authentication Token

```bash
# Get authentication token
gcloud auth print-identity-token

# Use token in requests
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  ${SERVICE_URL}/api/v1/predict
```

### Step 3: Configure CORS Properly

Edit `api/app/main.py`:

```python
# BEFORE (INSECURE):
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # DANGER
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AFTER (SECURE):
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)
```

### Step 4: Enable Cloud Armor (DDoS Protection)

```bash
# Create security policy
gcloud compute security-policies create cloudrun-security-policy \
  --description="Cloud Armor policy for Cloud Run"

# Add rate limiting rule
gcloud compute security-policies rules create 1000 \
  --security-policy=cloudrun-security-policy \
  --expression="true" \
  --action=rate-based-ban \
  --rate-limit-threshold-count=100 \
  --rate-limit-threshold-interval-sec=60 \
  --ban-duration-sec=600

# Attach to Cloud Run service (requires Cloud Load Balancer)
```

### Step 5: Enable Binary Authorization (Image Signing)

```bash
# Ensure only signed images can be deployed
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --binary-authorization=default
```

---

## Cost Optimization

### Step 1: Understand Cloud Run Pricing

**Cost Components:**
1. **CPU**: $0.00002400 per vCPU-second
2. **Memory**: $0.00000250 per GiB-second
3. **Requests**: $0.40 per million requests
4. **Networking**: $0.12 per GB egress

**Example Monthly Cost:**
- 1M requests/month
- 1 vCPU, 2 GiB memory
- 300ms average request duration
- Min instances: 1

```
CPU cost: 1M * 0.3s * $0.000024 * 1 vCPU = $7.20
Memory cost: 1M * 0.3s * $0.0000025 * 2 GiB = $1.50
Request cost: 1M * $0.40 / 1M = $0.40
Idle cost (min instances): 730 hours * 1 vCPU * $0.000024 * 3600 = $63.07

Total: ~$72.17/month
```

### Step 2: Optimize Costs

```bash
# 1. Reduce min instances (increases cold starts but saves cost)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --min-instances=0

# 2. Use smaller CPU allocation (if model is small)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --cpu=1 \
  --memory=1Gi

# 3. Enable CPU throttling (only allocate CPU during request)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --cpu-throttling

# 4. Set shorter timeout (prevent runaway requests)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --timeout=60s
```

### Step 3: Monitor Costs

```bash
# View cost breakdown in Cloud Console
open "https://console.cloud.google.com/billing/reports?project=${PROJECT_ID}"

# Set budget alert
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Cloud Run Budget" \
  --budget-amount=100 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

---

## Troubleshooting

### Issue 1: Container Fails to Start

**Symptoms:**
- Deployment succeeds but requests fail
- Logs show "Container failed to start"

**Diagnosis:**
```bash
# Check logs
gcloud run services logs read ${SERVICE_NAME} \
  --region=${REGION} \
  --limit=100

# Common causes:
# 1. Port mismatch (Cloud Run expects port from $PORT env variable)
# 2. Missing dependencies
# 3. Startup timeout (container takes >240s to start)
```

**Fix:**
```bash
# Ensure Dockerfile CMD uses $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}

# Increase startup timeout
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --timeout=600s
```

### Issue 2: "Permission Denied" Errors

**Symptoms:**
- Cannot read from GCS
- Cannot access secrets

**Fix:**
```bash
# Check service account permissions
gcloud projects get-iam-policy ${PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:${SERVICE_ACCOUNT}"

# Grant missing permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectViewer"
```

### Issue 3: High Latency / Slow Predictions

**Diagnosis:**
```bash
# Check Cloud Trace for bottlenecks
open "https://console.cloud.google.com/traces/list?project=${PROJECT_ID}"

# Common causes:
# 1. Model loading on every request (should cache)
# 2. Cold starts (increase min-instances)
# 3. Insufficient CPU/memory
```

**Fix:**
```bash
# Increase resources
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --cpu=2 \
  --memory=4Gi \
  --min-instances=2

# Enable CPU boost
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --cpu-boost
```

### Issue 4: "Out of Memory" Errors

**Symptoms:**
- Container crashes during requests
- Logs show "OOMKilled"

**Fix:**
```bash
# Increase memory allocation
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --memory=4Gi

# Reduce concurrency (fewer requests per container)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --concurrency=40
```

### Issue 5: Image Pull Errors

**Symptoms:**
- "Failed to pull image"
- "Unauthorized" errors

**Fix:**
```bash
# Ensure service account has Artifact Registry access
gcloud artifacts repositories add-iam-policy-binding ${IMAGE_NAME}-repo \
  --location=${REGION} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/artifactregistry.reader"

# Verify image exists
gcloud artifacts docker images list \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}-repo
```

---

## Quick Reference Commands

### Deploy New Version

```bash
# One-liner deployment
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}-repo/${IMAGE_NAME}:latest ./api && \
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}-repo/${IMAGE_NAME}:latest && \
gcloud run deploy ${SERVICE_NAME} --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}-repo/${IMAGE_NAME}:latest --region=${REGION}
```

### View Service Status

```bash
gcloud run services describe ${SERVICE_NAME} --region=${REGION}
```

### Stream Logs

```bash
gcloud run services logs tail ${SERVICE_NAME} --region=${REGION}
```

### Delete Service

```bash
gcloud run services delete ${SERVICE_NAME} --region=${REGION}
```

---

## Next Steps

After successful deployment:

1. Set up monitoring dashboards in Cloud Monitoring
2. Configure alerting policies for critical metrics
3. Implement A/B testing with traffic splits
4. Set up automated model retraining pipeline
5. Implement model performance monitoring
6. Configure backup and disaster recovery

---

## Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Best Practices for Cloud Run](https://cloud.google.com/run/docs/best-practices)
- [Cloud Run Pricing Calculator](https://cloud.google.com/products/calculator)
- [FastAPI on Cloud Run](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-13
**Maintained By:** MLOps Team
