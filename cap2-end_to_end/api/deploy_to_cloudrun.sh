#!/bin/bash
# =================================================================
# Cloud Run Deployment Script
# Purpose: Deploy FastAPI to Google Cloud Run
# Author: Carlos Daniel Jiménez
# =================================================================

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID}"
SERVICE_NAME="housing-price-api"
REGION="${GCP_REGION:-us-central1}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
PORT=8080

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Cloud Run Deployment${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID is not set${NC}"
    echo "Set it with: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo -e "${GREEN}Setting GCP project...${NC}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "${GREEN}Enabling required APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com

# Build container image with Cloud Build
echo -e "${GREEN}Building container image...${NC}"
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo -e "${GREEN}Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --port ${PORT} \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --allow-unauthenticated \
    --set-env-vars "GCS_BUCKET_NAME=${GCS_BUCKET_NAME},GCP_PROJECT_ID=${PROJECT_ID}"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Deployment successful!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Service URL: ${SERVICE_URL}"
echo -e "Health check: ${SERVICE_URL}/health"
echo -e "API docs: ${SERVICE_URL}/docs"
echo -e "${BLUE}========================================${NC}"
