#!/bin/bash

###############################################################################
# Complete MLOps Pipeline Execution Script
# Author: Carlos Daniel Jiménez
# Date: 2025-01-13
#
# This script executes the complete data pipeline:
# 1. Download data from source
# 2. Preprocess and impute missing values
# 3. (Future: Feature engineering, training, etc.)
###############################################################################

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
print_green() { echo -e "${GREEN}$1${NC}"; }
print_yellow() { echo -e "${YELLOW}$1${NC}"; }
print_red() { echo -e "${RED}$1${NC}"; }
print_blue() { echo -e "${BLUE}$1${NC}"; }

# Print header
print_header() {
    echo ""
    print_blue "=============================================="
    print_blue "  $1"
    print_blue "=============================================="
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_red "Error: .env file not found!"
    print_yellow "Please create a .env file with your configuration."
    print_yellow "You can copy .env.example and edit it:"
    echo "  cp .env.example .env"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Verify required environment variables
if [ -z "$GCS_BUCKET_NAME" ]; then
    print_red "Error: GCS_BUCKET_NAME not set in .env file"
    exit 1
fi

# Print configuration
print_header "Pipeline Configuration"
print_green "  GCS Bucket: $GCS_BUCKET_NAME"
print_green "  GCP Project: ${GCP_PROJECT_ID:-Not set}"
print_green "  W&B Project: ${WANDB_PROJECT:-housing-mlops-gcp}"
echo ""

# Track execution time
START_TIME=$(date +%s)

###############################################################################
# STEP 1: Download Data
###############################################################################
print_header "STEP 1/2: Downloading Data"

print_yellow "Starting data download from source..."

if python src/data/01_download_data/main.py \
    --file_url "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz" \
    --artifact_name "housing_data_raw" \
    --artifact_type "raw_data" \
    --artifact_description "California Housing - Raw data in GCS only" \
    --gcs_output_path "data/01-raw/housing.csv" \
    --bucket_name "$GCS_BUCKET_NAME" \
    --wandb_project "${WANDB_PROJECT:-housing-mlops-gcp}"; then

    print_green "Data download completed successfully!"
else
    print_red "Error: Data download failed!"
    exit 1
fi

###############################################################################
# STEP 2: Preprocessing and Imputation
###############################################################################
print_header "STEP 2/2: Preprocessing and Imputation"

print_yellow "Starting data preprocessing..."

if python src/data/02_preprocessing_and_imputation/main.py \
    --input_artifact_name "housing_data_raw:latest" \
    --gcs_input_path "data/01-raw/housing.csv" \
    --gcs_output_path "data/02-processed/housing_processed.csv" \
    --artifact_name "housing_data_processed" \
    --artifact_type "processed_data" \
    --artifact_description "California Housing - Processed data with imputation" \
    --bucket_name "$GCS_BUCKET_NAME" \
    --wandb_project "${WANDB_PROJECT:-housing-mlops-gcp}" \
    --imputation_strategy "auto" \
    --create_features "True"; then

    print_green "Data preprocessing completed successfully!"
else
    print_red "Error: Data preprocessing failed!"
    exit 1
fi

###############################################################################
# Summary
###############################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

print_header "Pipeline Execution Summary"
print_green "All steps completed successfully!"
print_yellow "Total execution time: ${DURATION} seconds"
print_blue "=============================================="
echo ""

print_yellow "Next steps:"
echo "  - Check GCS bucket: gs://$GCS_BUCKET_NAME"
echo "  - Review W&B artifacts at: https://wandb.ai/${WANDB_ENTITY:-your-entity}/${WANDB_PROJECT:-housing-mlops-gcp}"
echo "  - Run tests: make test"
echo "  - Check coverage: make coverage"
echo ""

exit 0
