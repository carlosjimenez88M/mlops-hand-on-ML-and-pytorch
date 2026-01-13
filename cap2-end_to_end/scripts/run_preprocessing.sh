#!/bin/bash

###############################################################################
# Preprocessing Module Execution Script
# Author: Carlos Daniel Jiménez
# Date: 2025-01-13
###############################################################################

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_green() { echo -e "${GREEN}$1${NC}"; }
print_yellow() { echo -e "${YELLOW}$1${NC}"; }
print_red() { echo -e "${RED}$1${NC}"; }

# Load environment variables
if [ ! -f .env ]; then
    print_red "Error: .env file not found!"
    exit 1
fi

set -a
source .env
set +a

print_green "==========================================="
print_green "  Running Preprocessing Module"
print_green "==========================================="
echo ""
print_yellow "Configuration:"
echo "  GCS Bucket: $GCS_BUCKET_NAME"
echo "  W&B Project: ${WANDB_PROJECT:-housing-mlops-gcp}"
echo "  Imputation: auto (selects best method)"
echo "  Feature Engineering: enabled"
print_green "==========================================="
echo ""

# Execute preprocessing
python src/data/02_preprocessing_and_imputation/main.py \
  --input_artifact_name "housing_data_raw:latest" \
  --gcs_input_path "data/01-raw/housing.csv" \
  --gcs_output_path "data/02-processed/housing_processed.csv" \
  --artifact_name "housing_data_processed" \
  --artifact_type "processed_data" \
  --artifact_description "California Housing - Processed data" \
  --bucket_name "$GCS_BUCKET_NAME" \
  --wandb_project "${WANDB_PROJECT:-housing-mlops-gcp}" \
  --imputation_strategy "auto" \
  --create_features "True"

print_green ""
print_green "Preprocessing completed successfully!"
