#!/bin/bash

# Script para ejecutar el download de datos localmente
# Carga automáticamente las variables del .env

# Cargar variables de entorno del .env
set -a
source .env
set +a

echo "========================================="
echo "Variables cargadas:"
echo "  GCS_BUCKET_NAME: $GCS_BUCKET_NAME"
echo "  GCP_PROJECT_ID: $GCP_PROJECT_ID"
echo "  WANDB_PROJECT: $WANDB_PROJECT"
echo "========================================="
echo ""

# Ejecutar el script
python src/data/01_download_data/main.py \
  --file_url "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz" \
  --artifact_name "housing_data_raw" \
  --artifact_type "raw_data" \
  --artifact_description "California Housing - Raw data in GCS only" \
  --gcs_output_path "data/01-raw/housing.csv" \
  --bucket_name "$GCS_BUCKET_NAME" \
  --wandb_project "$WANDB_PROJECT"
