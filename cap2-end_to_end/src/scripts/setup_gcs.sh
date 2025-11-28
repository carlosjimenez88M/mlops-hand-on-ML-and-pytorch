#!/bin/bash
# Setup GCS structure for MLOps Pipeline - Cap2 End-to-End
# Author: Carlos Daniel Hernandez
# Date: 2025-11-28

set -e

PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="${PROJECT_ID}-cap2-end_to_end"
REGION="us-central1"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        SETUP GCS PARA MLOPS PIPELINE - CAP2                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Project ID: $PROJECT_ID"
echo "Bucket: gs://$BUCKET_NAME"
echo "Region: $REGION"
echo ""

# Crear bucket si no existe
if ! gsutil ls gs://$BUCKET_NAME 2>/dev/null; then
    echo "📦 Creando bucket..."
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME

    # Configurar versionado
    gsutil versioning set on gs://$BUCKET_NAME

    # Configurar lifecycle (opcional: eliminar versiones antiguas después de 30 días)
    echo '{
      "lifecycle": {
        "rule": [
          {
            "action": {"type": "Delete"},
            "condition": {
              "numNewerVersions": 3,
              "isLive": false
            }
          }
        ]
      }
    }' | gsutil lifecycle set /dev/stdin gs://$BUCKET_NAME

    echo "✓ Bucket creado con versionado habilitado"
else
    echo "✓ Bucket ya existe"
fi

# Crear estructura de carpetas según pipeline
echo ""
echo "📁 Creando estructura de carpetas del pipeline..."
echo ""

# Crear archivo temporal para marcar carpetas
mkdir -p /tmp/gcs_structure
cat > /tmp/gcs_structure/.gitkeep << EOF
# This file keeps the directory structure in GCS
# MLOps Pipeline - Cap2 End-to-End
EOF

# ============================================
# DATA PIPELINE FOLDERS
# ============================================
echo "📊 Data Pipeline:"

# Step 01: Raw Data
echo "  → data/01-raw/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/01-raw/.gitkeep 2>/dev/null || true

# Step 02: Processed Data (after preprocessing & imputation)
echo "  → data/02-processed/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/02-processed/.gitkeep 2>/dev/null || true

# Step 03: Feature Engineering
echo "  → data/03-features/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/03-features/.gitkeep 2>/dev/null || true

# Step 04: Train/Test Split
echo "  → data/04-split/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/04-split/train/.gitkeep 2>/dev/null || true
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/04-split/test/.gitkeep 2>/dev/null || true
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/04-split/val/.gitkeep 2>/dev/null || true

# ============================================
# MODELS FOLDERS
# ============================================
echo ""
echo "🤖 Models:"

# Experiments
echo "  → models/experiments/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/models/experiments/.gitkeep 2>/dev/null || true

# Production models
echo "  → models/production/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/models/production/.gitkeep 2>/dev/null || true

# Model checkpoints
echo "  → models/checkpoints/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/models/checkpoints/.gitkeep 2>/dev/null || true

# ============================================
# ARTIFACTS & METRICS
# ============================================
echo ""
echo "📦 Artifacts & Metrics:"

# MLflow artifacts
echo "  → artifacts/mlflow/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/artifacts/mlflow/.gitkeep 2>/dev/null || true

# W&B artifacts references
echo "  → artifacts/wandb/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/artifacts/wandb/.gitkeep 2>/dev/null || true

# Metrics and plots
echo "  → metrics/plots/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/metrics/plots/.gitkeep 2>/dev/null || true
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/metrics/reports/.gitkeep 2>/dev/null || true

# ============================================
# LOGS
# ============================================
echo ""
echo "📝 Logs:"

echo "  → logs/pipeline/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/logs/pipeline/.gitkeep 2>/dev/null || true

echo "  → logs/training/"
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/logs/training/.gitkeep 2>/dev/null || true

# Limpiar
rm -rf /tmp/gcs_structure

echo ""
echo "✓ Estructura de carpetas creada"
echo ""

# Verificar estructura
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           ESTRUCTURA DEL BUCKET                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
gsutil ls -r gs://$BUCKET_NAME/ | grep "/$" | head -20

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           ✅ SETUP COMPLETADO EXITOSAMENTE                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Bucket URI: gs://$BUCKET_NAME"
echo "Region: $REGION"
echo "Versionado: HABILITADO"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           SIGUIENTES PASOS                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Actualiza tu archivo .env:"
echo "   GCS_BUCKET_NAME=$BUCKET_NAME"
echo "   GCP_PROJECT_ID=$PROJECT_ID"
echo "   GCP_REGION=$REGION"
echo ""
echo "2. Instala dependencias:"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Configura Weights & Biases:"
echo "   wandb login"
echo ""
echo "4. Ejecuta el pipeline de descarga de datos:"
echo "   python src/data/01_download_data/main.py \\"
echo "     --bucket_name $BUCKET_NAME"
echo ""
echo "5. Ejecuta el pipeline de preprocessing:"
echo "   python src/data/02_preprocessing_and_imputation/main.py \\"
echo "     --bucket_name $BUCKET_NAME \\"
echo "     --gcs_input_path data/01-raw/housing.csv"
echo ""