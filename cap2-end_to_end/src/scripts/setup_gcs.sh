#!/bin/bash

set -e

PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="${PROJECT_ID}-cap2-end_to_end"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           SETUP GCS PARA MLOPS                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Bucket: gs://$BUCKET_NAME"
echo ""

# Crear bucket si no existe
if ! gsutil ls gs://$BUCKET_NAME 2>/dev/null; then
    echo "📦 Creando bucket..."
    gsutil mb -p $PROJECT_ID -c STANDARD -l us-central1 gs://$BUCKET_NAME
    echo "✓ Bucket creado"
else
    echo "✓ Bucket ya existe"
fi

# Crear estructura de carpetas
echo ""
echo "📁 Creando estructura de carpetas..."

# Crear archivo temporal para marcar carpetas
mkdir -p /tmp/gcs_structure
touch /tmp/gcs_structure/.gitkeep

# Data folders
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/01-raw/.gitkeep
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/02-processed/.gitkeep
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/03-features/.gitkeep
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/data/04-split/.gitkeep

# Models folders
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/models/experiments/.gitkeep
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/models/production/.gitkeep

# Artifacts
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/artifacts/.gitkeep

# Metrics
gsutil cp /tmp/gcs_structure/.gitkeep gs://$BUCKET_NAME/metrics/.gitkeep

# Limpiar
rm -rf /tmp/gcs_structure

echo "✓ Estructura creada"
echo ""
echo "Verificando estructura:"
gsutil ls gs://$BUCKET_NAME/

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           ✅ SETUP COMPLETADO                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Bucket URI: gs://$BUCKET_NAME"
echo ""
echo "Siguiente paso:"
echo "  1. Configura .env con:"
echo "     GCS_BUCKET_NAME=$BUCKET_NAME"
echo "  2. Ejecuta: make install"
echo "  3. Configura W&B: wandb login"