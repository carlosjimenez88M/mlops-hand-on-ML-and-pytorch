# 🚀 Quickstart - Ejecutar Pipeline Completo

Autor: Carlos Daniel Jiménez
Fecha: 2025-01-13

## Formas de Ejecutar el Pipeline

### ✅ Opción 1: Comando Simple (Recomendado)

```bash
python main.py
```

Este comando ejecuta TODO el pipeline de principio a fin usando la configuración en `config.yaml`.

### ✅ Opción 2: Usando el Wrapper

```bash
python run_pipeline.py
```

O directamente:
```bash
./run_pipeline.py
```

### ✅ Opción 3: Con Makefile

```bash
make run-all
```

## ⚙️ Configuración Rápida

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
# Copiar template
cp .env.example .env

# Editar con tus credenciales
nano .env
```

Asegúrate de configurar:
```bash
GCS_BUCKET_NAME=your-bucket-name
GCP_PROJECT_ID=your-project-id
WANDB_PROJECT=housing-mlops-gcp
```

### 3. Ejecutar pipeline

```bash
python main.py
```

## 📝 Ver el Pipeline en Acción

Cuando ejecutes `python main.py`, verás:

```
==================================================================
  MLOPS PIPELINE ORCHESTRATOR
==================================================================
  Project: housing-mlops-gcp
  Experiment: end_to_end_pipeline
  GCS Bucket: your-bucket-name
==================================================================

Steps to execute: 01_download_data, 02_preprocessing_and_imputation

==================================================================
  STEP 1: DOWNLOAD DATA
==================================================================
2025-01-13 10:00:00 - INFO - Downloading from: https://...
2025-01-13 10:00:05 - INFO - Downloaded to memory: 1.5 MB
2025-01-13 10:00:06 - INFO - Extracting compressed file...
2025-01-13 10:00:07 - INFO - CSV Stats: 20,640 rows, 10 columns
2025-01-13 10:00:10 - INFO - Uploaded to GCS: gs://your-bucket/data/01-raw/housing.csv

✅ Data download completed successfully!

==================================================================
  STEP 2: PREPROCESSING AND IMPUTATION
==================================================================
2025-01-13 10:00:15 - INFO - Downloading from GCS...
2025-01-13 10:00:16 - INFO - Loaded DataFrame: 20,640 rows, 10 columns
2025-01-13 10:00:17 - INFO - AUTOMATIC IMPUTATION METHOD SELECTION
2025-01-13 10:00:20 - INFO - Evaluating Simple Imputer (median)...
2025-01-13 10:00:21 - INFO - Evaluating KNN Imputer (k=5)...
2025-01-13 10:00:25 - INFO - Evaluating Iterative Imputer (RF)...
2025-01-13 10:00:35 - INFO - Best method selected: Iterative Imputer (RF)
2025-01-13 10:00:36 - INFO - Creating engineered features...
2025-01-13 10:00:37 - INFO - Created 3 new features
2025-01-13 10:00:40 - INFO - Uploaded to GCS

✅ Preprocessing and imputation completed successfully!

==================================================================
  PIPELINE EXECUTION SUMMARY
==================================================================
  ✅ All steps completed successfully!
  ⏱️  Total execution time: 45.23 seconds
  📦 GCS Bucket: gs://your-bucket-name
  📊 W&B Project: housing-mlops-gcp
==================================================================
```

## 🎯 Ejecutar Solo Algunos Pasos

Edita `config.yaml` para ejecutar solo los pasos que quieras:

```yaml
main:
  execute_steps:
    - "01_download_data"  # Solo descarga
    # - "02_preprocessing_and_imputation"  # Comentar para omitir
```

O sobrescribe desde la línea de comando:

```bash
python main.py main.execute_steps=[01_download_data]
```

## 🔧 Cambiar Configuración

### Cambiar estrategia de imputación

```bash
python main.py preprocessing.imputation_strategy=median
```

### Desactivar feature engineering

```bash
python main.py preprocessing.create_features=False
```

### Cambiar bucket de GCS

```bash
python main.py gcs.bucket_name=my-other-bucket
```

### Ejecutar experimento diferente

```bash
python main.py main.experiment_name=experiment_v2
```

## 📊 Múltiples Configuraciones

Puedes crear múltiples archivos de configuración:

```bash
# config_dev.yaml
main:
  project_name: "housing-mlops-dev"
  experiment_name: "development"

# config_prod.yaml
main:
  project_name: "housing-mlops-prod"
  experiment_name: "production"
```

Luego ejecutar:
```bash
python main.py --config-name=config_dev
python main.py --config-name=config_prod
```

## ❌ Troubleshooting

### Error: "GCS_BUCKET_NAME not found"

**Solución:**
```bash
# Verificar .env
cat .env

# Re-cargar variables
source .env

# O exportar manualmente
export GCS_BUCKET_NAME=your-bucket-name
```

### Error: "Module not found"

**Solución:**
```bash
# Re-instalar dependencias
pip install -r requirements.txt

# Verificar instalación de Hydra
pip install hydra-core omegaconf
```

### Error: "MLproject not found"

**Solución:**
Verifica que existan los archivos `MLproject` en cada módulo:
```bash
ls src/data/01_download_data/MLproject
ls src/data/02_preprocessing_and_imputation/MLproject
```

### Ver logs detallados de MLflow

```bash
# Ejecutar con logging de MLflow
python main.py --verbose
```

## 🧪 Probar Sin Ejecutar Nada

```bash
# Dry run - solo muestra la configuración
python -c "
import hydra
from omegaconf import OmegaConf

with hydra.initialize(config_path='.', version_base='1.3'):
    cfg = hydra.compose(config_name='config')
    print(OmegaConf.to_yaml(cfg))
"
```

## 📈 Ver Resultados

### En Google Cloud Storage

```bash
# Listar archivos generados
gsutil ls -lh gs://$GCS_BUCKET_NAME/data/**

# Ver datos procesados
gsutil cat gs://$GCS_BUCKET_NAME/data/02-processed/housing_processed.csv | head -20
```

### En Weights & Biases

Visita: `https://wandb.ai/your-entity/housing-mlops-gcp`

### MLflow Tracking

```bash
# Iniciar UI de MLflow
mlflow ui

# Abrir en navegador
open http://localhost:5000
```

## 🎓 Próximos Pasos

1. ✅ Pipeline básico funcionando
2. ✅ Download y preprocessing automatizados
3. 🔄 Agregar feature engineering
4. 🔄 Agregar training del modelo
5. 🔄 Agregar evaluación
6. 🔄 Agregar deployment

Para agregar nuevos pasos, edita:
- `config.yaml`: Agregar configuración del nuevo paso
- `main.py`: Agregar función para ejecutar el nuevo paso
- Crear módulo en `src/` con su `MLproject`

---

**¿Listo?** Solo ejecuta:

```bash
python main.py
```

Y observa cómo todo se ejecuta automáticamente! 🚀
