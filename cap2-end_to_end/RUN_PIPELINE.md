# Guía de Ejecución del Pipeline MLOps

Autor: Carlos Daniel Jiménez
Fecha: 2025-01-13

## Tabla de Contenidos

- [Requisitos Previos](#requisitos-previos)
- [Configuración Inicial](#configuración-inicial)
- [Ejecución del Pipeline](#ejecución-del-pipeline)
- [Comandos Disponibles](#comandos-disponibles)
- [Troubleshooting](#troubleshooting)

## Requisitos Previos

1. **Python 3.9+** instalado
2. **Google Cloud SDK** configurado
3. **Credenciales de GCP** con acceso a Google Cloud Storage
4. **Cuenta de Weights & Biases** (opcional, para tracking)

## Configuración Inicial

### 1. Instalar Dependencias

```bash
# Opción 1: Usando Makefile (recomendado)
make install

# Opción 2: Manualmente
pip install -r requirements.txt
pip install pytest pytest-cov
```

### 2. Configurar Variables de Entorno

```bash
# Crear archivo .env desde el template
make setup-env

# O manualmente
cp .env.example .env

# Editar .env con tus credenciales
nano .env
```

**Configurar en `.env`:**

```bash
# GCP Configuration
GCS_BUCKET_NAME=your-bucket-name
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Weights & Biases
WANDB_PROJECT=housing-mlops-gcp
WANDB_ENTITY=your-wandb-username
WANDB_API_KEY=your-wandb-api-key
```

### 3. Autenticar con Google Cloud

```bash
# Configurar credenciales
gcloud auth application-default login

# O usar service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

## Ejecución del Pipeline

### Opción 1: Pipeline Completo (Recomendado)

Ejecuta todos los pasos en secuencia automáticamente:

```bash
# Usando Makefile
make run-all

# O directamente
bash scripts/run_pipeline.sh
```

**Esto ejecutará:**
1. Download de datos desde la fuente
2. Preprocessing e imputación automática
3. Feature engineering
4. Validación de resultados

### Opción 2: Ejecutar Módulos Individuales

#### 2.1 Solo Download de Datos

```bash
# Opción A: Usando Makefile
make run-download

# Opción B: Usando script
bash run_download.sh

# Opción C: Directamente con Python
python src/data/01_download_data/main.py \
  --file_url "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz" \
  --artifact_name "housing_data_raw" \
  --artifact_type "raw_data" \
  --bucket_name "$GCS_BUCKET_NAME" \
  --gcs_output_path "data/01-raw/housing.csv"
```

#### 2.2 Solo Preprocessing

```bash
# Opción A: Usando Makefile
make run-preprocess

# Opción B: Usando script
bash scripts/run_preprocessing.sh

# Opción C: Directamente con Python
python src/data/02_preprocessing_and_imputation/main.py \
  --input_artifact_name "housing_data_raw:latest" \
  --gcs_input_path "data/01-raw/housing.csv" \
  --gcs_output_path "data/02-processed/housing_processed.csv" \
  --bucket_name "$GCS_BUCKET_NAME" \
  --imputation_strategy "auto" \
  --create_features "True"
```

## Comandos Disponibles

### Mostrar Ayuda

```bash
make help
```

### Pipeline

```bash
make run-all           # Ejecutar pipeline completo
make run-download      # Solo download
make run-preprocess    # Solo preprocessing
```

### Testing

```bash
make test              # Ejecutar todos los tests
make coverage          # Tests con reporte de cobertura
make test-download     # Solo tests de download
```

### Calidad de Código

```bash
make lint              # Verificar linting
make format            # Formatear código
make type-check        # Verificar tipos
```

### Utilidades

```bash
make show-env          # Mostrar variables de entorno
make clean             # Limpiar archivos temporales
make clean-cache       # Limpiar caché de Python
```

## Estructura de Salida

Después de ejecutar el pipeline, los datos estarán en GCS:

```
gs://your-bucket-name/
├── data/
│   ├── 01-raw/
│   │   └── housing.csv          # Datos originales
│   └── 02-processed/
│       └── housing_processed.csv # Datos procesados
```

## Verificar Resultados

### 1. En Google Cloud Storage

```bash
# Listar archivos
gsutil ls -lh gs://$GCS_BUCKET_NAME/data/**

# Ver primeras líneas del archivo procesado
gsutil cat gs://$GCS_BUCKET_NAME/data/02-processed/housing_processed.csv | head -20
```

### 2. En Weights & Biases

Visita: `https://wandb.ai/your-entity/housing-mlops-gcp`

### 3. Reporte de Cobertura de Tests

```bash
# Generar reporte
make coverage

# Abrir reporte HTML
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Troubleshooting

### Error: "GCS_BUCKET_NAME not set"

**Solución:**
```bash
# Verificar que .env existe
ls -la .env

# Verificar contenido
cat .env

# Re-cargar variables
source .env
```

### Error: "Bucket does not exist"

**Solución:**
```bash
# Crear bucket
gsutil mb -p $GCP_PROJECT_ID -l $GCP_REGION gs://$GCS_BUCKET_NAME

# O verificar nombre
gsutil ls
```

### Error: "Permission denied"

**Solución:**
```bash
# Dar permisos a scripts
chmod +x scripts/*.sh
chmod +x run_download.sh

# Verificar autenticación GCP
gcloud auth list
gcloud auth application-default login
```

### Error: "Module not found"

**Solución:**
```bash
# Re-instalar dependencias
make install

# Verificar Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Error en tests: "Import errors"

**Solución:**
```bash
# Limpiar caché
make clean-cache

# Re-ejecutar tests
make test
```

## Ejemplos de Uso

### 1. Primera Ejecución

```bash
# Setup completo
make setup-env
# Editar .env con tus credenciales
nano .env

# Instalar
make install

# Verificar configuración
make show-env

# Ejecutar pipeline
make run-all
```

### 2. Desarrollo con Tests

```bash
# Hacer cambios en el código
vim src/data/01_download_data/downloader.py

# Ejecutar tests
make test

# Ver cobertura
make coverage

# Si todo bien, ejecutar pipeline
make run-all
```

### 3. Solo Preprocessing

```bash
# Si ya tienes los datos descargados
make run-preprocess
```

## Logs y Debugging

### Ver logs detallados

```bash
# Ejecutar con más verbosidad
bash -x scripts/run_pipeline.sh

# O para módulo específico
python -v src/data/01_download_data/main.py --help
```

### Activar modo debug en Python

Editar el archivo main.py y agregar:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Próximos Pasos

Después de ejecutar el pipeline exitosamente:

1. ✅ Verificar datos en GCS
2. ✅ Revisar métricas en W&B
3. ✅ Ejecutar tests: `make coverage`
4. 🔄 Continuar con feature engineering
5. 🔄 Entrenar modelo
6. 🔄 Evaluar y deployar

## Soporte

Para problemas o preguntas:

1. Revisar logs en la terminal
2. Verificar configuración en `.env`
3. Consultar documentación de cada módulo
4. Revisar tests para ejemplos de uso

---

**Última actualización:** 2025-01-13
**Autor:** Carlos Daniel Jiménez
