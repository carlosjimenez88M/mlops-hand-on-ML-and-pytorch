# Arquitectura y Operatividad del API de Predicción de Precios de Viviendas

## Introducción

Este documento describe en profundidad la arquitectura, funcionamiento y despliegue del **API REST de Predicción de Precios de Viviendas**, construido con FastAPI como parte de un pipeline MLOps completo. El sistema permite realizar predicciones en tiempo real sobre precios de viviendas en California utilizando modelos de Machine Learning entrenados mediante un pipeline automatizado.

---

## Tabla de Contenidos

1. [Arquitectura General](#1-arquitectura-general)
2. [Componentes del API](#2-componentes-del-api)
3. [Funcionamiento del Código](#3-funcionamiento-del-código)
4. [Dependencias de Docker](#4-dependencias-de-docker)
5. [Cómo Ejecutar el Sistema](#5-cómo-ejecutar-el-sistema)
6. [Integración con Streamlit](#6-integración-con-streamlit-propuesta)
7. [Monitoreo y Observabilidad](#7-monitoreo-y-observabilidad)

---

## 1. Arquitectura General

### 1.1 Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE ENTRENAMIENTO                     │
│                      (main.py + config.yaml)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │          MLflow + W&B Tracking              │
        │    (Experimentos, Modelos, Hyperparams)     │
        └─────────────────────────────────────────────┘
                ↓                    ↓                ↓
    ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
    │ Google Cloud  │    │ MLflow Registry  │    │ Local Storage│
    │   Storage     │    │  (Production)    │    │   (.pkl)     │
    └───────────────┘    └──────────────────┘    └──────────────┘
                                  ↓
        ┌─────────────────────────────────────────────┐
        │           FASTAPI REST API                  │
        │         (Puerto 8080)                       │
        │  ┌──────────────────────────────────────┐  │
        │  │ Lifecycle Manager (Lifespan)         │  │
        │  │  ├─ Model Loader (MLflow/GCS/Local)  │  │
        │  │  └─ W&B Logger Initialization        │  │
        │  └──────────────────────────────────────┘  │
        │                                             │
        │  Endpoints:                                 │
        │  ├─ GET  /              → Info API         │
        │  ├─ GET  /health        → Health Check     │
        │  ├─ POST /api/v1/predict → Predictions     │
        │  └─ GET  /api/v1/model/info → Model Info   │
        └─────────────────────────────────────────────┘
                         ↓
        ┌─────────────────────────────────────────────┐
        │         CLIENTES HTTP                       │
        │  ├─ Streamlit Frontend                      │
        │  ├─ curl / Postman (Testing)                │
        │  ├─ Cloud Run (Production)                  │
        │  └─ Jupyter Notebooks (Research)            │
        └─────────────────────────────────────────────┘
```

### 1.2 Stack Tecnológico

| Componente | Tecnología | Propósito |
|-----------|-----------|-----------|
| **API Framework** | FastAPI 0.104+ | REST API, validación automática, docs interactivas |
| **Server ASGI** | Uvicorn | Servidor web asíncrono de alto rendimiento |
| **Validación** | Pydantic v2 | Validación de esquemas y serialización |
| **ML Framework** | scikit-learn | Modelos de regresión (Random Forest) |
| **Model Registry** | MLflow | Versionamiento y gestión de modelos |
| **Storage** | Google Cloud Storage | Almacenamiento de modelos y datos |
| **Logging** | Weights & Biases | Tracking de predicciones y métricas |
| **Containerización** | Docker + docker-compose | Despliegue consistente |
| **Cloud Platform** | Google Cloud Run | Serverless deployment |

---

## 2. Componentes del API

### 2.1 Estructura de Directorios

```
api/
├── app/
│   ├── __init__.py
│   ├── main.py                          # Aplicación FastAPI principal
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                    # Configuración con Pydantic Settings
│   │   ├── model_loader.py              # Carga de modelos (MLflow/GCS/Local)
│   │   └── wandb_logger.py              # Logger de W&B para predicciones
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                   # Esquemas Pydantic (Request/Response)
│   └── routers/
│       ├── __init__.py
│       └── predict.py                   # Endpoint de predicciones
├── tests/                               # Tests del API
│   ├── test_api.py
│   └── test_model_loader.py
├── Dockerfile                           # Imagen Docker del API
├── requirements.txt                     # Dependencias Python
├── pytest.ini                           # Configuración de tests
└── README.md                            # Documentación del API
```

### 2.2 Módulos Core

#### 2.2.1 `app/main.py` - Aplicación FastAPI

**Responsabilidades:**
- Inicialización de la aplicación FastAPI
- Gestión del ciclo de vida (startup/shutdown)
- Configuración de CORS para desarrollo/producción
- Definición de endpoints base (`/`, `/health`)
- Manejo global de excepciones

**Características clave:**

```python
# Lifecycle Manager (Gestión del ciclo de vida)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al inicio y limpia recursos al cierre"""
    # Startup: Carga modelo + inicializa W&B
    model_loader = ModelLoader(...)
    model_loader.load_model()

    yield  # La aplicación está activa

    # Shutdown: Cierra conexiones W&B
    wandb_logger.close()
```

**Configuración CORS:** Restringida a localhost para desarrollo:
```python
allow_origins=[
    "http://localhost:3000",  # React/Streamlit
    "http://localhost:8080",  # API misma
]
```

#### 2.2.2 `app/core/config.py` - Configuración

**Sistema de configuración basado en Pydantic Settings:**

```python
class Settings(BaseSettings):
    # Información del API
    PROJECT_NAME: str = "Housing Price Prediction API"
    VERSION: str = "1.0.0"

    # Model Loading (prioridad: MLflow > GCS > Local)
    MLFLOW_MODEL_NAME: str = ""              # Nombre del modelo en MLflow
    MLFLOW_MODEL_STAGE: str = "Production"   # Stage: Production/Staging
    GCS_BUCKET: str = ""                     # Bucket de GCS
    GCS_MODEL_PATH: str = "models/..."       # Path en GCS
    LOCAL_MODEL_PATH: str = "models/..."     # Path local (fallback)

    # W&B para logging
    WANDB_API_KEY: str = ""
    WANDB_PROJECT: str = "housing-mlops-api"
```

**Variables de entorno soportadas:** `.env` file o variables del sistema.

#### 2.2.3 `app/core/model_loader.py` - Carga de Modelos

**Estrategia de carga multi-fuente con fallback:**

```
Prioridad 1: MLflow Model Registry (Production/Staging)
    ↓ (si falla)
Prioridad 2: Google Cloud Storage (GCS)
    ↓ (si falla)
Prioridad 3: Local Filesystem (.pkl)
```

**Métodos principales:**

| Método | Propósito | Source |
|--------|-----------|--------|
| `load_from_mlflow()` | Carga modelo desde MLflow Registry | `models:/{name}/{stage}` |
| `load_from_gcs()` | Descarga modelo desde GCS | `gs://{bucket}/{path}` |
| `load_from_local()` | Carga modelo desde filesystem | Path absoluto/relativo |
| `load_model()` | Auto-selección con fallback | Prioridad MLflow → GCS → Local |
| `predict()` | Inferencia sobre DataFrame | `model.predict(X)` |

**Cacheo de modelos:** Una vez cargado, el modelo se mantiene en memoria (`self._model`).

#### 2.2.4 `app/core/wandb_logger.py` - Logging a W&B

**Funcionalidad:**
- Log de predicciones individuales
- Tracking de response time
- Registro de errores de validación/predicción
- Métricas agregadas (predictions/hour, avg_price)

**Ejemplo de uso:**
```python
wandb_logger.log_prediction(
    features={"longitude": -122.23, "latitude": 37.88, ...},
    predictions=[452600.0, 385000.0],
    model_version="randomforest_v5",
    response_time_ms=23.5
)
```

#### 2.2.5 `app/models/schemas.py` - Validación Pydantic

**Esquemas de datos:**

1. **`HousingFeatures`**: Input de una vivienda
   - Campos: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, etc.
   - Validación: Rangos geográficos, valores positivos, categorías válidas

2. **`PredictionRequest`**: Request del endpoint
   - `instances: List[HousingFeatures]` (batch predictions)
   - Validación: Al menos 1 instancia

3. **`PredictionResponse`**: Response con predicciones
   - `predictions: List[PredictionResult]`
   - `model_version: str`

4. **`HealthResponse`**: Status del servicio
   - `status: str` (healthy/degraded)
   - `model_loaded: bool`
   - `version: str`

**Validación automática de `ocean_proximity`:**
```python
@field_validator('ocean_proximity')
def validate_ocean_proximity(cls, v: str) -> str:
    valid_values = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    if v.upper() not in valid_values:
        raise ValueError(f"Must be one of: {valid_values}")
    return v.upper()
```

#### 2.2.6 `app/routers/predict.py` - Endpoint de Predicción

**Endpoint principal: `POST /api/v1/predict`**

**Flujo de ejecución:**

```
1. Cliente envía POST request con JSON
   ↓
2. FastAPI valida con PredictionRequest (Pydantic)
   ↓
3. Router extrae features y convierte a DataFrame
   ↓
4. ModelLoader hace predicción (.predict(df))
   ↓
5. Resultados se formatean como PredictionResponse
   ↓
6. WandBLogger registra la predicción en W&B
   ↓
7. Response JSON enviada al cliente
```

**Manejo de errores:**
- `400 Bad Request`: Datos de entrada inválidos
- `500 Internal Server Error`: Fallos de predicción/modelo no cargado

**Endpoint adicional: `GET /api/v1/model/info`**
- Retorna: versión del modelo, status de carga

---

## 3. Funcionamiento del Código

### 3.1 Flujo de Inicialización (Startup)

```python
# 1. Cargar configuración desde variables de entorno
settings = Settings()  # Lee .env o env vars

# 2. Crear aplicación FastAPI con lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan  # Lifecycle manager
)

# 3. En lifespan startup:
async def lifespan(app: FastAPI):
    # 3.1 Inicializar W&B Logger
    wandb_logger = WandBLogger(
        project=settings.WANDB_PROJECT,
        enabled=True
    )

    # 3.2 Inicializar ModelLoader
    model_loader = ModelLoader(
        local_model_path=settings.LOCAL_MODEL_PATH,
        gcs_bucket=settings.GCS_BUCKET,
        mlflow_model_name=settings.MLFLOW_MODEL_NAME,
        mlflow_model_stage=settings.MLFLOW_MODEL_STAGE
    )

    # 3.3 Cargar modelo (con fallback automático)
    try:
        model_loader.load_model()
        logger.info(f"Model loaded: {model_loader.model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # API inicia pero las predicciones fallarán

    # 3.4 Almacenar en app.state para acceso global
    app.state.model_loader = model_loader
    app.state.wandb_logger = wandb_logger

    yield  # Aplicación en ejecución

    # 4. En shutdown: Cerrar conexiones
    wandb_logger.close()
```

### 3.2 Flujo de Predicción (Request → Response)

**Ejemplo de request:**

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
      }
    ]
  }'
```

**Flujo interno:**

```python
# 1. FastAPI recibe request y valida con PydanticRequest
request: PredictionRequest = {
    "instances": [HousingFeatures(...)]
}

# 2. Router convierte a DataFrame
features_list = []
for instance in request.instances:
    features_list.append({
        'longitude': instance.longitude,
        'latitude': instance.latitude,
        # ... resto de features
    })
df = pd.DataFrame(features_list)

# 3. ModelLoader predice
predictions = model_loader.predict(df)
# Output: array([452600., 385000., ...])

# 4. Formato de response
results = [
    PredictionResult(
        predicted_price=float(pred),
        confidence_interval=None
    )
    for pred in predictions
]

# 5. Log a W&B (asíncrono)
wandb_logger.log_prediction(
    features=features_list,
    predictions=[float(p) for p in predictions],
    model_version=model_loader.model_version,
    response_time_ms=23.5
)

# 6. Return response
return PredictionResponse(
    predictions=results,
    model_version="randomforest_v5"
)
```

**Response:**

```json
{
  "predictions": [
    {
      "predicted_price": 452600.0,
      "confidence_interval": null
    }
  ],
  "model_version": "randomforest_v5"
}
```

### 3.3 Manejo de Errores

**Niveles de manejo:**

1. **Validación de entrada (Pydantic):**
   ```python
   # Si ocean_proximity es inválido:
   raise ValueError("Must be one of: <1H OCEAN, INLAND, ...")
   # FastAPI convierte a 422 Unprocessable Entity
   ```

2. **Errores de negocio (HTTPException):**
   ```python
   if not model_loader.is_loaded:
       raise HTTPException(
           status_code=500,
           detail="Model not loaded"
       )
   ```

3. **Excepciones no controladas (Global Handler):**
   ```python
   @app.exception_handler(Exception)
   async def global_exception_handler(_request, exc):
       logger.error(f"Unhandled error: {exc}", exc_info=True)
       return JSONResponse(
           status_code=500,
           content={"error": "Internal server error"}
       )
   ```

---

## 4. Dependencias de Docker

### 4.1 Dockerfile del API

**Ubicación:** `api/Dockerfile`

```dockerfile
FROM python:3.12-slim

# Metadata
LABEL maintainer="danieljimenez88m@gmail.com"
LABEL description="Housing Price Prediction API - FastAPI Service"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create models directory (para volumen de docker-compose)
RUN mkdir -p models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

EXPOSE 8080

# Health check (cada 30s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
```

**Características:**
- **Base image:** `python:3.12-slim` (pequeña, segura)
- **Layer caching:** Requirements se instalan antes del código
- **Health check:** Endpoint `/health` monitoreado cada 30s
- **Non-root user:** (recomendado para producción)

### 4.2 docker-compose.yaml

**Propósito:** Orquestar servicios localmente (API + futuro Streamlit)

```yaml
services:
  # FastAPI Model Prediction Service
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    container_name: housing-price-api
    ports:
      - "8080:8080"  # Host:Container
    environment:
      # Model loading - Local prioritario en Docker
      - LOCAL_MODEL_PATH=/app/models/trained/housing_price_model.pkl
      # Optional: GCS/MLflow si se configuran
      - GCS_BUCKET=${GCS_BUCKET_NAME:-}
      - MLFLOW_MODEL_NAME=${MLFLOW_MODEL_NAME:-}
      # W&B para logging
      - WANDB_API_KEY=${WANDB_API_KEY}
      - WANDB_PROJECT=${WANDB_PROJECT:-housing-mlops-api}
    env_file:
      - .env  # Variables adicionales
    volumes:
      # Mount código para development (hot reload)
      - ./api/app:/app/app:ro
      # Mount modelos (CRÍTICO: debe existir el .pkl)
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mlops-network
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G

networks:
  mlops-network:
    driver: bridge
```

**Volúmenes críticos:**
1. `./api/app:/app/app:ro` → Código del API (read-only)
2. `./models:/app/models:ro` → Modelos entrenados (read-only)

**Ventajas:**
- Hot reload en desarrollo
- Aislamiento de red
- Resource limits para prevenir consumo excesivo

### 4.3 Dependencias Python

**`api/requirements.txt`:**

```txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Data validation
pydantic==2.5.0
pydantic-settings==2.1.0

# ML Libraries
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2

# Model Management
mlflow==2.10.0

# Cloud Storage
google-cloud-storage==2.14.0

# Monitoring
wandb==0.16.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0
httpx==0.25.2  # Para tests de FastAPI
```

**Notas:**
- `uvicorn[standard]`: Incluye optimizaciones (httptools, websockets)
- `httpx`: Cliente HTTP asíncrono para tests
- Versiones fijas para reproducibilidad

---

## 5. Cómo Ejecutar el Sistema

### 5.1 Requisitos Previos

```bash
# 1. Python 3.12+
python --version  # Debe ser >= 3.12

# 2. Docker (opcional)
docker --version

# 3. Variables de entorno (crear .env en la raíz)
cat > .env << EOF
# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_NAME=your-bucket-name

# W&B Configuration
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=housing-mlops-api

# MLflow (opcional)
MLFLOW_MODEL_NAME=housing-price-model
MLFLOW_MODEL_STAGE=Production
MLFLOW_TRACKING_URI=http://localhost:5000
EOF

# 4. Modelo entrenado en models/trained/housing_price_model.pkl
ls -lh models/trained/housing_price_model.pkl
```

### 5.2 Ejecución Local (Sin Docker)

#### Opción 1: Usando Makefile

```bash
# 1. Instalar dependencias del API
make api-install

# 2. Ejecutar API localmente con hot reload
make api-local

# Output:
# Starting API locally on port 8080...
# API will be available at http://localhost:8080
# API docs at http://localhost:8080/docs

# 3. En otra terminal, probar el API
curl http://localhost:8080/health
```

#### Opción 2: Manual

```bash
# 1. Entrar al directorio del API
cd api

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
export LOCAL_MODEL_PATH=../models/trained/housing_price_model.pkl
export WANDB_API_KEY=your-key

# 4. Ejecutar con uvicorn
uvicorn app.main:app --reload --port 8080

# 5. Acceder a:
# - API: http://localhost:8080
# - Docs interactivas: http://localhost:8080/docs
# - ReDoc: http://localhost:8080/redoc
```

### 5.3 Ejecución con Docker

#### Opción 1: docker-compose (Recomendado)

```bash
# 1. Construir imágenes y levantar servicios
make compose-up

# Output:
# Starting services with docker-compose...
# [+] Running 2/2
#  ✔ Network housing-mlops-network  Created
#  ✔ Container housing-price-api    Started
# API available at http://localhost:8080

# 2. Ver logs
docker-compose logs -f api

# 3. Detener servicios
make compose-down
```

#### Opción 2: Docker manual

```bash
# 1. Construir imagen del API
make api-build

# 2. Ejecutar contenedor
docker run -d \
  --name housing-api \
  -p 8080:8080 \
  -e LOCAL_MODEL_PATH=/app/models/trained/housing_price_model.pkl \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $(pwd)/models:/app/models:ro \
  housing-api:latest

# 3. Ver logs
docker logs -f housing-api

# 4. Detener
docker stop housing-api && docker rm housing-api
```

### 5.4 Testing del API

#### Health Check

```bash
# Test básico
curl http://localhost:8080/health

# Response esperada:
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Predicción Individual

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
      }
    ]
  }'

# Response:
{
  "predictions": [
    {
      "predicted_price": 452600.0,
      "confidence_interval": null
    }
  ],
  "model_version": "housing_price_model"
}
```

#### Predicción Batch (Múltiples viviendas)

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
      },
      {
        "longitude": -118.30,
        "latitude": 34.26,
        "housing_median_age": 15.0,
        "total_rooms": 5612.0,
        "total_bedrooms": 1283.0,
        "population": 1015.0,
        "households": 472.0,
        "median_income": 1.4936,
        "ocean_proximity": "INLAND"
      }
    ]
  }'
```

#### Tests Automatizados

```bash
# Ejecutar tests del API
make api-test

# Tests con coverage
make api-test-cov

# Output:
# ==================== test session starts ====================
# tests/test_api.py::test_health_check PASSED          [ 25%]
# tests/test_api.py::test_predict_success PASSED       [ 50%]
# tests/test_api.py::test_predict_invalid_input PASSED [ 75%]
# tests/test_model_loader.py::test_load_local PASSED   [100%]
# ==================== 4 passed in 2.31s =====================
```

### 5.5 Acceso a Documentación Interactiva

FastAPI genera automáticamente documentación interactiva:

1. **Swagger UI (OpenAPI):**
   - URL: http://localhost:8080/docs
   - Permite probar endpoints directamente
   - Muestra esquemas de request/response

2. **ReDoc:**
   - URL: http://localhost:8080/redoc
   - Documentación más formal y legible

3. **OpenAPI JSON:**
   - URL: http://localhost:8080/openapi.json
   - Especificación completa en JSON

---

## 6. Integración con Streamlit (Propuesta)

### 6.1 Arquitectura Propuesta

```
┌─────────────────────────────────────────────────┐
│        STREAMLIT FRONTEND (Puerto 8501)         │
│  ┌──────────────────────────────────────────┐  │
│  │  Interfaz de Usuario                     │  │
│  │  ├─ Formulario de entrada (features)     │  │
│  │  ├─ Botón "Predict Price"                │  │
│  │  └─ Visualización de resultados          │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                     ↓ HTTP POST
        ┌────────────────────────────────┐
        │   FASTAPI REST API (8080)     │
        │   POST /api/v1/predict        │
        └────────────────────────────────┘
                     ↓
        ┌────────────────────────────────┐
        │   Modelo ML (scikit-learn)    │
        │   Random Forest Regressor     │
        └────────────────────────────────┘
```

### 6.2 Implementación de Streamlit

**Archivo:** `streamlit_app/app.py` (a crear)

```python
import streamlit as st
import requests
import pandas as pd

# Configuración
API_URL = "http://localhost:8080/api/v1/predict"

st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 California Housing Price Predictor")
st.markdown("Predict median house values based on location and features")

# Sidebar para inputs
with st.sidebar:
    st.header("📊 House Features")

    # Ubicación
    st.subheader("Location")
    longitude = st.number_input("Longitude", value=-122.23, format="%.2f")
    latitude = st.number_input("Latitude", value=37.88, format="%.2f")

    # Características del área
    st.subheader("Area Characteristics")
    housing_median_age = st.slider("Median Age (years)", 1, 52, 41)
    total_rooms = st.number_input("Total Rooms", value=880, step=100)
    total_bedrooms = st.number_input("Total Bedrooms", value=129, step=10)
    population = st.number_input("Population", value=322, step=50)
    households = st.number_input("Households", value=126, step=10)
    median_income = st.number_input("Median Income (×$10k)", value=8.3252, format="%.4f")

    # Proximidad al océano
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    )

# Botón de predicción
if st.button("🔮 Predict Price", type="primary"):
    # Construir payload
    payload = {
        "instances": [
            {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity
            }
        ]
    }

    # Hacer request al API
    with st.spinner("Making prediction..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()

            result = response.json()
            prediction = result["predictions"][0]["predicted_price"]
            model_version = result["model_version"]

            # Mostrar resultado
            st.success("✅ Prediction Complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Predicted Median House Value",
                    value=f"${prediction:,.2f}",
                    delta=None
                )
            with col2:
                st.info(f"**Model Version:** {model_version}")

            # Mostrar features usadas
            with st.expander("📋 Input Features"):
                df = pd.DataFrame([payload["instances"][0]])
                st.dataframe(df.T, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to API: {str(e)}")
        except KeyError as e:
            st.error(f"❌ Invalid response format: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by FastAPI + scikit-learn + MLflow")
```

### 6.3 Dockerfile para Streamlit

**Archivo:** `streamlit_app/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**`streamlit_app/requirements.txt`:**

```txt
streamlit==1.31.0
requests==2.31.0
pandas==2.1.3
```

### 6.4 Actualización de docker-compose.yaml

```yaml
services:
  # Servicio API (existente)
  api:
    # ... configuración existente ...

  # Nuevo servicio Streamlit
  streamlit:
    build:
      context: ./streamlit_app
      dockerfile: Dockerfile
    container_name: housing-streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8080/api/v1/predict
    depends_on:
      - api
    networks:
      - mlops-network
    restart: unless-stopped

networks:
  mlops-network:
    driver: bridge
```

**Nota:** El servicio Streamlit se comunica con el API usando el nombre del servicio (`api`) en la red interna de Docker.

---

## 7. Monitoreo y Observabilidad

### 7.1 Logs Estructurados

**Formato:**
```
2024-01-13 18:45:23,123 - app.main - INFO - Starting up API...
2024-01-13 18:45:24,567 - app.core.model_loader - INFO - Loading model from local path: models/trained/housing_price_model.pkl
2024-01-13 18:45:25,890 - app.core.model_loader - INFO - Model loaded successfully: housing_price_model
2024-01-13 18:45:26,012 - uvicorn - INFO - Application startup complete
```

**Acceso a logs:**
```bash
# Docker Compose
docker-compose logs -f api

# Container individual
docker logs -f housing-price-api

# Local
# Los logs se imprimen en stdout
```

### 7.2 Métricas en W&B

**Dashboard automático:**
- URL: https://wandb.ai/<entity>/housing-mlops-api

**Métricas trackeadas:**
- `predictions/count`: Total de predicciones
- `predictions/avg_price`: Precio promedio predicho
- `predictions/response_time_ms`: Latencia promedio
- `predictions/errors`: Conteo de errores

**Visualización:**
```python
# En W&B dashboard:
- Line chart: response_time_ms vs timestamp
- Histogram: Distribución de precios predichos
- Table: Features + predicciones recientes
```

### 7.3 Health Checks

**Endpoint:** `GET /health`

**Checks realizados:**
1. API está respondiendo
2. Modelo cargado en memoria
3. Version del API

**Integración con Docker:**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s    # Cada 30 segundos
  timeout: 10s     # Timeout de 10s
  retries: 3       # 3 intentos antes de marcar como unhealthy
  start_period: 40s # Grace period de 40s al inicio
```

**Kubernetes/Cloud Run:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

### 7.4 Tracing con MLflow

**Tracking de modelos:**
- Cada carga de modelo loguea: fuente, versión, timestamp
- MLflow UI: http://localhost:5000 (si corre localmente)

**Queries útiles:**
```bash
# Ver experimentos
mlflow experiments list

# Ver runs de un experimento
mlflow runs list --experiment-id 1

# Ver modelos registrados
mlflow models list

# Comparar versiones
mlflow models compare housing-price-model
```

---

## 8. Despliegue en Producción

### 8.1 Google Cloud Run

**Pasos:**

```bash
# 1. Autenticarse en GCP
gcloud auth login
gcloud config set project <PROJECT_ID>

# 2. Build y push a Container Registry
cd api
gcloud builds submit --tag gcr.io/<PROJECT_ID>/housing-api:latest

# 3. Deploy a Cloud Run
gcloud run deploy housing-price-api \
  --image gcr.io/<PROJECT_ID>/housing-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars "LOCAL_MODEL_PATH=/app/models/trained/housing_price_model.pkl" \
  --set-env-vars "WANDB_API_KEY=<KEY>" \
  --set-env-vars "WANDB_PROJECT=housing-mlops-api-prod"

# 4. Obtener URL
gcloud run services describe housing-price-api --region us-central1 --format 'value(status.url)'
```

**Configuración recomendada:**
- **Min instances:** 1 (evita cold starts)
- **Max instances:** 10 (auto-scaling)
- **Memory:** 2Gi (modelo en memoria)
- **CPU:** 2 (predicciones rápidas)
- **Timeout:** 300s

### 8.2 CI/CD con GitHub Actions

**Workflow:** `.github/workflows/deploy-api.yaml`

```yaml
name: Deploy API to Cloud Run

on:
  push:
    branches:
      - main
    paths:
      - 'api/**'

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Build and push Docker image
        run: |
          cd api
          gcloud builds submit --tag gcr.io/${{ secrets.GCP_PROJECT }}/housing-api:${{ github.sha }}

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy housing-price-api \
            --image gcr.io/${{ secrets.GCP_PROJECT }}/housing-api:${{ github.sha }} \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated
```

---

## 9. Conclusiones

### 9.1 Resumen de Capacidades

El sistema implementado proporciona:

1. **API REST robusta:**
   - Validación automática de inputs (Pydantic)
   - Documentación interactiva (OpenAPI)
   - Manejo de errores completo
   - Health checks integrados

2. **Gestión de modelos flexible:**
   - Carga desde múltiples fuentes (MLflow/GCS/Local)
   - Fallback automático
   - Cacheo en memoria

3. **Observabilidad completa:**
   - Logs estructurados
   - Métricas en W&B
   - Health checks
   - Tracing de modelos

4. **Deployment ready:**
   - Dockerizado
   - docker-compose para desarrollo
   - CI/CD para Cloud Run
   - Auto-scaling

### 9.2 Próximos Pasos

1. **Implementar Streamlit:** Frontend interactivo para usuarios finales
2. **Autenticación:** JWT tokens para API en producción
3. **Rate limiting:** Protección contra abuso
4. **A/B Testing:** Comparar múltiples versiones de modelos
5. **Batch predictions:** Endpoint para procesamiento masivo
6. **Cache:** Redis para predicciones frecuentes

---

## Referencias

- **FastAPI Docs:** https://fastapi.tiangolo.com
- **MLflow Docs:** https://mlflow.org/docs/latest
- **Pydantic Docs:** https://docs.pydantic.dev
- **Docker Compose:** https://docs.docker.com/compose
- **Google Cloud Run:** https://cloud.google.com/run/docs
- **W&B Docs:** https://docs.wandb.ai

---

**Autor:** Carlos Daniel Jiménez
**Fecha:** Enero 2024
**Versión:** 1.0.0
**Licencia:** MIT
