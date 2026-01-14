# Guía de Inicio Rápido - Housing Price Prediction System

## Introducción

Este sistema completo de MLOps incluye:
- **Pipeline de entrenamiento**: 7 pasos automatizados (descarga → procesamiento → entrenamiento → registro)
- **API REST (FastAPI)**: Servicio de predicción HTTP en puerto 8080
- **Frontend Web (Streamlit)**: Interfaz interactiva en puerto 8501
- **Orquestación**: Docker Compose para desarrollo local
- **Deployment**: Ready para Google Cloud Run

---

## Tabla de Contenidos

1. [Instalación Rápida](#1-instalación-rápida)
2. [Opción A: Ejecución con Docker Compose (Recomendado)](#2-opción-a-ejecución-con-docker-compose-recomendado)
3. [Opción B: Ejecución Local sin Docker](#3-opción-b-ejecución-local-sin-docker)
4. [Comandos Makefile Disponibles](#4-comandos-makefile-disponibles)
5. [Arquitectura del Sistema](#5-arquitectura-del-sistema)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Instalación Rápida

### Prerequisitos

```bash
# Verificar versiones
python --version   # >= 3.12
docker --version   # >= 20.10
docker-compose --version  # >= 2.0
make --version     # GNU Make
```

### Clonar y Configurar

```bash
# 1. Clonar repositorio (si no lo has hecho)
cd mlops-hand-on-ML-and-pytorch/cap2-end_to_end

# 2. Crear archivo .env con tus credenciales
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

# 3. Verificar que existe el modelo entrenado
ls -lh models/trained/housing_price_model.pkl

# Si NO existe, ejecutar el pipeline primero:
# make run-pipeline
```

---

## 2. Opción A: Ejecución con Docker Compose (Recomendado)

### Inicio Rápido (1 comando)

```bash
# Iniciar API + Streamlit con docker-compose
make compose-up
```

**Output esperado:**
```
Starting services with docker-compose...
[+] Running 3/3
 ✔ Network housing-mlops-network  Created
 ✔ Container housing-price-api    Started
 ✔ Container housing-streamlit    Started

===================================================================
  Services are now running!
===================================================================

🌐 API:        http://localhost:8080
   - Docs:     http://localhost:8080/docs
   - Health:   curl http://localhost:8080/health

🎨 Streamlit:  http://localhost:8501

To view logs: make compose-logs
To stop:      make compose-down
===================================================================
```

### Verificar Servicios

```bash
# 1. Verificar estado de contenedores
docker ps

# Deberías ver:
# - housing-price-api (puerto 8080)
# - housing-streamlit (puerto 8501)

# 2. Test del API
curl http://localhost:8080/health

# Response esperada:
# {"status":"healthy","model_loaded":true,"version":"1.0.0"}

# 3. Abrir Streamlit en navegador
open http://localhost:8501
```

### Ver Logs en Tiempo Real

```bash
# Ver logs de todos los servicios
make compose-logs

# Ver logs solo del API
docker-compose logs -f api

# Ver logs solo de Streamlit
docker-compose logs -f streamlit
```

### Detener Servicios

```bash
# Detener y eliminar contenedores
make compose-down

# Output:
# Stopping docker-compose services...
# [+] Running 3/3
#  ✔ Container housing-streamlit  Removed
#  ✔ Container housing-price-api  Removed
#  ✔ Network housing-mlops-network  Removed
```

---

## 3. Opción B: Ejecución Local sin Docker

Esta opción es útil para desarrollo con hot reload.

### Paso 1: Instalar Dependencias

```bash
# Instalar dependencias del API
make api-install

# Instalar dependencias de Streamlit
make streamlit-install
```

### Paso 2: Opción 2A - Ejecutar Ambos Servicios en Paralelo

```bash
# Ejecutar API + Streamlit simultáneamente
make run-full-stack
```

Este comando ejecutará:
- API en http://localhost:8080 (con hot reload)
- Streamlit en http://localhost:8501 (con hot reload)

**Ventajas:**
- Hot reload automático cuando editas código
- Logs visibles en la misma terminal
- Ideal para desarrollo

**Para detener:** Presiona `Ctrl+C`

### Paso 2: Opción 2B - Ejecutar Servicios en Terminales Separadas

#### Terminal 1: API

```bash
# Ejecutar API localmente
make api-local
```

**Output:**
```
Starting API locally on port 8080...
API will be available at http://localhost:8080
API docs at http://localhost:8080/docs
INFO:     Uvicorn running on http://0.0.0.0:8080
INFO:     Application startup complete
```

#### Terminal 2: Streamlit

```bash
# Ejecutar Streamlit localmente
make streamlit-local
```

**Output:**
```
Starting Streamlit locally on port 8501...
Streamlit will be available at http://localhost:8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501
```

---

## 4. Comandos Makefile Disponibles

### Ver Todos los Comandos

```bash
make help
```

### Pipeline de Entrenamiento

```bash
# Ejecutar pipeline completo (7 pasos)
make run-pipeline

# Ejecutar solo descarga de datos
make run-download

# Ejecutar solo preprocessing
make run-preprocessing

# Ejecutar hyperparameter sweep (W&B)
make run-sweep
```

### Comandos del API

```bash
# Instalar dependencias
make api-install

# Ejecutar localmente (puerto 8080)
make api-local

# Ejecutar tests
make api-test

# Ejecutar tests con coverage
make api-test-cov

# Build Docker image
make api-build
```

### Comandos de Streamlit

```bash
# Instalar dependencias
make streamlit-install

# Ejecutar localmente (puerto 8501)
make streamlit-local

# Build Docker image
make streamlit-build
```

### Comandos de Docker Compose

```bash
# Iniciar todos los servicios
make compose-up

# Ver logs
make compose-logs

# Detener servicios
make compose-down
```

### Comandos de Testing

```bash
# Ejecutar tests del pipeline
make run-tests

# Tests con coverage
make test-cov

# Tests del API
make api-test

# Tests con coverage del API
make api-test-cov
```

### Comandos de Desarrollo

```bash
# Lint código
make lint

# Format código
make format

# Limpiar archivos temporales
make clean
```

---

## 5. Arquitectura del Sistema

### Flujo Completo

```
┌─────────────────────────────────────────────────────────────┐
│               FASE 1: ENTRENAMIENTO (main.py)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 01. Download Data        → GCS                       │  │
│  │ 02. Preprocessing        → Imputación                │  │
│  │ 03. Feature Engineering  → K-Means clustering        │  │
│  │ 04. Data Segregation     → Train/Test split          │  │
│  │ 05. Model Selection      → 5 algoritmos              │  │
│  │ 06. Hyperparameter Sweep → W&B Bayesian             │  │
│  │ 07. Model Registration   → MLflow + Local PKL       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             ↓
        ┌────────────────────────────────────────┐
        │    MODELO ENTRENADO (.pkl)             │
        │    models/trained/                     │
        │    housing_price_model.pkl             │
        └────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│               FASE 2: SERVING (API + Streamlit)             │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Usuario → Streamlit (Puerto 8501)            │   │
│  │         - Formulario de entrada                      │   │
│  │         - Visualizaciones                            │   │
│  │         - Resultados interactivos                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                        ↓ HTTP POST                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         FastAPI REST API (Puerto 8080)               │   │
│  │         - POST /api/v1/predict                       │   │
│  │         - GET /health                                │   │
│  │         - GET /docs (Swagger UI)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                        ↓ Inferencia                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Modelo ML (Random Forest)                    │   │
│  │         - Predicción                                 │   │
│  │         - Logging a W&B                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Comunicación entre Servicios

#### En Docker Compose:
```
Streamlit (container: housing-streamlit)
    ↓
  API_URL=http://api:8080  (nombre del servicio en red interna)
    ↓
FastAPI (container: housing-price-api)
    ↓
Modelo en /app/models/trained/housing_price_model.pkl (volumen montado)
```

#### En Local:
```
Streamlit (localhost:8501)
    ↓
  API_URL=http://localhost:8080
    ↓
FastAPI (localhost:8080)
    ↓
Modelo en models/trained/housing_price_model.pkl
```

---

## 6. Uso de la Aplicación Streamlit

### Acceso

```bash
# Abrir en navegador
open http://localhost:8501

# O visitar manualmente:
http://localhost:8501
```

### Flujo de Uso

1. **Verificar Status del API**
   - Sidebar muestra estado: ✅ API is healthy
   - Si el modelo está cargado: **Model:** housing_price_model

2. **Ingresar Features de la Vivienda**
   - **Location**: Longitude y Latitude
   - **Area Characteristics**: Edad, rooms, bedrooms, población, households, income
   - **Ocean Proximity**: Seleccionar de dropdown

3. **Ver Visualización de Ubicación**
   - Mapa interactivo muestra la ubicación geográfica
   - Zoom y pan disponibles

4. **Presionar "Predict House Price"**
   - El sistema envía request al API
   - Muestra spinner mientras procesa

5. **Interpretar Resultados**
   - **Predicted Median House Value**: Precio predicho
   - **Feature Comparison**: Radar chart comparando con valores típicos
   - **Price Context**: Métricas derivadas (price per room, etc.)
   - **Detailed Analysis**: Expander con interpretación

### Ejemplos de Predicción

#### Ejemplo 1: San Francisco Bay Area (Precio Alto)

```
Longitude: -122.23
Latitude: 37.88
Housing Median Age: 41
Total Rooms: 880
Total Bedrooms: 129
Population: 322
Households: 126
Median Income: 8.3252
Ocean Proximity: NEAR BAY

Predicción esperada: ~$450,000 - $500,000
```

#### Ejemplo 2: Los Angeles Inland (Precio Medio)

```
Longitude: -118.30
Latitude: 34.26
Housing Median Age: 15
Total Rooms: 5612
Total Bedrooms: 1283
Population: 1015
Households: 472
Median Income: 1.4936
Ocean Proximity: INLAND

Predicción esperada: ~$150,000 - $200,000
```

#### Ejemplo 3: San Diego Coastal (Precio Alto)

```
Longitude: -117.15
Latitude: 32.75
Housing Median Age: 35
Total Rooms: 3000
Total Bedrooms: 600
Population: 800
Households: 550
Median Income: 5.5
Ocean Proximity: <1H OCEAN

Predicción esperada: ~$300,000 - $350,000
```

---

## 7. Testing del Sistema Completo

### Test End-to-End

#### 1. Verificar Health del API

```bash
curl http://localhost:8080/health
```

**Response esperada:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### 2. Test de Predicción via API

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

**Response esperada:**
```json
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

#### 3. Test de Streamlit

1. Abrir http://localhost:8501
2. Verificar que sidebar muestra "✅ API is healthy"
3. Ingresar features del ejemplo 1
4. Presionar "Predict House Price"
5. Verificar que aparece resultado con precio predicho

---

## 8. Troubleshooting

### Problema: "API is unavailable" en Streamlit

**Síntomas:**
- Streamlit muestra "❌ API is unavailable"
- Error: Connection refused

**Solución:**

```bash
# 1. Verificar que el API está corriendo
curl http://localhost:8080/health

# Si falla:

# Para Docker Compose:
docker-compose logs api
docker-compose restart api

# Para ejecución local:
# Verificar en la terminal del API si hay errores
# Reiniciar: Ctrl+C y luego make api-local
```

### Problema: "Model not loaded"

**Síntomas:**
- Health check muestra `"model_loaded": false`
- Predicciones fallan con 500 error

**Solución:**

```bash
# 1. Verificar que existe el archivo del modelo
ls -lh models/trained/housing_price_model.pkl

# Si NO existe:
# Ejecutar pipeline para entrenar modelo
make run-pipeline

# 2. Verificar variables de entorno
echo $LOCAL_MODEL_PATH

# Debe ser: models/trained/housing_price_model.pkl

# 3. Reiniciar servicios
make compose-down
make compose-up
```

### Problema: Puerto ya en uso

**Síntomas:**
```
Error: bind: address already in use
Port 8080 is already in use
```

**Solución:**

```bash
# 1. Encontrar proceso usando el puerto
lsof -i :8080  # Para API
lsof -i :8501  # Para Streamlit

# 2. Matar proceso
kill -9 <PID>

# 3. O cambiar puerto en docker-compose.yaml
# Editar:
# ports:
#   - "8081:8080"  # Cambiar puerto host
```

### Problema: Contenedores no inician

**Síntomas:**
- `docker-compose up` falla
- Contenedores se reinician constantemente

**Solución:**

```bash
# 1. Ver logs detallados
docker-compose logs

# 2. Verificar imágenes
docker images | grep housing

# 3. Rebuild imágenes
docker-compose build --no-cache

# 4. Limpiar todo y reiniciar
docker-compose down -v
docker system prune -a
make compose-up
```

### Problema: Streamlit no se conecta al API en Docker

**Síntomas:**
- Streamlit muestra error de conexión
- En logs: "Connection refused to http://localhost:8080"

**Solución:**

En `docker-compose.yaml`, el servicio Streamlit debe usar:
```yaml
environment:
  - API_URL=http://api:8080  # ✅ Correcto (nombre del servicio)
  # NO usar http://localhost:8080  # ❌ Incorrecto
```

Si ya está correcto, rebuild:
```bash
docker-compose down
docker-compose build streamlit
docker-compose up -d
```

---

## 9. Monitoreo y Observabilidad

### Logs en Tiempo Real

```bash
# Todos los servicios
make compose-logs

# Solo API
docker-compose logs -f api

# Solo Streamlit
docker-compose logs -f streamlit

# Últimas 100 líneas
docker-compose logs --tail=100
```

### Métricas en W&B

```bash
# Acceder al dashboard
# https://wandb.ai/<your-entity>/housing-mlops-api

# Métricas trackeadas:
# - predictions/count
# - predictions/avg_price
# - predictions/response_time_ms
# - predictions/errors
```

### MLflow Tracking

```bash
# Si corre MLflow localmente:
mlflow ui --port 5000

# Acceder a:
# http://localhost:5000

# Ver experimentos, runs, modelos registrados
```

---

## 10. Deployment a Producción

### Google Cloud Run

#### Deploy API

```bash
# 1. Build y push imagen
cd api
gcloud builds submit --tag gcr.io/<PROJECT_ID>/housing-api:latest

# 2. Deploy a Cloud Run
gcloud run deploy housing-price-api \
  --image gcr.io/<PROJECT_ID>/housing-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "LOCAL_MODEL_PATH=/app/models/trained/housing_price_model.pkl,WANDB_API_KEY=$WANDB_API_KEY"

# 3. Obtener URL
gcloud run services describe housing-price-api --region us-central1 --format 'value(status.url)'
```

#### Deploy Streamlit

```bash
# 1. Build y push imagen
cd streamlit_app
gcloud builds submit --tag gcr.io/<PROJECT_ID>/housing-streamlit:latest

# 2. Deploy a Cloud Run
gcloud run deploy housing-streamlit \
  --image gcr.io/<PROJECT_ID>/housing-streamlit:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars "API_URL=https://<API-URL>.run.app"

# Reemplazar <API-URL> con la URL del API deployado
```

---

## 11. Próximos Pasos

### Mejoras Recomendadas

1. **Autenticación**
   - Implementar JWT tokens en API
   - OAuth2 para Streamlit

2. **Caching**
   - Redis para predicciones frecuentes
   - TTL de 1 hora

3. **A/B Testing**
   - Comparar múltiples versiones de modelos
   - Routing basado en flags

4. **Batch Predictions**
   - Endpoint para procesar múltiples archivos
   - Queue con Celery + RabbitMQ

5. **Monitoring Avanzado**
   - Prometheus + Grafana
   - Alertas con PagerDuty

---

## 12. Recursos Adicionales

### Documentación

- **API Architecture Post**: `API_ARCHITECTURE_POST.md` (detalle completo)
- **API README**: `api/README.md`
- **Streamlit README**: `streamlit_app/README.md`
- **Testing Guide**: `tests/README_TESTING_PHILOSOPHY.md`

### Acceso Rápido

```bash
# Ver documentación del API
cat API_ARCHITECTURE_POST.md

# API docs interactivas (cuando API está corriendo)
open http://localhost:8080/docs

# ReDoc (documentación alternativa)
open http://localhost:8080/redoc
```

---

## Resumen de Comandos Principales

```bash
# ========================================
# INICIO RÁPIDO (1 comando)
# ========================================
make compose-up          # Inicia API + Streamlit con Docker

# ========================================
# DESARROLLO LOCAL
# ========================================
make run-full-stack     # API + Streamlit en paralelo (hot reload)

# O en terminales separadas:
make api-local          # Terminal 1: API en 8080
make streamlit-local    # Terminal 2: Streamlit en 8501

# ========================================
# PIPELINE DE ENTRENAMIENTO
# ========================================
make run-pipeline       # Ejecutar 7 pasos completos

# ========================================
# TESTING
# ========================================
make api-test           # Tests del API
make test-cov           # Tests con coverage

# ========================================
# UTILITIES
# ========================================
make compose-logs       # Ver logs
make compose-down       # Detener servicios
make clean              # Limpiar archivos temporales
make help               # Ver todos los comandos
```

---

**¡Listo! Ya tienes todo el sistema funcionando. Disfruta prediciendo precios de viviendas! 🏠**

**Author**: Carlos Daniel Jiménez
**Version**: 1.0.0
**Last Updated**: January 2024
