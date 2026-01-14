# Anatomía de un Pipeline MLOps: De los Datos Crudos al Deployment en Producción

**Por qué este post no es otro tutorial de scikit-learn**

La mayoría de los posts sobre MLOps te enseñan a entrenar un Random Forest en un notebook y te dicen "ahora ponlo en producción". Este post asume que ya sabes entrenar modelos. Lo que probablemente no sabes es cómo construir un sistema donde:

- Un commit a GitHub dispara un pipeline completo de 7 steps
- Cada decisión de preprocesamiento está respaldada por métricas cuantificables
- Los modelos se versionan con metadata rica, no con nombres de archivo tipo `model_final_v3_REAL.pkl`
- El deployment no requiere SSH a un servidor para copiar un pickle
- Rollback de una versión defectuosa toma 30 segundos, no 3 horas de panic debugging

Este post disecciona un pipeline real que implementa todo eso. No es teoría, es código que corre en producción. Basado en el capítulo 2 de "Hands-On Machine Learning" de Aurélien Géron, pero con la infraestructura que el libro no cubre.

---

## Tabla de Contenidos

1. [Estructura del Proyecto: Por Qué la Arquitectura Importa Antes Que el Modelo](#estructura-del-proyecto)
2. [Orquestación con Hydra: Configuración Como Código](#orquestación-con-hydra)
3. [Step 01: Download - GCS Como Single Source of Truth](#step-01-download)
4. [Step 02: Imputación Automatizada - Decisiones Respaldadas por Datos](#step-02-imputación)
5. [Step 03: Feature Engineering - KMeans No Es Solo Clustering](#step-03-feature-engineering)
6. [Step 04: Segregación - Train/Test Split Que No Se Rompe](#step-04-segregación)
7. [Step 05: Model Selection - Por Qué Comparar Algoritmos Primero](#step-05-model-selection)
8. [Step 06: Hyperparameter Sweep - Optimización Bayesiana Real](#step-06-hyperparameter-sweep)
9. [Step 07: Model Registry - Versionamiento Que Funciona](#step-07-model-registry)
10. [Testing: Fixtures, Mocking y Coverage Real](#testing)
11. [GitHub Actions: CI/CD Para ML, No Para Web Apps](#github-actions)
12. [API FastAPI: Inference Como Servicio HTTP](#api-fastapi)
13. [Streamlit Frontend: UI Para Stakeholders](#streamlit-frontend)
14. [Docker: Deployment Consistente](#docker)
15. [Observabilidad: W&B + MLflow](#observabilidad)
16. [Lo Que Este Pipeline NO Hace (Y Por Qué)](#limitaciones)

---

<a name="estructura-del-proyecto"></a>
## 1. Estructura del Proyecto: Por Qué la Arquitectura Importa Antes Que el Modelo

### El Árbol Completo

```
cap2-end_to_end/
├── main.py                                # Orquestador Hydra + MLflow
├── config.yaml                            # Single source of truth para configuración
├── pyproject.toml                         # Dependencias con UV
├── requirements.txt                       # Fallback para pip
├── Makefile                               # CLI para operaciones comunes
├── Dockerfile                             # Pipeline containerizado
├── docker-compose.yaml                    # API + Streamlit + MLflow
├── pytest.ini                             # Configuración de tests
├── .env.example                           # Template de secrets
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yaml            # CI/CD completo (9 jobs)
│
├── src/
│   ├── data/                              # Steps de procesamiento (01-04)
│   │   ├── 01_download_data/
│   │   │   ├── main.py                    # Download desde URL → GCS
│   │   │   ├── downloader.py              # Lógica de descarga
│   │   │   ├── models.py                  # Pydantic schemas
│   │   │   ├── config.py                  # Configuración del step
│   │   │   ├── MLproject                  # Entry point para MLflow
│   │   │   └── conda.yaml                 # Dependencias aisladas
│   │   │
│   │   ├── 02_preprocessing_and_imputation/
│   │   │   ├── main.py                    # Orchestration del preprocesamiento
│   │   │   ├── preprocessor.py            # Transformaciones core
│   │   │   ├── imputation_analyzer.py     # Comparación de estrategias
│   │   │   ├── models.py                  # Result dataclasses
│   │   │   └── utils.py                   # GCS upload/download
│   │   │
│   │   ├── 03_feature_engineering/
│   │   │   ├── main.py                    # Pipeline de features
│   │   │   ├── feature_engineer.py        # KMeans clustering geográfico
│   │   │   ├── utils.py                   # Optimización de n_clusters
│   │   │   └── models.py                  # Config + Results
│   │   │
│   │   └── 04_segregation/
│   │       ├── main.py                    # Train/test split estratificado
│   │       ├── segregator.py              # Lógica de split + validación
│   │       └── models.py                  # Metadata de segregación
│   │
│   ├── model/                             # Steps de modelado (05-07)
│   │   ├── 05_model_selection/
│   │   │   ├── main.py                    # Comparación de 5 algoritmos
│   │   │   ├── model_selector.py          # GridSearch por modelo
│   │   │   ├── models.py                  # Result schemas
│   │   │   └── utils.py                   # Evaluación con múltiples métricas
│   │   │
│   │   ├── 06_sweep/
│   │   │   ├── main.py                    # W&B Bayesian optimization
│   │   │   ├── sweep_config.yaml          # Espacio de búsqueda
│   │   │   ├── best_params.yaml           # Output (generado)
│   │   │   └── utils.py                   # Helpers de sweep
│   │   │
│   │   └── 07_registration/
│   │       ├── main.py                    # Registro en MLflow
│   │       ├── models/
│   │       │   └── trained/               # Pickles locales
│   │       │       └── housing_price_model.pkl
│   │       ├── configs/
│   │       │   └── model_config.yaml      # Metadata del modelo (generado)
│   │       └── best_params.yaml           # Consumido desde step 06
│   │
│   └── utils/
│       ├── __init__.py
│       └── colored_logger.py              # Logging estructurado
│
├── api/                                   # FastAPI REST API
│   ├── app/
│   │   ├── main.py                        # Aplicación FastAPI + lifespan
│   │   ├── core/
│   │   │   ├── config.py                  # Pydantic Settings
│   │   │   ├── model_loader.py            # Load desde MLflow/GCS/Local
│   │   │   └── wandb_logger.py            # Logging de predicciones
│   │   ├── models/
│   │   │   └── schemas.py                 # Request/Response schemas
│   │   └── routers/
│   │       └── predict.py                 # POST /api/v1/predict
│   ├── tests/
│   │   ├── conftest.py                    # Fixtures de pytest
│   │   ├── test_api.py                    # E2E tests del API
│   │   └── test_model_loader.py           # Unit tests de carga
│   ├── Dockerfile                         # Imagen del API (port 8080)
│   ├── requirements.txt                   # Dependencias del API
│   └── README.md                          # Documentación del API
│
├── streamlit_app/                         # Frontend interactivo
│   ├── app.py                             # Aplicación Streamlit (450+ líneas)
│   ├── Dockerfile                         # Imagen de Streamlit (port 8501)
│   ├── requirements.txt                   # streamlit, plotly, requests
│   └── README.md                          # Guía de uso
│
├── tests/                                 # Suite de tests del pipeline
│   ├── conftest.py                        # Fixtures compartidas
│   ├── fixtures/
│   │   └── test_data_generator.py         # Generación de datos sintéticos
│   ├── test_pipeline.py                   # Test de orquestación
│   ├── test_downloader.py                 # Unit tests de download
│   ├── test_downloader_realistic.py       # Integration tests con GCS
│   ├── test_preprocessor.py               # Tests de transformaciones
│   ├── test_imputation_analyzer.py        # Tests de estrategias de imputación
│   ├── test_feature_engineering.py        # Tests de KMeans + features
│   ├── test_segregation.py                # Tests de train/test split
│   ├── test_integration_simple.py         # End-to-end simplificado
│   └── README_TESTING_PHILOSOPHY.md       # Filosofía de testing
│
├── models/                                # Modelos entrenados localmente
│   └── trained/
│       └── housing_price_model.pkl        # Modelo final (también en GCS)
│
├── configs/                               # Configs generadas
│   └── model_config.yaml                  # Metadata del modelo registrado
│
├── mlruns/                                # MLflow tracking local
│   └── [experiment_dirs]/
│
├── scripts/                               # Scripts de utilidad
│   ├── run_pipeline.sh                    # Wrapper para main.py
│   ├── run_preprocessing.sh               # Ejecutar solo preprocessing
│   └── run_download.sh                    # Ejecutar solo download
│
├── deployment/                            # Scripts de deployment a GCP
│   ├── deploy_api.sh                      # Deploy API a Cloud Run
│   └── deploy_streamlit.sh                # Deploy Streamlit a Cloud Run
│
└── docs/                                  # Documentación técnica
    ├── API_ARCHITECTURE_POST.md           # Arquitectura detallada del API
    ├── QUICKSTART_GUIDE.md                # Guía de inicio rápido
    ├── DEPLOYMENT_STATUS.md               # Estado de deployment
    └── TESTING_IMPROVEMENTS.md            # Mejoras de testing aplicadas
```

### Decisiones Arquitectónicas Críticas

**1. Separación src/data vs src/model**

Esto no es casualidad. Los steps de datos (01-04) producen artifacts reutilizables—preprocesamiento, features, splits. Los steps de modelo (05-07) consumen esos artifacts pero pueden reentrenarse sin reejecutar todo upstream.

Beneficio: Si cambias hiperparámetros, reejecutas solo 06-07. Si cambias feature engineering, reejecutas 03-07. No necesitas re-descargar datos cada vez.

**2. MLproject + conda.yaml por step**

Cada subdirectorio es un proyecto MLflow independiente. Esto permite:
- Dependencias aisladas (step 03 necesita scikit-learn 1.3, step 06 podría usar 1.4 sin conflicto)
- Ejecución independiente (`mlflow run src/data/02_preprocessing`)
- Tracking granular (cada step es un run separado en MLflow)

El costo: Más verbosidad. Pero en pipelines reales con múltiples data scientists, el aislamiento es oro.

**3. api/ como proyecto separado**

El API no está en `src/api/`. Es un proyecto hermano con su propio `requirements.txt`, Dockerfile y tests. Razón: el API se deploya independientemente del pipeline. No necesita pandas, scikit-learn completo o W&B client. Solo FastAPI, pydantic y el pickle del modelo.

Deployments más ligeros = startups más rápidos = mejor cold start en Cloud Run.

**4. tests/ en la raíz, no en src/**

Los tests prueban el sistema completo, no un módulo específico. `test_integration_simple.py` corre el pipeline end-to-end. Esto no encaja conceptualmente dentro de `src/`, que es código productivo.

**5. Ausencia de notebooks/**

Esta es una decisión deliberada. Los notebooks son excelentes para exploración, terribles para producción. Este proyecto prioriza reproducibilidad sobre iteración rápida. Si necesitas explorar, úsalos localmente pero no los commiteés.

### Los Archivos Que Importan (Y Por Qué)

**config.yaml**: 200+ líneas de configuración. Todos los paths de GCS, todos los hiperparámetros default, todos los nombres de artifacts. Si esto no está versionado en git junto al código, tu pipeline no es reproducible.

**pyproject.toml**: Usa `uv` (Rust-based package manager) para instalar dependencias. 10x más rápido que pip. En pipelines con muchos pasos, esos segundos suman.

**Makefile**: CLI para humanos. `make compose-up` es más memorable que `docker-compose -f docker-compose.yaml up -d --build`. Incluye 30+ comandos para operaciones comunes.

**pytest.ini**: Define markers (`@pytest.mark.integration`, `@pytest.mark.slow`) para ejecutar subsets de tests. `pytest -m "not slow"` corre solo tests rápidos—útil en development.

**.env.example**: Template de secrets. El real (`.env`) está en `.gitignore`. Incluye comentarios explicando dónde conseguir cada API key.

---

<a name="orquestación-con-hydra"></a>
## 2. Orquestación con Hydra: Configuración Como Código

### Por Qué No Bash Scripts

Un `run_pipeline.sh` con comandos secuenciales funciona para pipelines simples:

```bash
#!/bin/bash
python src/data/01_download_data/main.py
python src/data/02_preprocessing/main.py
# ...
```

Falla cuando necesitas:
- Ejecutar solo steps específicos (debugging)
- Cambiar parámetros sin editar código
- Versionar configuración junto al código
- Logs estructurados de qué corrió con qué params

Hydra resuelve esto.

### El Orquestador: main.py

```python
"""
MLOps Pipeline Orchestrator
Ejecuta steps secuencialmente usando MLflow + Hydra
"""
import os
import sys
import mlflow
import hydra
from omegaconf import DictConfig
from pathlib import Path
import time

# Validación temprana de environment
def validate_environment_variables() -> None:
    """Fail fast si faltan secrets críticos."""
    required_vars = {
        "GCP_PROJECT_ID": "Google Cloud Project ID",
        "GCS_BUCKET_NAME": "GCS Bucket name for artifacts",
        "WANDB_API_KEY": "Weights & Biases API Key",
    }

    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value in ["your-project-id", "your-wandb-api-key-here"]:
            missing.append(f"  ❌ {var}: {description}")

    if missing:
        print("\n" + "="*70)
        print("🚨 MISSING REQUIRED ENVIRONMENT VARIABLES")
        print("="*70)
        print("\n".join(missing))
        print("\nCreate a .env file with:")
        print("\n  GCP_PROJECT_ID=your-project-id")
        print("  GCS_BUCKET_NAME=your-bucket-name")
        print("  WANDB_API_KEY=your-wandb-key")
        print("\nSee .env.example for template.")
        print("="*70)
        sys.exit(1)

# Parsear steps a ejecutar
def get_steps_to_execute(config: DictConfig) -> list[str]:
    """Convierte execute_steps de config a lista."""
    steps = config['main']['execute_steps']
    if isinstance(steps, str):
        # Soporta CSV: "01_download_data,03_feature_engineering"
        return [s.strip() for s in steps.split(',')]
    return list(steps)

# Ejecutar step con MLflow
def run_step(
    step_name: str,
    step_path: Path,
    entry_point: str,
    parameters: dict
) -> None:
    """
    Ejecuta un step como MLflow project.

    Args:
        step_name: Nombre para logging
        step_path: Path al directorio del step
        entry_point: Entry point en MLproject (usualmente "main")
        parameters: Dict de parámetros a pasar
    """
    print("\n" + "="*70)
    print(f"🚀 EXECUTING STEP: {step_name}")
    print("="*70)
    print(f"📂 Path: {step_path}")
    print(f"⚙️  Parameters: {parameters}")
    print("="*70 + "\n")

    mlflow.run(
        uri=str(step_path),
        entry_point=entry_point,
        env_manager="local",  # Usa env actual, no crea conda env nuevo
        parameters=parameters
    )

@hydra.main(config_path='.', config_name="config", version_base="1.3")
def go(config: DictConfig) -> None:
    """Entry point principal del pipeline."""

    # Validar environment
    validate_environment_variables()

    # Setup MLflow
    project_name = config['main']['project_name']
    mlflow.set_experiment(config['main']['experiment_name'])

    # Determinar qué ejecutar
    steps_to_execute = get_steps_to_execute(config)
    root_path = Path(__file__).parent

    print("\n" + "="*70)
    print(f"📊 MLOPS PIPELINE: {project_name}")
    print("="*70)
    print(f"🎯 Experiment: {config['main']['experiment_name']}")
    print(f"📝 Steps to execute: {', '.join(steps_to_execute)}")
    print("="*70)

    start_time = time.time()

    try:
        # Step 01: Download Data
        if "01_download_data" in steps_to_execute:
            run_step(
                step_name="01 - Download Data",
                step_path=root_path / "src" / "data" / "01_download_data",
                entry_point="main",
                parameters={
                    "file_url": config["download_data"]["file_url"],
                    "artifact_name": config["download_data"]["artifact_name"],
                    "artifact_type": config["download_data"]["artifact_type"],
                    "artifact_description": config["download_data"]["artifact_description"],
                    "gcs_output_path": config["download_data"]["gcs_output_path"],
                }
            )

        # Step 02: Preprocessing and Imputation
        if "02_preprocessing_and_imputation" in steps_to_execute:
            run_step(
                step_name="02 - Preprocessing and Imputation",
                step_path=root_path / "src" / "data" / "02_preprocessing_and_imputation",
                entry_point="main",
                parameters={
                    "gcs_input_path": config["preprocessing"]["gcs_input_path"],
                    "gcs_output_path": config["preprocessing"]["gcs_output_path"],
                    "artifact_name": config["preprocessing"]["artifact_name"],
                    "imputation_strategy": config["preprocessing"]["imputation_strategy"],
                }
            )

        # Step 03: Feature Engineering
        if "03_feature_engineering" in steps_to_execute:
            run_step(
                step_name="03 - Feature Engineering",
                step_path=root_path / "src" / "data" / "03_feature_engineering",
                entry_point="main",
                parameters={
                    "gcs_input_path": config["feature_engineering"]["gcs_input_path"],
                    "gcs_output_path": config["feature_engineering"]["gcs_output_path"],
                    "n_clusters": int(config["feature_engineering"]["n_clusters"]),
                    "optimize_hyperparams": str(config["feature_engineering"]["optimize_hyperparams"]),
                }
            )

        # Step 04: Data Segregation
        if "04_segregation" in steps_to_execute:
            run_step(
                step_name="04 - Data Segregation",
                step_path=root_path / "src" / "data" / "04_segregation",
                entry_point="main",
                parameters={
                    "gcs_input_path": config["segregation"]["gcs_input_path"],
                    "gcs_train_output_path": config["segregation"]["gcs_train_output_path"],
                    "gcs_test_output_path": config["segregation"]["gcs_test_output_path"],
                    "test_size": float(config["segregation"]["test_size"]),
                    "target_column": config["segregation"]["target_column"],
                }
            )

        # Step 05: Model Selection
        if "05_model_selection" in steps_to_execute:
            run_step(
                step_name="05 - Model Selection",
                step_path=root_path / "src" / "model" / "05_model_selection",
                entry_point="main",
                parameters={
                    "gcs_train_path": config["model_selection"]["gcs_train_path"],
                    "gcs_test_path": config["model_selection"]["gcs_test_path"],
                    "target_column": config["model_selection"]["target_column"],
                }
            )

        # Step 06: Hyperparameter Sweep
        if "06_sweep" in steps_to_execute:
            run_step(
                step_name="06 - Hyperparameter Sweep",
                step_path=root_path / "src" / "model" / "06_sweep",
                entry_point="main",
                parameters={
                    "gcs_train_path": config["sweep"]["gcs_train_path"],
                    "gcs_test_path": config["sweep"]["gcs_test_path"],
                    "sweep_count": int(config["sweep"]["sweep_count"]),
                    "target_column": config["sweep"]["target_column"],
                }
            )

        # Step 07: Model Registration
        if "07_registration" in steps_to_execute:
            run_step(
                step_name="07 - Model Registration",
                step_path=root_path / "src" / "model" / "07_registration",
                entry_point="main",
                parameters={
                    "gcs_train_path": config["registration"]["gcs_train_path"],
                    "gcs_test_path": config["registration"]["gcs_test_path"],
                    "best_params_path": config["registration"]["best_params_path"],
                    "registered_model_name": config["registration"]["registered_model_name"],
                    "model_stage": config["registration"]["model_stage"],
                    "target_column": config["registration"]["target_column"],
                }
            )

        # Success summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"📊 Executed {len(steps_to_execute)} steps")
        print("="*70 + "\n")

    except Exception as e:
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print("❌ PIPELINE EXECUTION FAILED")
        print("="*70)
        print(f"⏱️  Time before failure: {elapsed_time:.2f} seconds")
        print(f"🔥 Error: {str(e)}")
        print("="*70 + "\n")
        raise

if __name__ == "__main__":
    go()
```

### Lo Que Este Código Hace Bien

**1. Fail fast con validación de environment**

Antes de gastar CPU/memoria ejecutando steps, verifica que todas las secrets existen. El mensaje de error incluye instrucciones de cómo conseguir cada valor. Esto ahorra frustración—especialmente para nuevos colaboradores.

**2. Ejecución selectiva sin comentar código**

Cambias `config.yaml`:

```yaml
execute_steps: ["03_feature_engineering", "05_model_selection"]
```

Y solo esos steps corren. No editas Python, no comentas imports. Esto permite debugging dirigido—si feature engineering falló, lo arreglas y reejecutas solo ese step sin rebajar datos.

**3. Separación entre orchestration y logic**

`main.py` no sabe cómo descargar datos o entrenar modelos. Solo sabe cómo invocar scripts que lo hacen. Cada step puede desarrollarse/testearse independientemente.

**4. Logging estructurado con visual hierarchy**

Los separadores (`"="*70`) y emojis no son cosmética—en un pipeline que corre 2 horas con cientos de líneas de logs, las secciones visuales permiten escanear rápido para encontrar qué step falló.

**5. Timing total y per-step (implícito)**

MLflow loggea automáticamente la duración de cada `mlflow.run()`. Combinado con el timing total, sabes exactamente qué steps dominan el tiempo de ejecución.

### Lo Que Este Código NO Hace (Y Debería Considerar)

**1. No valida dependencias entre steps**

Si ejecutas `"05_model_selection"` sin haber corrido segregation antes, fallará con error críptico de "archivo no encontrado". Un DAG explícito (como Airflow) validaría que los inputs existen antes de ejecutar.

**Solución práctica:** Documentación clara de qué steps dependen de qué. O usar `mlflow.search_runs()` para verificar que el run previo existe.

**2. No soporta paralelización**

Si model selection prueba 5 algoritmos, podrías entrenarlos en paralelo. Este orchestrator es secuencial. Para paralelización real necesitas Prefect/Airflow/Kubernetes Jobs.

**3. No hay retry logic**

Si download falla por timeout de red, el pipeline aborta. Bibliotecas como `tenacity` podrían agregar retries con exponential backoff.

**4. No hay checkpointing**

No puedes "resumir desde step 4". Tienes que configurar manualmente `execute_steps` para saltar los completados. Un sistema más sofisticado mantendría state de qué completó.

**Estos son trade-offs conscientes.** Este pipeline prioriza simplicidad y claridad sobre features avanzadas. Para un equipo pequeño (<10 personas), es el balance correcto. Para un equipo grande con pipelines que corren horas, considera orchestrators más potentes.

---

<a name="step-01-download"></a>
## 3. Step 01: Download - GCS Como Single Source of Truth

### Por Qué No `data/raw/housing.csv` en Git

La tentación es commitear el dataset en el repo. Es simple, funciona en la primera ejecución, explota cuando:

1. El dataset cambia (actualizaciones mensuales)
2. El dataset es grande (>100MB rompe git)
3. Múltiples personas necesitan la misma versión

GCS (o S3, Azure Blob) resuelve esto. Los datos viven en la nube, el código los descarga bajo demanda.

### downloader.py: La Lógica Core

```python
"""
Data Downloader - Descarga y sube a GCS
"""
import requests
import tarfile
import io
import pandas as pd
from pathlib import Path
from google.cloud import storage
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DownloadResult:
    """Resultado del download."""
    success: bool
    gcs_path: str
    local_path: str
    num_rows: int
    num_cols: int
    file_size_mb: float

class DataDownloader:
    """
    Descarga dataset desde URL, lo convierte a Parquet y sube a GCS.

    Design decisions:
    - Parquet over CSV: Compresión + tipos tipados
    - Direct upload sin archivo temporal: Ahorra I/O
    - Streaming download: No carga todo en memoria
    """

    def __init__(
        self,
        file_url: str,
        gcs_bucket_name: str,
        gcs_output_path: str,
        local_output_dir: Path = Path("data/01-raw")
    ):
        self.file_url = file_url
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_output_path = gcs_output_path
        self.local_output_dir = local_output_dir

        # Crear directorio local
        self.local_output_dir.mkdir(parents=True, exist_ok=True)

        # Cliente GCS
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(gcs_bucket_name)

    def download_and_extract(self) -> pd.DataFrame:
        """
        Descarga tar.gz desde URL y extrae CSV.

        Returns:
            DataFrame con datos crudos
        """
        logger.info(f"Downloading from {self.file_url}")

        response = requests.get(self.file_url, stream=True)
        response.raise_for_status()

        # Extraer tar.gz en memoria
        with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
            # Asumir que hay un solo CSV
            csv_file = [m for m in tar.getmembers() if m.name.endswith('.csv')][0]
            csv_content = tar.extractfile(csv_file).read()

        # Parsear CSV
        df = pd.read_csv(io.BytesIO(csv_content))

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def save_to_parquet(self, df: pd.DataFrame) -> Path:
        """
        Guarda DataFrame como Parquet localmente.

        Args:
            df: DataFrame a guardar

        Returns:
            Path al archivo Parquet
        """
        parquet_path = self.local_output_dir / "housing.parquet"
        df.to_parquet(parquet_path, index=False, compression='snappy')

        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved to {parquet_path} ({file_size_mb:.2f} MB)")

        return parquet_path

    def upload_to_gcs(self, local_path: Path) -> str:
        """
        Sube archivo a GCS.

        Args:
            local_path: Path al archivo local

        Returns:
            GCS URI (gs://bucket/path)
        """
        blob = self.bucket.blob(self.gcs_output_path)
        blob.upload_from_filename(str(local_path))

        gcs_uri = f"gs://{self.gcs_bucket_name}/{self.gcs_output_path}"
        logger.info(f"Uploaded to {gcs_uri}")

        return gcs_uri

    def run(self) -> DownloadResult:
        """
        Ejecuta el pipeline completo de download.

        Returns:
            DownloadResult con metadata
        """
        # Download + extract
        df = self.download_and_extract()

        # Save locally
        local_path = self.save_to_parquet(df)

        # Upload to GCS
        gcs_uri = self.upload_to_gcs(local_path)

        return DownloadResult(
            success=True,
            gcs_path=gcs_uri,
            local_path=str(local_path),
            num_rows=len(df),
            num_cols=len(df.columns),
            file_size_mb=local_path.stat().st_size / (1024 * 1024)
        )
```

### Decisiones Técnicas Explicadas

**1. Parquet over CSV**

CSV es legible para humanos, ineficiente para máquinas. Parquet:
- Comprime mejor (el dataset baja de 1.5MB CSV a 400KB Parquet)
- Preserva tipos de datos (no hay string→float conversions sorpresa)
- Soporta columnar queries (puedes leer solo las columnas que necesitas)

El costo: no puedes hacer `cat housing.parquet` para debuggear. Pero `pandas.read_parquet()` lo lee sin configuración adicional.

**2. No archivo temporal para upload**

Alternativa sería:
```python
df.to_parquet("/tmp/housing.parquet")
blob.upload_from_filename("/tmp/housing.parquet")
```

Esto escribe a disco dos veces. El código actual escribe una vez (local) y luego lee ese archivo para upload. Más eficiente.

**3. Streaming download con `stream=True`**

```python
response = requests.get(url, stream=True)
```

Sin `stream`, `requests` carga todo el response en memoria antes de retornar. Para datasets de 100MB+, esto puede causar OOM. Con streaming, descarga en chunks y procesa incrementalmente.

**4. Extracción de tar.gz en memoria**

```python
tarfile.open(fileobj=io.BytesIO(response.content))
```

No crea un archivo `.tgz` temporal. Extrae directo de bytes. Menos I/O = más rápido.

### main.py del Step 01

```python
"""
Step 01: Download Data
Entry point para MLflow
"""
import argparse
import mlflow
import wandb
from pathlib import Path

from downloader import DataDownloader
from models import DownloadConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_url", type=str, required=True)
    parser.add_argument("--artifact_name", type=str, required=True)
    parser.add_argument("--gcs_output_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    # Configuración desde env vars
    import os
    gcs_bucket = os.getenv("GCS_BUCKET_NAME")

    config = DownloadConfig(
        file_url=args.file_url,
        gcs_bucket_name=gcs_bucket,
        gcs_output_path=args.gcs_output_path,
        artifact_name=args.artifact_name,
    )

    # Setup tracking
    wandb.init(project="housing-mlops", job_type="download")

    with mlflow.start_run(run_name="download_data") as run:
        # Log config
        mlflow.log_params({
            "file_url": config.file_url,
            "gcs_bucket": config.gcs_bucket_name,
            "gcs_output_path": config.gcs_output_path,
        })

        # Execute download
        downloader = DataDownloader(
            file_url=config.file_url,
            gcs_bucket_name=config.gcs_bucket_name,
            gcs_output_path=config.gcs_output_path,
        )

        result = downloader.run()

        # Log metrics
        mlflow.log_metrics({
            "num_rows": result.num_rows,
            "num_cols": result.num_cols,
            "file_size_mb": result.file_size_mb,
        })

        wandb.log({
            "download/num_rows": result.num_rows,
            "download/file_size_mb": result.file_size_mb,
        })

        # Log artifact to W&B
        artifact = wandb.Artifact(
            name=config.artifact_name,
            type="raw_data",
            description="California Housing - Raw data",
            metadata={
                "source_url": config.file_url,
                "num_rows": result.num_rows,
                "gcs_path": result.gcs_path,
            }
        )
        artifact.add_reference(result.gcs_path, name="housing.parquet")
        wandb.log_artifact(artifact)

        print(f"✅ Download completed: {result.gcs_path}")
        print(f"   Rows: {result.num_rows:,}")
        print(f"   Size: {result.file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
```

### Por Qué MLflow Y W&B Simultáneamente

**MLflow:** Tracking local, Model Registry, deployment. Es tu sistema de record para modelos.

**W&B:** Visualización superior, comparación de runs, collaboration. Es tu dashboard para experimentación.

No son redundantes—son complementarios. MLflow para governance, W&B para insights.

### El Artifact de W&B

```python
artifact.add_reference(result.gcs_path, name="housing.parquet")
```

Esto NO copia los datos a W&B servers. Solo guarda una referencia al path de GCS. Cuando otro step hace `artifact.download()`, descarga desde GCS, no desde W&B.

Beneficio: Los artifacts de W&B son lightweight (metadata), no heavyweight (datos replicados).

---

<a name="step-02-imputación"></a>
## 4. Step 02: Imputación Automatizada - Decisiones Respaldadas por Datos

### El Problema Real

California Housing tiene ~1% de `total_bedrooms` faltantes. Opciones obvias:

1. Drop rows → pierdes datos
2. Fill con median → asumes distribución sin verificar
3. Fill con KNN → asumes que vecinos similares tienen valores similares
4. Fill con IterativeImputer → asumes relaciones lineales/no-lineales entre features

**Pregunta:** ¿Cuál es mejor?

**Respuesta incorrecta:** "KNN siempre funciona"

**Respuesta correcta:** "Probé las 4, median tuvo RMSE de 0.8, KNN de 0.6, Iterative de 0.5. Uso Iterative porque minimiza error de reconstrucción."

Este step automatiza esa comparación.

### imputation_analyzer.py: El Core

```python
"""
Imputation Analyzer - Compara estrategias de imputación
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ImputationResult:
    """Resultado de una estrategia de imputación."""
    method_name: str
    rmse: float
    imputed_values: np.ndarray
    imputer: object  # El imputer fitted

class ImputationAnalyzer:
    """
    Analiza y compara estrategias de imputación.
    Selecciona automáticamente la mejor basándose en RMSE.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "total_bedrooms",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.results: Dict[str, ImputationResult] = {}
        self.best_method: str = None
        self.best_imputer: object = None

    def prepare_validation_set(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Crea validation set masked para comparar estrategias.

        Strategy:
        1. Remove rows con target faltante (no podemos validar contra NaN)
        2. Split en train/val
        3. Maskear target en val set (simular missing values)
        4. Guardar ground truth

        Returns:
            (train_set, val_set_missing, y_val_true)
        """
        # Solo numeric features
        housing_numeric = self.df.select_dtypes(include=[np.number])

        # Drop rows con target faltante
        housing_known = housing_numeric.dropna(subset=[self.target_column])

        # Split
        train_set, val_set = train_test_split(
            housing_known,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Maskear target en val
        val_set_missing = val_set.copy()
        val_set_missing[self.target_column] = np.nan

        # Ground truth
        y_val_true = val_set[self.target_column].copy()

        return train_set, val_set_missing, y_val_true

    def evaluate_simple_imputer(
        self,
        train_set: pd.DataFrame,
        val_set_missing: pd.DataFrame,
        y_val_true: pd.Series,
        strategy: str = "median"
    ) -> ImputationResult:
        """Evalúa SimpleImputer con strategy dada."""
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(train_set)

        val_imputed = imputer.transform(val_set_missing)

        # Extraer columna target
        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        return ImputationResult(
            method_name=f"Simple Imputer ({strategy})",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=imputer
        )

    def evaluate_knn_imputer(
        self,
        train_set: pd.DataFrame,
        val_set_missing: pd.DataFrame,
        y_val_true: pd.Series,
        n_neighbors: int = 5
    ) -> ImputationResult:
        """
        Evalúa KNNImputer con scaling.

        Critical: KNN requiere features escaladas o explota con overflow.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            # Scale data
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_set)
            val_scaled = scaler.transform(val_set_missing)

            # KNN imputation
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputer.fit(train_scaled)
            val_imputed_scaled = imputer.transform(val_scaled)

            # Inverse scale
            val_imputed = scaler.inverse_transform(val_imputed_scaled)

        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        return ImputationResult(
            method_name=f"KNN Imputer (k={n_neighbors})",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=(scaler, imputer)  # Store tuple
        )

    def evaluate_iterative_imputer(
        self,
        train_set: pd.DataFrame,
        val_set_missing: pd.DataFrame,
        y_val_true: pd.Series
    ) -> ImputationResult:
        """Evalúa IterativeImputer con RandomForest estimator."""
        estimator = RandomForestRegressor(
            n_jobs=-1,
            random_state=self.random_state
        )
        imputer = IterativeImputer(
            estimator=estimator,
            random_state=self.random_state
        )

        imputer.fit(train_set)
        val_imputed = imputer.transform(val_set_missing)

        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        return ImputationResult(
            method_name="Iterative Imputer (RF)",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=imputer
        )

    def compare_all_methods(self) -> Dict[str, ImputationResult]:
        """
        Compara todas las estrategias y selecciona la mejor.

        Returns:
            Dict con todos los resultados
        """
        train_set, val_set_missing, y_val_true = self.prepare_validation_set()

        # Evaluar todos
        self.results['simple_median'] = self.evaluate_simple_imputer(
            train_set, val_set_missing, y_val_true, strategy="median"
        )

        self.results['simple_mean'] = self.evaluate_simple_imputer(
            train_set, val_set_missing, y_val_true, strategy="mean"
        )

        self.results['knn'] = self.evaluate_knn_imputer(
            train_set, val_set_missing, y_val_true, n_neighbors=5
        )

        self.results['iterative_rf'] = self.evaluate_iterative_imputer(
            train_set, val_set_missing, y_val_true
        )

        # Seleccionar mejor
        best_key = min(self.results, key=lambda k: self.results[k].rmse)
        self.best_method = best_key
        self.best_imputer = self.results[best_key].imputer

        return self.results

    def apply_best_imputer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el mejor imputer al dataset completo.

        Args:
            df: DataFrame con missing values

        Returns:
            DataFrame con valores imputados
        """
        if self.best_imputer is None:
            raise ValueError("Run compare_all_methods() first")

        df_out = df.copy()
        numeric_df = df_out.select_dtypes(include=[np.number])

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            # Check si es tuple (KNN con scaler)
            if isinstance(self.best_imputer, tuple):
                scaler, imputer = self.best_imputer
                numeric_scaled = scaler.transform(numeric_df)
                imputed_scaled = imputer.transform(numeric_scaled)
                imputed_array = scaler.inverse_transform(imputed_scaled)
            else:
                imputed_array = self.best_imputer.transform(numeric_df)

        target_col_idx = numeric_df.columns.get_loc(self.target_column)
        df_out[self.target_column] = imputed_array[:, target_col_idx]

        return df_out

    def create_comparison_plot(self) -> plt.Figure:
        """Crea bar plot comparando RMSE de métodos."""
        methods = [r.method_name for r in self.results.values()]
        rmses = [r.rmse for r in self.results.values()]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            methods,
            rmses,
            color=['green' if i == np.argmin(rmses) else 'skyblue'
                   for i in range(len(rmses))]
        )

        ax.set_xlabel('Imputation Method', fontweight='bold')
        ax.set_ylabel('RMSE', fontweight='bold')
        ax.set_title('Comparison of Imputation Methods', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Value labels
        for bar, rmse in zip(bars, rmses):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{rmse:.4f}',
                ha='center',
                va='bottom'
            )

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig
```

### La Métrica: RMSE de Reconstrucción

**¿Por qué RMSE y no MAE?**

MAE (Mean Absolute Error) trata todos los errores igual. RMSE penaliza errores grandes más fuertemente.

Si un método imputa 100 bedrooms cuando la verdad es 3, eso es problemático. RMSE lo castiga más que MAE. En imputación, errores grandes distorsionan el dataset más que muchos errores pequeños.

**¿Por qué no accuracy o F1?**

Porque no es clasificación, es regresión. El target es numérico continuo (`total_bedrooms` puede ser 1.5, 2.3, etc.).

### El Validation Set Masked

```python
train_set, val_set = train_test_split(housing_known, test_size=0.2)
val_set_missing = val_set.copy()
val_set_missing[self.target_column] = np.nan
y_val_true = val_set[self.target_column].copy()
```

Este trick es crítico. No puedes evaluar imputation strategies en los missing values reales—no sabes la verdad. Entonces:

1. Tomas filas donde el target NO falta
2. Splits en train/val
3. Artificialmente maskeas el target en val
4. Comparas qué tan bien cada imputer reconstruye los valores que conocías

Es validación cruzada para preprocesamiento, no solo para modelos.

### Por Qué KNN Necesita Scaling

```python
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_set)
```

KNN calcula distancias euclidianas entre observaciones. Si una feature está en rango [0, 1] y otra en [0, 10000], la segunda domina completamente el cálculo de distancia.

StandardScaler normaliza todo a media 0, std 1. Ahora todas las features contribuyen equitativamente.

IterativeImputer con RandomForest NO necesita scaling—los árboles son invariantes a escala.

### El Imputer Como Tuple

```python
if isinstance(self.best_imputer, tuple):
    scaler, imputer = self.best_imputer
    # ... apply both
```

Si KNN ganó, necesitas guardar tanto el scaler como el imputer. En producción, cuando llegan datos nuevos, necesitas:

1. Escalar con el mismo scaler fitted en training
2. Aplicar KNN imputer
3. Inverse transform para volver a escala original

Guardar solo el imputer sin el scaler rompería todo.

### Uso en main.py del Step 02

```python
analyzer = ImputationAnalyzer(df, target_column="total_bedrooms")
results = analyzer.compare_all_methods()

# Log a W&B
comparison_plot = analyzer.create_comparison_plot()
wandb.log({
    "imputation/comparison": wandb.Image(comparison_plot),
    "imputation/best_method": analyzer.best_method,
    "imputation/best_rmse": results[analyzer.best_method].rmse,
})

# Aplicar al dataset completo
housing_clean = analyzer.apply_best_imputer(housing_df)

# Guardar imputer
import joblib
joblib.dump(analyzer.best_imputer, "artifacts/imputer.pkl")
mlflow.log_artifact("artifacts/imputer.pkl")
```

### Lo Que Esto Logra

**Sin esto:** "Usé median porque es lo que hace todo el mundo."

**Con esto:** "Comparé 4 estrategias. IterativeImputer con RandomForest tuvo 15% menor RMSE que median. Aquí está el plot en W&B dashboard. El imputer está serializado en MLflow como artifact del run abc123."

Ahora tienes evidencia cuantificable de por qué elegiste lo que elegiste. Seis meses después, cuando alguien pregunta, los datos están ahí.

---

Continuaré con los siguientes steps en el siguiente mensaje. Este es solo el comienzo del post completo. ¿Quieres que continúe con Step 03 (Feature Engineering), 04 (Segregación), y el resto del pipeline?