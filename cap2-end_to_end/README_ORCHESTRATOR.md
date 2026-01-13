# 🎯 Pipeline Orchestrator - Guía Completa

Autor: Carlos Daniel Jiménez
Fecha: 2025-01-13

## 📋 Índice

1. [¿Qué es el Orchestrator?](#qué-es-el-orchestrator)
2. [Arquitectura](#arquitectura)
3. [Cómo Ejecutar](#cómo-ejecutar)
4. [Configuración](#configuración)
5. [Agregar Nuevos Pasos](#agregar-nuevos-pasos)

---

## ¿Qué es el Orchestrator?

El orchestrator (`main.py`) es el **punto de entrada único** para ejecutar todo el pipeline MLOps de manera secuencial y automatizada.

### ✅ Ventajas

- **Un solo comando**: `python main.py` ejecuta todo
- **Configuración centralizada**: Todo en `config.yaml`
- **Tracking automático**: Integración con MLflow y W&B
- **Flexible**: Ejecuta solo los pasos que necesites
- **Reproducible**: Misma configuración = mismos resultados
- **Fácil de extender**: Agregar nuevos pasos es simple

### 🏗️ Arquitectura

```
cap2-end_to_end/
├── main.py                    # 🎯 ORCHESTRATOR (ejecuta todo)
├── config.yaml                # ⚙️  Configuración centralizada
├── .env                       # 🔐 Credenciales (no en git)
│
├── src/
│   └── data/
│       ├── 01_download_data/
│       │   ├── main.py        # Lógica del paso 1
│       │   ├── MLproject      # Definición MLflow
│       │   ├── downloader.py  # Implementación
│       │   └── models.py      # Modelos Pydantic
│       │
│       └── 02_preprocessing_and_imputation/
│           ├── main.py        # Lógica del paso 2
│           ├── MLproject      # Definición MLflow
│           ├── preprocessor.py
│           └── models.py
│
└── tests/                     # 🧪 Pruebas unitarias
```

---

## 🚀 Cómo Ejecutar

### Opción 1: Comando Directo (Más Simple)

```bash
python main.py
```

Esto ejecuta TODO el pipeline usando la configuración de `config.yaml`.

### Opción 2: Con Makefile

```bash
make run-all
```

### Opción 3: Probar Configuración Primero

```bash
# 1. Verificar que todo esté configurado correctamente
python test_pipeline.py

# 2. Si todo está OK, ejecutar pipeline
python main.py
```

### Opción 4: Ejecutar con Configuración Personalizada

```bash
# Cambiar parámetros desde línea de comando
python main.py preprocessing.imputation_strategy=median

# Ejecutar solo algunos pasos
python main.py main.execute_steps=[01_download_data]

# Usar archivo de configuración diferente
python main.py --config-name=config_dev
```

---

## ⚙️ Configuración

### Estructura de `config.yaml`

```yaml
main:
  project_name: "housing-mlops-gcp"
  experiment_name: "end_to_end_pipeline"
  execute_steps:
    - "01_download_data"
    - "02_preprocessing_and_imputation"
    # - "03_feature_engineering"  # Próximamente

download_data:
  file_url: "https://..."
  artifact_name: "housing_data_raw"
  gcs_output_path: "data/01-raw/housing.csv"

preprocessing:
  imputation_strategy: "auto"  # auto, median, mean, mode, drop
  create_features: "True"      # True/False

gcs:
  bucket_name: ${oc.env:GCS_BUCKET_NAME}  # Desde .env
  project_id: ${oc.env:GCP_PROJECT_ID}
```

### Variables de Entorno (`.env`)

```bash
# GCP
GCS_BUCKET_NAME=your-bucket-name
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# W&B
WANDB_PROJECT=housing-mlops-gcp
WANDB_API_KEY=your-api-key
```

---

## 📊 Flujo de Ejecución

```
┌─────────────────────────────────────────┐
│         python main.py                  │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│    1. Cargar config.yaml                │
│    2. Cargar variables .env             │
│    3. Configurar W&B y MLflow           │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  STEP 1: Download Data                  │
│  ┌───────────────────────────────────┐  │
│  │ mlflow.run("01_download_data")    │  │
│  │   - Descargar desde URL           │  │
│  │   - Extraer tar.gz                │  │
│  │   - Subir a GCS                   │  │
│  │   - Registrar en W&B              │  │
│  └───────────────────────────────────┘  │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  STEP 2: Preprocessing                  │
│  ┌───────────────────────────────────┐  │
│  │ mlflow.run("02_preprocessing")    │  │
│  │   - Descargar de GCS              │  │
│  │   - Analizar missing values       │  │
│  │   - Seleccionar mejor imputación  │  │
│  │   - Feature engineering           │  │
│  │   - Subir procesado a GCS         │  │
│  │   - Registrar métricas            │  │
│  └───────────────────────────────────┘  │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│         RESUMEN FINAL                   │
│  ✅ Todos los pasos completados         │
│  ⏱️  Tiempo total: 45.23s               │
│  📦 Datos en: gs://bucket/data/         │
│  📊 Métricas en: wandb.ai/project       │
└─────────────────────────────────────────┘
```

---

## 🔧 Agregar Nuevos Pasos

### Paso 1: Crear el Módulo

```bash
mkdir -p src/data/03_feature_engineering
cd src/data/03_feature_engineering
```

### Paso 2: Crear `MLproject`

```yaml
name: feature_engineering
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      input_artifact:
        type: string
      output_artifact:
        type: string
    command: "python main.py --input {input_artifact} --output {output_artifact}"
```

### Paso 3: Crear `main.py`

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Tu lógica aquí
    print(f"Processing {args.input} -> {args.output}")

if __name__ == "__main__":
    main()
```

### Paso 4: Actualizar `config.yaml`

```yaml
main:
  execute_steps:
    - "01_download_data"
    - "02_preprocessing_and_imputation"
    - "03_feature_engineering"  # ⬅️ Nuevo paso

feature_engineering:
  input_artifact: "housing_data_processed:latest"
  output_artifact: "housing_features"
```

### Paso 5: Actualizar `main.py` (orchestrator)

```python
def run_feature_engineering(config: DictConfig, root_path: Path) -> None:
    """Execute feature engineering step."""
    print("\n" + "=" * 70)
    print("  STEP 3: FEATURE ENGINEERING")
    print("=" * 70)

    step_path = root_path / "src" / "data" / "03_feature_engineering"

    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        parameters={
            "input_artifact": config["feature_engineering"]["input_artifact"],
            "output_artifact": config["feature_engineering"]["output_artifact"],
        },
    )

    print("\n✅ Feature engineering completed!\n")

# En la función go(), agregar:
if "03_feature_engineering" in steps_to_execute:
    run_feature_engineering(config, root_path)
```

### Paso 6: ¡Ejecutar!

```bash
python main.py
```

---

## 🎨 Personalización Avanzada

### Ejecutar Solo Un Paso

```bash
# Solo download
python main.py main.execute_steps=[01_download_data]

# Solo preprocessing
python main.py main.execute_steps=[02_preprocessing_and_imputation]
```

### Cambiar Parámetros

```bash
# Cambiar estrategia de imputación
python main.py preprocessing.imputation_strategy=median

# Desactivar feature engineering
python main.py preprocessing.create_features=False

# Cambiar bucket
python main.py gcs.bucket_name=otro-bucket
```

### Múltiples Configuraciones

```bash
# Crear config_dev.yaml, config_prod.yaml, etc.

# Ejecutar con config específico
python main.py --config-name=config_dev
python main.py --config-name=config_prod
```

### Override Desde Archivo

```yaml
# overrides.yaml
preprocessing:
  imputation_strategy: knn
  create_features: False
```

```bash
python main.py --config-path=. --config-name=config +override=overrides
```

---

## 📈 Monitoreo

### Durante la Ejecución

El orchestrator muestra en tiempo real:
- ✅ Cada paso que se completa
- ⏱️ Tiempo de ejecución
- 📊 Métricas importantes
- ❌ Errores si ocurren

### Después de la Ejecución

1. **GCS**: Ver archivos generados
   ```bash
   gsutil ls -lh gs://$GCS_BUCKET_NAME/data/**
   ```

2. **W&B**: Ver experimentos y artefactos
   - https://wandb.ai/your-entity/housing-mlops-gcp

3. **MLflow**: Ver runs y métricas
   ```bash
   mlflow ui
   # Abrir http://localhost:5000
   ```

---

## 🐛 Debugging

### Ver Configuración Cargada

```python
python -c "
import hydra
from omegaconf import OmegaConf

with hydra.initialize(config_path='.', version_base='1.3'):
    cfg = hydra.compose(config_name='config')
    print(OmegaConf.to_yaml(cfg))
"
```

### Modo Verbose

```bash
# Ver logs detallados de MLflow
export MLFLOW_TRACKING_URI=./mlruns
python main.py
```

### Dry Run

Modifica temporalmente `main.py` para solo imprimir sin ejecutar:

```python
# En cada función run_*
print(f"Would execute: mlflow.run({step_path})")
# return  # Comentar la ejecución real
```

---

## 📚 Recursos

- [Hydra Documentation](https://hydra.cc/)
- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)
- [W&B Artifacts](https://docs.wandb.ai/guides/artifacts)

---

## ✅ Checklist Pre-Ejecución

- [ ] `.env` configurado con credenciales
- [ ] `config.yaml` con pasos deseados
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Tests pasan (`python test_pipeline.py`)
- [ ] GCS bucket existe y es accesible
- [ ] Autenticado en GCP (`gcloud auth list`)

Si todo está ✅, ejecuta:

```bash
python main.py
```

**¡Y observa la magia! 🎉**
