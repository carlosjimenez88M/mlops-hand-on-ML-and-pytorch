# cap3-classification

Pipeline MLOps modular para el capítulo 3 de clasificación de Hands-On Machine Learning. Toma el caso de `MNIST` del notebook de Géron y prioriza una competencia de modelos coherente con ese capítulo: modelos lineales para clasificación de dígitos, vecinos cercanos, ensambles y una red neuronal fully connected en PyTorch. Si `MNIST` no está disponible por red, hace fallback a `digits` de scikit-learn para desarrollo, pruebas y CI.

## Qué contiene

- Pipeline por etapas con `MLflow Projects`:
  - `01_ingest`
  - `02_validate`
  - `03_feature_engineering`
  - `04_split`
  - `05_model_competition`
  - `06_sweep`
  - `07_register`
- Competencia de modelos base alineada con clasificación de dígitos:
  - `DummyClassifier`
  - `SGDClassifier`
  - `LinearSVC`
  - `KNeighborsClassifier`
  - `RandomForestClassifier`
  - `TorchMLPClassifier`
- Selección del mejor baseline y ajuste posterior con `W&B Sweeps` usando `method: bayes` cuando el entorno está online.
- Fallback local para sweep cuando se trabaja offline o en CI sin acceso al servicio administrado de W&B.
- Registro final del mejor modelo en `MLflow Model Registry`.
- API de inferencia en `FastAPI` cargando desde MLflow Registry con fallback a artefacto local.
- Monitoreo de data drift con `PSI`.
- Tests unitarios y de API, más cobertura dentro del mismo contenedor Docker de la API.
- Automatización con `uv`, `Makefile` y `GitHub Actions`.

## Arquitectura

```text
conf/config.yaml
main.py
src/cap3_classification/
  data/
  modeling/
  monitoring/
  tracking/
  steps/
api/
scripts/
tests/
```

Cada etapa es un `MLproject` independiente y el orquestador resuelve dinámicamente la ruta del paso desde su módulo Python. No hay rutas de steps hardcodeadas en el orquestador.

## Instalación

```bash
cd /Users/carlosdaniel/Documents/Projects/Personales/Machine_Learning/mlops-hand-on-ML-and-pytorch/cap3-classification
uv sync --extra dev
cp .env.example .env
```

## Ejecución

Pipeline completo por el entrypoint raíz:

```bash
make run-pipeline
# equivalente a:
uv run mlflow run . --env-manager local
```

En la validación local de esta rama, el pipeline por defecto corrió completo con `MNIST`, registró el mejor modelo en MLflow y tardó varios minutos porque evalúa una competencia realista antes del sweep. Para una regresión rápida de desarrollo usa `make smoke`.

Smoke local sin depender de MNIST online:

```bash
make smoke
```

Sólo competencia de modelos:

```bash
make run-competition
```

Competencia + sweep:

```bash
make run-sweep
```

Drift report:

```bash
make monitor-drift
```

## W&B y MLflow

- `MLflow` se usa como tracking local y registry local sobre `sqlite:///mlflow.db`.
- `W&B` se usa para tracking de experimentos.
- Si `WANDB_MODE=online`, el paso `06_sweep` ejecuta un `W&B Sweep` real con `method: bayes`.
- Si `WANDB_MODE=offline` o `disabled`, el pipeline activa un `shadow sweep` local con el mismo search space para que desarrollo y CI sigan siendo reproducibles sin nube.

El flujo principal está pensado para ejecutarse con `mlflow run .`, mientras que internamente el orquestador dispara cada etapa como un `MLflow Project` independiente para conservar trazabilidad por componente.

## API de inferencia

Arranque local:

```bash
make api-local
```

Swagger:

- `http://localhost:8080/docs`

Payload:

```json
{
  "instances": [[0.0, 1.0, 2.0, 3.0]]
}
```

La longitud de cada instancia debe coincidir con `input_dim` del metadata generado en `artifacts/serving/model_info.json`.

## Docker

Build:

```bash
make docker-api-build
```

Tests y coverage dentro del Docker:

```bash
make docker-api-test
```

Run:

```bash
make docker-api-run
```

## Drift

El script `scripts/monitor_drift.py` compara un dataset de referencia contra uno actual y genera `reports/drift/drift_report.json` usando `PSI`. En un flujo más maduro, este script se puede disparar después de scoring batch o desde datos de serving capturados.

## GitHub Actions

El workflow `cap3-classification-ci.yml` hace:

- lint
- tests + coverage
- smoke offline del pipeline
- verificación del registro del modelo
- build del contenedor de API y ejecución de tests dentro del mismo

## Buenas prácticas aplicadas

- Configuración centralizada en YAML.
- Orquestación sin rutas hardcodeadas a steps.
- Separación entre lógica reusable y entrypoints.
- Competencia de modelos más fiel al problema original de clasificación de dígitos.
- Registro explícito del modelo final y metadata de serving.
- Fallbacks offline para mantener ejecutabilidad local.
- API desacoplada del entrenamiento, cargando desde registry o artefacto local.
- Drift monitoring versionable y ejecutable por CLI.
