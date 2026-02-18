# GitHub Actions Workflows

Workflows activos para `cap2-end_to_end`:

## 1) `mlops-complete-pipeline.yml`
Pipeline CI/CD principal.

### Qué hace
- **PR/Push**: quality gate estricto (ruff, format-check, tests, smoke de `mlflow run`).
- **Manual (`workflow_dispatch`)**:
  - Ejecuta pipeline completo (01-07) con `uv run mlflow run .`.
  - Genera artifacts de entrenamiento/registro.
  - Opcionalmente promueve modelo `Staging -> Production` con guardrails (`r2` mínimo y `mape` máximo).
  - Ejecuta smoke test de API en Docker usando el modelo entrenado.

### Inputs manuales
- `run_training_pipeline` (bool)
- `sweep_count` (string)
- `promote_to_production` (bool)
- `min_r2` (string)
- `max_mape` (string)

## 2) `mlops-model-monitoring.yml`
Monitoreo de drift por PSI.

### Qué hace
- **Schedule semanal** y ejecución manual.
- Lee datasets en GCS (referencia y actual).
- Calcula PSI por feature numérica.
- Genera `artifacts/monitoring/drift_report.json`.
- Puede fallar el workflow si detecta drift (`fail_on_drift=true`).

### Inputs manuales
- `reference_path`
- `current_path`
- `max_psi`
- `fail_on_drift`

## Secrets requeridos
- `GCP_SA_KEY`
- `GCP_PROJECT_ID`
- `GCS_BUCKET_NAME`
- `WANDB_API_KEY`
- `WANDB_ENTITY` (opcional)

## Versiones de Actions (actualizadas)
- `actions/checkout@v6`
- `actions/setup-python@v6`
- `astral-sh/setup-uv@v7`
- `google-github-actions/auth@v3`
- `actions/upload-artifact@v6`
- `actions/download-artifact@v7`
- `docker/build-push-action@v6`

## Notas
- El pipeline ya no usa `|| true` en checks críticos.
- La instalación de dependencias usa `uv sync --frozen`.
- MLflow usa backend SQLite (`mlflow.db`) por defecto para evitar warnings deprecados del file backend.
- Los backups legacy (`*.backup`) se mantienen solo como referencia.
