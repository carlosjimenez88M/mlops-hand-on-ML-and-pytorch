# GitHub Actions Workflows

Este directorio contiene los workflows de CI/CD para el proyecto MLOps.

## 🚀 Workflows Activos

### 1. MLOps Complete Pipeline (`mlops-complete-pipeline.yml`)
**Propósito**: Pipeline completo end-to-end de MLOps con 7 pasos.

**Triggers**:
- Push a `master` o `main`
- Pull requests
- Manual trigger (workflow_dispatch)

**Pasos del Pipeline**:
1. **Setup & Validation** - Valida estructura del proyecto e instala dependencias
2. **Code Quality** - Linting y formateo con Ruff
3. **Unit Tests** - Tests unitarios con coverage
4. **Step 01: Download Data** - Descarga datos desde fuente externa
5. **Step 02: Preprocessing** - Limpieza e imputación de datos
6. **Step 03: Feature Engineering** - Creación de features y clustering
7. **Step 04: Data Segregation** - Split train/test con visualizaciones
8. **Step 05: Model Selection** - Entrenamiento de 5 modelos y selección del mejor
9. **Step 06: Hyperparameter Sweep** - Optimización con W&B (default: 5 runs)
10. **Step 07: Model Registration** - Registro del modelo final en MLflow
11. **Docker Build & Test** - Construcción y test de imagen Docker
12. **Pipeline Summary** - Resumen completo del pipeline

**Variables Requeridas** (GitHub Secrets):
```
GCP_SERVICE_ACCOUNT_KEY  # JSON key de GCP service account
GCS_BUCKET_NAME          # Nombre del bucket GCS
GCP_PROJECT_ID           # ID del proyecto GCP
WANDB_API_KEY            # API key de Weights & Biases
WANDB_ENTITY             # Entity de W&B (opcional)
```

**Cómo ejecutar manualmente**:
1. Ve a Actions → MLOps CI/CD Pipeline
2. Click en "Run workflow"
3. Selecciona opciones:
   - `run_full_pipeline`: true para ejecutar todos los pasos
   - `sweep_count`: número de runs del sweep (default: 5)

**Artifacts generados**:
- Outputs de cada paso (7 días de retención)
- Modelos entrenados (30 días de retención)
- Modelo registrado final (90 días de retención)

---

### 2. Deploy to Cloud Run (`deploy-cloud-run.yml`)
**Propósito**: Deployment automático de la API a Google Cloud Run.

**Triggers**:
- Push a `master`/`main` que modifique `cap2-end_to_end/api/**`
- Manual trigger con selección de environment (staging/production)

**Pasos**:
1. Tests pre-deployment
2. Build de imagen Docker
3. Scan de vulnerabilidades
4. Push a Artifact Registry
5. Deploy a Cloud Run
6. Health checks

---

## 📋 Workflows Archivados (Backups)

Los siguientes workflows fueron archivados para evitar duplicados:

- `mlops-pipeline-manual.yml.backup` - Version anterior del pipeline manual
- `pr-tests.yml.backup` - Tests de PR (ahora integrados en mlops-complete-pipeline.yml)

**No borrar estos archivos** - contienen configuraciones de respaldo.

---

## 🔧 Troubleshooting

### Error: "Could not open requirements file"
**Solución**: El proyecto usa `pyproject.toml`, no `requirements.txt`. El workflow ya está configurado correctamente con `uv pip install --system -e .`

### Error: "best_params.yaml not found"
**Solución**: Asegúrate de que Step 06 (Sweep) completó exitosamente antes de ejecutar Step 07 (Registration). El workflow ya incluye validaciones.

### Error: "GOOGLE_APPLICATION_CREDENTIALS not found"
**Solución**: El workflow establece esta variable como vacía (`GOOGLE_APPLICATION_CREDENTIALS=`) porque la autenticación se hace vía `google-github-actions/auth@v2`.

### Pipeline tarda mucho
**Solución**:
- Reduce `sweep_count` a 5 o menos en el input manual
- Los pasos 5-6 son los más lentos (entrenan múltiples modelos)

---

## 📊 Monitoreo

### Weights & Biases
Todos los experimentos se loguean a W&B:
```
https://wandb.ai/[ENTITY]/housing-mlops-gcp
```

### MLflow
Los artifacts de MLflow se guardan en los artifacts del workflow:
- Descarga `step-07-registered-model` artifact
- Ejecuta `mlflow ui --backend-store-uri ./mlruns`

---

## 🎯 Best Practices

1. **Siempre ejecuta tests primero**: El workflow lo hace automáticamente
2. **Valida localmente antes de push**: `make run-tests` en tu máquina
3. **Usa manual trigger para experimentar**: Evita ejecutar el pipeline completo en cada push
4. **Revisa los artifacts**: Contienen información valiosa de cada paso
5. **Monitorea W&B**: Para ver métricas detalladas y comparar experimentos

---

## 📝 Notas de Versión

### v2.0 (2026-01-13)
- ✅ Workflow único y robusto con todos los pasos
- ✅ Validaciones en cada paso
- ✅ Manejo correcto de artifacts entre pasos
- ✅ Fix de path absoluto para best_params.yaml
- ✅ Feature importance plot en Step 07
- ✅ Sweep reducido a 5 runs por default
- ✅ Artifacts de segregation con plots y tables

### v1.0
- Primera versión con workflows separados (archivados)
