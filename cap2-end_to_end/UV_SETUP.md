# UV Setup Guide - Using UV Instead of Pip

Este proyecto usa **UV** como gestor de paquetes en lugar de pip para instalación más rápida y gestión de dependencias mejorada.

---

## ¿Qué es UV?

UV es un gestor de paquetes Python extremadamente rápido escrito en Rust, compatible con pip y diseñado para ser su reemplazo directo.

**Beneficios:**
- 10-100x más rápido que pip
- Compatible con pip (usa la misma API)
- Mejor resolución de dependencias
- Caché inteligente
- Instalaciones reproducibles

---

## Instalación de UV

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Verificar instalación
```bash
uv --version
```

---

## Uso en Este Proyecto

### 1. Instalación Inicial del Proyecto

```bash
# Desde el directorio raíz del proyecto
uv pip install --system -e .
```

**Nota:** El flag `--system` instala en el Python del sistema. Si usas un venv, puedes omitirlo.

### 2. Trabajar con Virtual Environments

#### Opción A: Usar venv existente con UV
```bash
# Crear venv
python -m venv .venv

# Activar
source .venv/bin/activate  # macOS/Linux
# o
.venv\Scripts\activate  # Windows

# Instalar dependencias con UV
uv pip install -e .
```

#### Opción B: Crear venv con UV (más rápido)
```bash
# UV puede crear y gestionar venvs automáticamente
uv venv .venv

# Activar
source .venv/bin/activate

# Instalar dependencias
uv pip install -e .
```

### 3. Instalación en Submódulos (06_sweep, 07_registration)

Cada submódulo tiene su `requirements.txt`. Para instalarlos con UV:

```bash
# Step 06: Sweep
cd src/model/06_sweep
uv pip install -r requirements.txt

# Step 07: Registration
cd ../07_registration
uv pip install -r requirements.txt
```

**O desde el root del proyecto:**
```bash
uv pip install -r src/model/06_sweep/requirements.txt
uv pip install -r src/model/07_registration/requirements.txt
```

---

## Comandos UV Comunes

### Instalar paquete específico
```bash
uv pip install pandas
uv pip install "pandas>=2.0.0"
```

### Instalar desde requirements.txt
```bash
uv pip install -r requirements.txt
```

### Instalar proyecto editable
```bash
uv pip install -e .
```

### Listar paquetes instalados
```bash
uv pip list
```

### Desinstalar paquete
```bash
uv pip uninstall pandas
```

### Congelar dependencias
```bash
uv pip freeze > requirements.txt
```

### Sincronizar dependencias (instalar exactamente lo especificado)
```bash
uv pip sync requirements.txt
```

---

## Integración con CI/CD

### GitHub Actions

Nuestro workflow ya usa UV:

```yaml
- name: 📦 Install uv package manager
  run: |
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "$HOME/.cargo/bin" >> $GITHUB_PATH

- name: 🔧 Install dependencies
  run: |
    uv pip install --system -e .
```

---

## Conversión de Comandos Pip a UV

| Comando Pip | Comando UV |
|------------|------------|
| `pip install pandas` | `uv pip install pandas` |
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| `pip install -e .` | `uv pip install -e .` |
| `pip list` | `uv pip list` |
| `pip freeze` | `uv pip freeze` |
| `pip uninstall pandas` | `uv pip uninstall pandas` |

**Nota:** Simplemente reemplaza `pip` con `uv pip`.

---

## Estructura del Proyecto con UV

```
cap2-end_to_end/
├── pyproject.toml          # Dependencias principales (usa uv pip install -e .)
├── .venv/                  # Virtual environment (creado con uv venv)
│
├── src/model/
│   ├── 06_sweep/
│   │   ├── requirements.txt  # Dependencias específicas
│   │   └── conda.yaml        # Para MLflow (puede usar uv dentro)
│   │
│   └── 07_registration/
│       ├── requirements.txt  # Dependencias específicas
│       └── conda.yaml        # Para MLflow (puede usar uv dentro)
│
└── api/
    ├── requirements.txt      # Dependencias de API
    └── Dockerfile           # Puede usar uv para build más rápido
```

---

## MLflow con UV

MLflow puede ejecutar proyectos con diferentes gestores de paquetes.

### Opción 1: Usar conda.yaml (actual)
MLflow usa conda, pero dentro del conda.yaml podemos usar pip que puede ser reemplazado por uv:

```yaml
name: sweep
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - scikit-learn>=1.0.0
      - pandas>=2.0.0
      # etc...
```

### Opción 2: Usar UV directamente en MLproject

En lugar de `conda_env`, podemos especificar comandos personalizados:

```yaml
name: hyperparameter_sweep

entry_points:
  main:
    parameters:
      # ... parámetros ...

    command: |
      uv pip install -r requirements.txt &&
      python main.py {parameters}
```

**Nota:** La implementación actual usa conda.yaml para compatibilidad con MLflow, pero las instalaciones dentro siguen siendo con pip (que puede ser UV).

---

## Dockerfile con UV (Opcional)

Para builds de Docker más rápidos:

```dockerfile
FROM python:3.12-slim

# Instalar UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copiar dependencias
COPY requirements.txt .

# Instalar con UV (mucho más rápido)
RUN uv pip install --system -r requirements.txt

# ... resto del Dockerfile ...
```

---

## Troubleshooting

### Error: "uv: command not found"

**Solución:** Asegúrate de que UV está instalado y en tu PATH:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### Error: "No module named 'xyz'"

**Solución:** Verifica que instalaste las dependencias:
```bash
uv pip list  # Ver paquetes instalados
uv pip install -e .  # Reinstalar proyecto
```

### UV es lento en primera ejecución

**Normal:** UV construye su caché en la primera ejecución. Las siguientes serán mucho más rápidas.

### Conflictos con pip instalaciones previas

**Solución:** Crea un venv limpio con UV:
```bash
rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

---

## Migración Completa a UV (Para referencia)

Si queremos migrar completamente el proyecto a UV (opcional):

1. **Reemplazar todos los comandos pip con uv pip**
2. **Actualizar GitHub Actions** (ya hecho ✅)
3. **Actualizar Makefiles:**
   ```makefile
   install:
       uv pip install -e .
   ```
4. **Actualizar documentación** (este archivo)
5. **Opcional:** Crear `uv.lock` para dependencias reproducibles

---

## Estado Actual del Proyecto

✅ **GitHub Actions**: Ya usa UV
✅ **Documentación**: Incluye referencias a UV
✅ **pyproject.toml**: Compatible con UV
⏳ **MLflow projects**: Usan conda pero pueden usar UV internamente
⏳ **Docker**: Puede actualizarse para usar UV (opcional)

---

## Comando Rápido de Referencia

```bash
# Setup inicial
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Ejecutar pipeline
python main.py

# Instalar dependencias de submódulos
uv pip install -r src/model/06_sweep/requirements.txt
uv pip install -r src/model/07_registration/requirements.txt

# Actualizar dependencias
uv pip install --upgrade -e .
```

---

## Recursos

- **UV Documentation**: https://github.com/astral-sh/uv
- **Installation Guide**: https://astral.sh/uv
- **Comparison with pip**: https://github.com/astral-sh/uv#comparison

---

## Conclusión

UV es un reemplazo drop-in para pip con mejor performance. El proyecto está configurado para usar UV, especialmente en GitHub Actions. Para uso local, simplemente reemplaza `pip` con `uv pip` en cualquier comando de instalación.

**Comando más importante:**
```bash
uv pip install -e .
```

Este comando instala el proyecto completo con todas sus dependencias, de la forma más rápida posible.
