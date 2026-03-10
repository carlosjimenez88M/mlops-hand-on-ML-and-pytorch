FROM python:3.12-slim AS base

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY api ./api
COPY tests ./tests
COPY main.py ./
COPY conf ./conf
COPY scripts ./scripts
COPY MLproject ./

RUN pip install --no-cache-dir uv
RUN uv sync --extra dev

FROM base AS test

CMD ["uv", "run", "pytest", "tests", "api/tests", "--cov=src/cap3_classification", "--cov=api/app", "--cov-report=term-missing"]

FROM base AS runtime

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
