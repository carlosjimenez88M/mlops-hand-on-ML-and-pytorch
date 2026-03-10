"""FastAPI application serving the registered model."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.app.core.config import Settings
from api.app.core.model_loader import ModelService
from api.app.models.schemas import HealthResponse
from api.app.routers.predict import router as predict_router
from api.app.routers.predict import set_model_service

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ModelService(
        tracking_uri=settings.mlflow_tracking_uri,
        model_name=settings.mlflow_model_name,
        model_alias=settings.mlflow_model_alias,
        local_model_path=settings.local_model_path,
        serving_metadata_path=settings.serving_metadata_path,
    )
    try:
        service.load()
        set_model_service(service)
        app.state.model_service = service
    except Exception:
        app.state.model_service = None
    yield


app = FastAPI(title=settings.project_name, version=settings.version, lifespan=lifespan)
app.include_router(predict_router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    service = getattr(app.state, "model_service", None)
    return HealthResponse(
        status="healthy" if service is not None else "degraded",
        model_loaded=service is not None,
        details={
            "model_name": settings.mlflow_model_name,
            "model_alias": settings.mlflow_model_alias,
        },
    )


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {
        "name": settings.project_name,
        "version": settings.version,
        "docs_url": "/docs",
        "health_url": "/health",
    }
