"""
FastAPI application for housing price prediction.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys

from app.core.config import Settings
from app.core.model_loader import ModelLoader
from app.core.wandb_logger import WandBLogger
from app.routers import predict
from app.models.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Loads the model on startup and cleans up on shutdown.
    """
    logger.info("Starting up API...")

    # Initialize W&B logger
    wandb_logger = WandBLogger(
        project=settings.WANDB_PROJECT,
        enabled=True
    )
    predict.set_wandb_logger(wandb_logger)

    # Initialize model loader
    model_loader = ModelLoader(
        local_model_path=settings.LOCAL_MODEL_PATH,
        gcs_bucket=settings.GCS_BUCKET if settings.GCS_BUCKET else None,
        gcs_model_path=settings.GCS_MODEL_PATH if settings.GCS_BUCKET else None,
        mlflow_model_name=settings.MLFLOW_MODEL_NAME if settings.MLFLOW_MODEL_NAME else None,
        mlflow_model_stage=settings.MLFLOW_MODEL_STAGE,
        mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI if settings.MLFLOW_TRACKING_URI else None
    )

    # Load model
    try:
        logger.info("Loading model...")
        model_loader.load_model()
        logger.info(f"Model loaded successfully: {model_loader.model_version}")

        # Set model loader in router
        predict.set_model_loader(model_loader)

        # Store in app state
        app.state.model_loader = model_loader
        app.state.wandb_logger = wandb_logger

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.warning("API will start but predictions will fail")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down API...")
    wandb_logger.close()


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for predicting California housing prices",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)


@app.get(
    "/",
    response_model=dict,
    tags=["root"],
    summary="Root endpoint",
    description="Returns basic API information"
)
async def root() -> dict:
    """Root endpoint providing basic API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "Housing Price Prediction API",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Check API health and model status"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    model_loaded = False
    if hasattr(app.state, 'model_loader'):
        model_loaded = app.state.model_loader.is_loaded

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=settings.VERSION
    )


@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
