"""
Prediction router for housing price predictions.
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
import logging

from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    ErrorResponse
)
from app.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Global model loader instance (initialized in main.py)
model_loader: ModelLoader = None


def set_model_loader(loader: ModelLoader) -> None:
    """Set the global model loader instance."""
    global model_loader
    model_loader = loader


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
    summary="Predict housing prices",
    description="Make predictions for housing prices based on input features"
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict housing prices for given features.

    Args:
        request: Prediction request with housing features

    Returns:
        PredictionResponse with predicted prices

    Raises:
        HTTPException: If prediction fails
    """
    if model_loader is None or not model_loader.is_loaded:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded"
        )

    try:
        # Convert input features to DataFrame
        features_list = []
        for instance in request.instances:
            features_list.append({
                'longitude': instance.longitude,
                'latitude': instance.latitude,
                'housing_median_age': instance.housing_median_age,
                'total_rooms': instance.total_rooms,
                'total_bedrooms': instance.total_bedrooms,
                'population': instance.population,
                'households': instance.households,
                'median_income': instance.median_income,
                'ocean_proximity': instance.ocean_proximity
            })

        df = pd.DataFrame(features_list)

        # Make predictions
        predictions = model_loader.predict(df)

        # Format response
        results = [
            PredictionResult(
                predicted_price=float(pred),
                confidence_interval=None
            )
            for pred in predictions
        ]

        return PredictionResponse(
            predictions=results,
            model_version=model_loader.model_version
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/model/info",
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description="Get information about the loaded model"
)
async def model_info() -> JSONResponse:
    """
    Get information about the loaded model.

    Returns:
        Model information including version and status
    """
    if model_loader is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_initialized",
                "model_loaded": False,
                "model_version": None
            }
        )

    return JSONResponse(
        content={
            "status": "ready" if model_loader.is_loaded else "not_loaded",
            "model_loaded": model_loader.is_loaded,
            "model_version": model_loader.model_version if model_loader.is_loaded else None
        }
    )
