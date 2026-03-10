"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.app.models.schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/predict", tags=["predict"])

_service = None


def set_model_service(service) -> None:
    global _service
    _service = service


@router.post("", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model service is not available")
    try:
        result = _service.predict(payload.instances)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictResponse(**result)
