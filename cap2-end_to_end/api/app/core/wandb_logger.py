"""
Weights & Biases logger for API predictions monitoring.
"""
import os
import logging
from typing import Dict, List, Optional
import wandb
from datetime import datetime

logger = logging.getLogger(__name__)


class WandBLogger:
    """Logger for tracking API predictions in Weights & Biases."""

    def __init__(
        self,
        project: str = "housing-mlops-api",
        enabled: bool = True
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            enabled: Whether to enable W&B logging
        """
        self.enabled = enabled and bool(os.getenv("WANDB_API_KEY"))
        self.project = project
        self._run = None

        if self.enabled:
            try:
                # Initialize W&B in service mode for APIs
                self._run = wandb.init(
                    project=self.project,
                    job_type="api-inference",
                    config={
                        "environment": os.getenv("ENVIRONMENT", "production"),
                        "model_version": os.getenv("MODEL_VERSION", "unknown")
                    },
                    reinit=True
                )
                logger.info(f"W&B logging enabled for project: {self.project}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {str(e)}")
                self.enabled = False
        else:
            logger.info("W&B logging disabled")

    def log_prediction(
        self,
        features: List[Dict],
        predictions: List[float],
        model_version: str,
        response_time_ms: float
    ) -> None:
        """
        Log prediction request to W&B.

        Args:
            features: Input features
            predictions: Model predictions
            model_version: Version of model used
            response_time_ms: API response time in milliseconds
        """
        if not self.enabled:
            return

        try:
            # Create summary statistics
            wandb.log({
                "prediction/count": len(predictions),
                "prediction/mean": sum(predictions) / len(predictions),
                "prediction/min": min(predictions),
                "prediction/max": max(predictions),
                "performance/response_time_ms": response_time_ms,
                "model/version": model_version,
                "timestamp": datetime.now().isoformat()
            })

            # Log feature distributions (sample first 100 predictions)
            if len(features) <= 100:
                for i, (feat, pred) in enumerate(zip(features, predictions)):
                    wandb.log({
                        f"features/instance_{i}/median_income": feat.get("median_income", 0),
                        f"features/instance_{i}/housing_median_age": feat.get("housing_median_age", 0),
                        f"predictions/instance_{i}": pred
                    })

        except Exception as e:
            logger.error(f"Failed to log prediction to W&B: {str(e)}")

    def log_error(
        self,
        error_type: str,
        error_message: str,
        features: Optional[List[Dict]] = None
    ) -> None:
        """
        Log prediction error to W&B.

        Args:
            error_type: Type of error
            error_message: Error message
            features: Input features that caused error (for debugging)
        """
        if not self.enabled:
            return

        try:
            log_data = {
                "error/type": error_type,
                "error/message": error_message,
                "error/count": 1,
                "timestamp": datetime.now().isoformat()
            }

            # Optionally log feature count if provided
            if features:
                log_data["error/feature_count"] = len(features)

            wandb.log(log_data)
        except Exception as e:
            logger.error(f"Failed to log error to W&B: {str(e)}")

    def log_health_check(self, status: str, model_loaded: bool) -> None:
        """
        Log health check status.

        Args:
            status: Health status
            model_loaded: Whether model is loaded
        """
        if not self.enabled:
            return

        try:
            wandb.log({
                "health/status": status,
                "health/model_loaded": 1 if model_loaded else 0,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to log health check to W&B: {str(e)}")

    def close(self) -> None:
        """Close W&B run."""
        if self.enabled and self._run:
            try:
                self._run.finish()
            except Exception as e:
                logger.error(f"Failed to close W&B run: {str(e)}")
