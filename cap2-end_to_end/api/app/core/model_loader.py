"""
Model loader for loading trained models from MLflow, GCS or local filesystem.
"""
import pickle
from pathlib import Path
from typing import Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading ML models from various sources."""

    def __init__(
        self,
        local_model_path: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        gcs_model_path: Optional[str] = None,
        mlflow_model_name: Optional[str] = None,
        mlflow_model_stage: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Initialize model loader.

        Args:
            local_model_path: Path to local model file
            gcs_bucket: GCS bucket name
            gcs_model_path: Path to model in GCS bucket
            mlflow_model_name: Name of registered model in MLflow
            mlflow_model_stage: Model stage (Staging/Production/None)
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.local_model_path = local_model_path
        self.gcs_bucket = gcs_bucket
        self.gcs_model_path = gcs_model_path
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_stage = mlflow_model_stage
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._model: Optional[Any] = None
        self._model_version: str = "unknown"

    def load_from_local(self, model_path: str) -> Any:
        """
        Load model from local filesystem.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from local path: {model_path}")

        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

            self._model_version = model_file.stem
            logger.info(f"Model loaded successfully: {self._model_version}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")

    def load_from_gcs(self, bucket_name: str, blob_path: str) -> Any:
        """
        Load model from Google Cloud Storage.

        Args:
            bucket_name: GCS bucket name
            blob_path: Path to model blob in bucket

        Returns:
            Loaded model object

        Raises:
            ImportError: If google-cloud-storage is not installed
            Exception: If model loading fails
        """
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS loading. "
                "Install with: pip install google-cloud-storage"
            )

        logger.info(f"Loading model from GCS: gs://{bucket_name}/{blob_path}")

        try:
            # Initialize GCS client
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Download model to memory
            model_bytes = blob.download_as_bytes()
            model = pickle.loads(model_bytes)

            self._model_version = Path(blob_path).stem
            logger.info(f"Model loaded successfully from GCS: {self._model_version}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model from GCS: {str(e)}")
            raise Exception(f"GCS model loading failed: {str(e)}")

    def load_from_mlflow(
        self,
        model_name: str,
        stage: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ) -> Any:
        """
        Load model from MLflow Model Registry.

        Args:
            model_name: Name of registered model
            stage: Model stage (Staging/Production/None for latest version)
            tracking_uri: MLflow tracking server URI

        Returns:
            Loaded model object

        Raises:
            ImportError: If mlflow is not installed
            Exception: If model loading fails
        """
        try:
            import mlflow
            import mlflow.pyfunc
        except ImportError:
            raise ImportError(
                "mlflow is required for MLflow model loading. "
                "Install with: pip install mlflow"
            )

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
        else:
            logger.info("Using default MLflow tracking URI")

        try:
            # Build model URI
            if stage:
                model_uri = f"models:/{model_name}/{stage}"
                logger.info(f"Loading model from MLflow: {model_uri}")
            else:
                # Get latest version
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                latest_versions = client.get_latest_versions(model_name, stages=["Staging", "Production"])

                if not latest_versions:
                    # Try to get any version
                    all_versions = client.search_model_versions(f"name='{model_name}'")
                    if not all_versions:
                        raise Exception(f"No versions found for model: {model_name}")
                    latest_version = max(all_versions, key=lambda v: int(v.version))
                else:
                    # Prefer Production over Staging
                    production_versions = [v for v in latest_versions if v.current_stage == "Production"]
                    latest_version = production_versions[0] if production_versions else latest_versions[0]

                model_uri = f"models:/{model_name}/{latest_version.version}"
                logger.info(f"Loading model from MLflow: {model_uri} (stage: {latest_version.current_stage})")

            # Load model
            model = mlflow.sklearn.load_model(model_uri)

            # Extract version info
            self._model_version = model_uri
            logger.info(f"Model loaded successfully from MLflow: {self._model_version}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {str(e)}")
            raise Exception(f"MLflow model loading failed: {str(e)}")

    def load_model(self) -> Any:
        """
        Load model using configured settings.
        Priority: MLflow > GCS > Local

        Returns:
            Loaded model object

        Raises:
            ValueError: If no model path is configured
            Exception: If all loading attempts fail
        """
        if self._model is not None:
            logger.info("Returning cached model")
            return self._model

        # Try MLflow first if configured
        if self.mlflow_model_name:
            try:
                self._model = self.load_from_mlflow(
                    self.mlflow_model_name,
                    self.mlflow_model_stage,
                    self.mlflow_tracking_uri
                )
                return self._model
            except Exception as e:
                logger.warning(f"MLflow loading failed, trying GCS: {str(e)}")

        # Try GCS if configured
        if self.gcs_bucket and self.gcs_model_path:
            try:
                self._model = self.load_from_gcs(
                    self.gcs_bucket,
                    self.gcs_model_path
                )
                return self._model
            except Exception as e:
                logger.warning(f"GCS loading failed, trying local: {str(e)}")

        # Fallback to local
        if self.local_model_path:
            try:
                self._model = self.load_from_local(self.local_model_path)
                return self._model
            except Exception as e:
                logger.error(f"Local loading failed: {str(e)}")
                raise

        raise ValueError(
            "No model path configured. Set either mlflow_model_name, "
            "gcs_bucket + gcs_model_path, or local_model_path"
        )

    def predict(self, features: Any) -> Any:
        """
        Make predictions using loaded model.

        Args:
            features: Input features for prediction

        Returns:
            Model predictions

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return self._model.predict(features)

    @property
    def model_version(self) -> str:
        """Get the loaded model version."""
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
