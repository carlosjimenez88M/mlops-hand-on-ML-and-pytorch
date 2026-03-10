"""MLflow setup helpers."""

from __future__ import annotations

from pathlib import Path

import mlflow

from cap3_classification.utils.paths import resolve_path


def configure_mlflow(
    root_path: Path, tracking_uri: str, registry_uri: str, experiment_name: str
) -> None:
    """Configure MLflow tracking and registry backends."""
    if tracking_uri.startswith("sqlite:///mlflow.db"):
        tracking_uri = f"sqlite:///{resolve_path('mlflow.db', root_path)}"
    if registry_uri.startswith("sqlite:///mlflow.db"):
        registry_uri = tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(experiment_name)
