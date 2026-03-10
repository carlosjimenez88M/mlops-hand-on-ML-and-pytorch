"""Final training and MLflow registration helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from cap3_classification.modeling.evaluation import evaluate_predictions, split_xy
from cap3_classification.modeling.factory import ModelFactory
from cap3_classification.schemas import RegisteredModelArtifact
from cap3_classification.utils.paths import ensure_parent


def register_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    selected_model_name: str,
    best_params: dict,
    registered_model_name: str,
    model_stage: str,
    local_model_output_path: Path,
) -> RegisteredModelArtifact:
    """Train the selected model, register it in MLflow and persist serving metadata."""
    combined_train = pd.concat([train_df, validation_df], ignore_index=True)
    x_train, y_train = split_xy(combined_train, target_column=target_column)
    x_test, y_test = split_xy(test_df, target_column=target_column)

    clean_params = ModelFactory.sanitize_params(selected_model_name, best_params)
    model = ModelFactory.create_candidate(
        model_name=selected_model_name,
        random_state=42,
        params=clean_params,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = evaluate_predictions(y_true=y_test, y_pred=predictions)

    signature = infer_signature(x_test.head(5), predictions[:5])
    mlflow.log_param("model_name", selected_model_name)
    mlflow.log_params(clean_params)
    mlflow.log_metrics(metrics)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=x_test.head(3),
        registered_model_name=registered_model_name,
    )

    ensure_parent(local_model_output_path)
    joblib.dump(model, local_model_output_path)

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("Registration requires an active MLflow run")

    client = MlflowClient()
    versions = [
        version
        for version in client.search_model_versions(f"name='{registered_model_name}'")
        if version.run_id == run.info.run_id
    ]
    if not versions:
        raise RuntimeError(
            "MLflow did not return a registered model version for the current run. "
            f"Logged model URI: {model_info.model_uri}"
        )
    version = max(
        versions,
        key=lambda model_version: int(model_version.version),
    )
    alias = model_stage.lower()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version.version,
    )

    return RegisteredModelArtifact(
        registered_model_name=registered_model_name,
        alias=alias,
        model_version=str(version.version),
        run_id=run.info.run_id,
        model_name=selected_model_name,
        target_column=target_column,
        input_dim=int(x_train.shape[1]),
        class_labels=sorted(combined_train[target_column].unique().tolist()),
        metrics=metrics,
        local_model_path=str(local_model_output_path),
    )
