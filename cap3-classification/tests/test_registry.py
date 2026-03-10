"""Tests for the step registry and path resolution."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from cap3_classification.pipeline_registry import STEP_DEFINITIONS


def test_step_paths_exist():
    for definition in STEP_DEFINITIONS.values():
        assert definition.step_path().exists()
        assert (definition.step_path() / "MLproject").exists()


def test_registration_builder_makes_best_params_absolute():
    root_path = Path("/tmp/project")
    config = OmegaConf.create(
        {
            "registration": {
                "best_params_path": "artifacts/sweeps/best_params.yaml",
                "train_path": "train.parquet",
                "validation_path": "validation.parquet",
                "test_path": "test.parquet",
                "registered_model_name": "demo",
                "model_stage": "Staging",
                "target_column": "target",
                "local_model_output_path": "models/model.joblib",
                "serving_metadata_path": "artifacts/model.json",
            },
            "wandb": {"project": "demo"},
        }
    )
    payload = STEP_DEFINITIONS["07_register"].parameter_builder(config, root_path)
    assert Path(payload["best_params_path"]).is_absolute()
