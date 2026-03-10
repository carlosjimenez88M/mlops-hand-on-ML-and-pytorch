"""Dynamic registry of MLflow project steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from omegaconf import DictConfig

from cap3_classification.utils.paths import find_project_root, resolve_path

ParameterBuilder = Callable[[DictConfig, Path], dict[str, object]]


@dataclass(frozen=True)
class StepDefinition:
    step_id: str
    module_name: str
    parameter_builder: ParameterBuilder

    def step_path(self) -> Path:
        module = import_module(self.module_name)
        module_path = Path(module.__file__).resolve()
        return module_path.parent


def _identity_builder(config_section: DictConfig, root_path: Path) -> dict[str, object]:
    del root_path
    return {key: value for key, value in config_section.items()}


def _registration_builder(config_section: DictConfig, root_path: Path) -> dict[str, object]:
    payload = {key: value for key, value in config_section.items()}
    best_params_path = payload["best_params_path"]
    payload["best_params_path"] = str(resolve_path(best_params_path, root_path))
    return payload


STEP_DEFINITIONS = {
    "01_ingest": StepDefinition(
        step_id="01_ingest",
        module_name="cap3_classification.steps.ingest.main",
        parameter_builder=lambda config, root_path: _identity_builder(config["dataset"], root_path),
    ),
    "02_validate": StepDefinition(
        step_id="02_validate",
        module_name="cap3_classification.steps.validate.main",
        parameter_builder=lambda config, root_path: _identity_builder(
            config["validation"], root_path
        ),
    ),
    "03_feature_engineering": StepDefinition(
        step_id="03_feature_engineering",
        module_name="cap3_classification.steps.feature_engineering.main",
        parameter_builder=lambda config, root_path: _identity_builder(
            config["features"], root_path
        ),
    ),
    "04_split": StepDefinition(
        step_id="04_split",
        module_name="cap3_classification.steps.split.main",
        parameter_builder=lambda config, root_path: _identity_builder(config["split"], root_path),
    ),
    "05_model_competition": StepDefinition(
        step_id="05_model_competition",
        module_name="cap3_classification.steps.model_competition.main",
        parameter_builder=lambda config, root_path: {
            **_identity_builder(config["competition"], root_path),
            "candidate_models": ",".join(list(config["competition"]["candidate_models"])),
        },
    ),
    "06_sweep": StepDefinition(
        step_id="06_sweep",
        module_name="cap3_classification.steps.sweep.main",
        parameter_builder=lambda config, root_path: {
            **_identity_builder(config["sweep"], root_path),
            "wandb_project": config["wandb"]["project"],
            "wandb_mode": config["wandb"]["mode"],
        },
    ),
    "07_register": StepDefinition(
        step_id="07_register",
        module_name="cap3_classification.steps.register.main",
        parameter_builder=lambda config, root_path: {
            **_registration_builder(config["registration"], root_path),
            "wandb_project": config["wandb"]["project"],
        },
    ),
}


def get_project_root() -> Path:
    """Return project root for orchestrator and tests."""
    return find_project_root()
