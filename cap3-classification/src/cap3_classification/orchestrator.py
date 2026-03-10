"""Config-driven MLflow orchestrator for the modular chapter 3 pipeline."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from cap3_classification.pipeline_registry import STEP_DEFINITIONS, get_project_root
from cap3_classification.tracking.mlflow_utils import configure_mlflow
from cap3_classification.tracking.wandb_utils import configure_wandb_env
from cap3_classification.utils.logging import get_logger

logger = get_logger(__name__)


def _load_environment() -> None:
    if any(key for key in os.environ if key not in {"PATH", "PWD", "SHELL", "HOME"}):
        load_dotenv()


def _steps_to_execute(config: DictConfig) -> list[str]:
    steps = config["main"]["execute_steps"]
    return (
        list(steps) if not isinstance(steps, str) else [item.strip() for item in steps.split(",")]
    )


def _run_step(step_id: str, config: DictConfig, root_path: Path) -> None:
    step = STEP_DEFINITIONS[step_id]
    step_path = step.step_path()
    parameters = step.parameter_builder(config, root_path)

    logger.info("Running step %s from %s", step_id, step_path)
    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        env_manager="local",
        parameters={key: str(value) for key, value in parameters.items()},
    )


def _load_config(argv: list[str], root_path: Path) -> DictConfig:
    base_config = OmegaConf.load(root_path / "conf" / "config.yaml")
    normalized_argv: list[str] = []
    for argument in argv:
        if not argument.strip():
            continue
        if "=" in argument and " " in argument:
            normalized_argv.extend(shlex.split(argument))
        else:
            normalized_argv.append(argument)
    cli_config = OmegaConf.from_dotlist(normalized_argv)
    return OmegaConf.merge(base_config, cli_config)


def main(argv: list[str] | None = None) -> None:
    """Execute the configured pipeline steps sequentially."""
    _load_environment()
    root_path = get_project_root()
    config = _load_config(argv or sys.argv[1:], root_path=root_path)
    OmegaConf.resolve(config)

    configure_mlflow(
        root_path=root_path,
        tracking_uri=config["mlflow"]["tracking_uri"],
        registry_uri=config["mlflow"]["registry_uri"],
        experiment_name=config["mlflow"]["experiment_name"],
    )
    configure_wandb_env(project=config["wandb"]["project"], mode=config["wandb"]["mode"])

    logger.info("Project root: %s", root_path)
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    for step_id in _steps_to_execute(config):
        if step_id not in STEP_DEFINITIONS:
            raise KeyError(f"Unknown step configured: {step_id}")
        _run_step(step_id=step_id, config=config, root_path=root_path)


if __name__ == "__main__":
    main()
