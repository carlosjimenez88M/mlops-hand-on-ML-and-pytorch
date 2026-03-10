"""W&B initialization helpers."""

from __future__ import annotations

import os
from typing import Any

import wandb


def configure_wandb_env(project: str, mode: str) -> None:
    """Configure process-level W&B environment variables."""
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_MODE"] = mode


def init_run(
    project: str,
    job_type: str,
    mode: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """Create a W&B run in a consistent way."""
    return wandb.init(project=project, job_type=job_type, mode=mode, config=config, tags=tags)


def can_use_managed_sweeps(mode: str) -> bool:
    """Return whether the environment can call the managed W&B sweep service."""
    return mode not in {"disabled", "offline", "dryrun"}
