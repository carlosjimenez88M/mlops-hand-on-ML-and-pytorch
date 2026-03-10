"""Tests for root orchestrator configuration handling."""

from __future__ import annotations

from cap3_classification.orchestrator import _load_config
from cap3_classification.pipeline_registry import get_project_root


def test_load_config_ignores_empty_mlproject_argument():
    config = _load_config([""], root_path=get_project_root())
    assert config["main"]["project_name"] == "cap3-classification"
    assert "" not in config
