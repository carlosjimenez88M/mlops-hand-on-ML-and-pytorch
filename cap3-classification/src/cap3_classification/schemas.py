"""Shared schema models for persisted artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    source_requested: str
    source_used: str
    n_samples: int
    n_features: int
    target_column: str
    class_labels: list[int]


class ValidationReport(BaseModel):
    status: Literal["passed", "failed"]
    row_count: int
    feature_count: int
    class_count: int
    missing_values: int
    duplicate_rows: int
    notes: list[str] = Field(default_factory=list)


class FeatureManifest(BaseModel):
    base_feature_count: int
    derived_feature_names: list[str]
    total_feature_count: int


class ModelResult(BaseModel):
    model_name: str
    metrics: dict[str, float]
    params: dict[str, Any]
    training_time_seconds: float


class CompetitionSummary(BaseModel):
    metric_name: str
    best_model_name: str
    best_score: float
    candidate_models: list[str]
    results: list[ModelResult]


class SweepArtifact(BaseModel):
    model_name: str
    source: Literal["wandb_bayes", "offline_shadow"]
    metric_name: str
    goal: Literal["maximize", "minimize"]
    best_score: float
    best_params: dict[str, Any]
    sweep_id: str | None = None


class RegisteredModelArtifact(BaseModel):
    registered_model_name: str
    alias: str
    model_version: str
    run_id: str
    model_name: str
    target_column: str
    input_dim: int
    class_labels: list[int]
    metrics: dict[str, float]
    local_model_path: str


class DriftFeatureResult(BaseModel):
    feature: str
    psi: float


class DriftReport(BaseModel):
    drift_detected: bool
    max_psi_threshold: float
    drifted_feature_count: int
    top_features_by_psi: list[DriftFeatureResult]
