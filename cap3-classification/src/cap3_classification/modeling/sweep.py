"""Sweep runners for managed W&B Bayes search and local fallback."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import wandb
from sklearn.model_selection import ParameterSampler

from cap3_classification.modeling.evaluation import fit_and_evaluate, split_xy
from cap3_classification.modeling.factory import ModelFactory
from cap3_classification.schemas import SweepArtifact
from cap3_classification.tracking.wandb_utils import can_use_managed_sweeps, init_run


@dataclass
class SweepContext:
    train_df: Any
    validation_df: Any
    target_column: str
    model_name: str
    metric_name: str
    goal: str
    wandb_project: str
    wandb_mode: str
    random_state: int


def _is_better(current_best: float | None, candidate: float, goal: str) -> bool:
    if current_best is None:
        return True
    if goal == "maximize":
        return candidate > current_best
    return candidate < current_best


class OfflineSweepRunner:
    """Local shadow runner that keeps the same search space when W&B cannot orchestrate."""

    def __init__(self, context: SweepContext):
        self.context = context

    def run(self, n_runs: int) -> SweepArtifact:
        search_space = ModelFactory.default_search_space(
            model_name=self.context.model_name,
            metric_name=self.context.metric_name,
            goal=self.context.goal,
        ).offline
        x_train, y_train = split_xy(self.context.train_df, self.context.target_column)
        x_validation, y_validation = split_xy(
            self.context.validation_df, self.context.target_column
        )

        best_params: dict[str, Any] | None = None
        best_score: float | None = None

        for params in ParameterSampler(
            search_space,
            n_iter=n_runs,
            random_state=self.context.random_state,
        ):
            clean_params = ModelFactory.sanitize_params(self.context.model_name, params)
            run = init_run(
                project=self.context.wandb_project,
                job_type="shadow_sweep",
                mode=self.context.wandb_mode,
                config=clean_params,
            )
            model = ModelFactory.create_candidate(
                model_name=self.context.model_name,
                random_state=self.context.random_state,
                params=clean_params,
            )
            metrics, _ = fit_and_evaluate(
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_eval=x_validation,
                y_eval=y_validation,
            )
            if run is not None:
                wandb.log({**clean_params, **metrics})
                run.finish()
            score = metrics[self.context.metric_name]
            if _is_better(best_score, score, self.context.goal):
                best_score = score
                best_params = clean_params

        if best_params is None or best_score is None:
            raise RuntimeError("Offline sweep did not evaluate any parameter set")

        return SweepArtifact(
            model_name=self.context.model_name,
            source="offline_shadow",
            metric_name=self.context.metric_name,
            goal=self.context.goal,
            best_score=float(best_score),
            best_params=best_params,
            sweep_id=f"offline-{uuid.uuid4()}",
        )


class ManagedWandbSweepRunner:
    """Real W&B sweep runner using `method: bayes`."""

    def __init__(self, context: SweepContext):
        self.context = context
        self.best_params: dict[str, Any] | None = None
        self.best_score: float | None = None
        self.sweep_id: str | None = None
        self.x_train, self.y_train = split_xy(self.context.train_df, self.context.target_column)
        self.x_validation, self.y_validation = split_xy(
            self.context.validation_df,
            self.context.target_column,
        )

    def _train_once(self) -> None:
        run = wandb.init(project=self.context.wandb_project, job_type="bayes_sweep")
        params = ModelFactory.sanitize_params(self.context.model_name, dict(wandb.config))
        model = ModelFactory.create_candidate(
            model_name=self.context.model_name,
            random_state=self.context.random_state,
            params=params,
        )
        metrics, _ = fit_and_evaluate(
            model=model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_eval=self.x_validation,
            y_eval=self.y_validation,
        )
        wandb.log({**params, **metrics})
        score = metrics[self.context.metric_name]
        if _is_better(self.best_score, score, self.context.goal):
            self.best_score = score
            self.best_params = params
        run.finish()

    def run(self, n_runs: int) -> SweepArtifact:
        sweep_config = ModelFactory.default_search_space(
            model_name=self.context.model_name,
            metric_name=self.context.metric_name,
            goal=self.context.goal,
        ).wandb
        self.sweep_id = wandb.sweep(sweep_config, project=self.context.wandb_project)
        wandb.agent(
            self.sweep_id,
            function=self._train_once,
            count=n_runs,
            project=self.context.wandb_project,
        )
        if self.best_params is None or self.best_score is None:
            raise RuntimeError("Managed sweep did not return a best run")
        return SweepArtifact(
            model_name=self.context.model_name,
            source="wandb_bayes",
            metric_name=self.context.metric_name,
            goal=self.context.goal,
            best_score=float(self.best_score),
            best_params=self.best_params,
            sweep_id=self.sweep_id,
        )


def run_sweep(context: SweepContext, online_runs: int, offline_runs: int) -> SweepArtifact:
    """Dispatch to the managed W&B sweep or the local shadow runner."""
    if can_use_managed_sweeps(context.wandb_mode):
        return ManagedWandbSweepRunner(context=context).run(n_runs=online_runs)
    return OfflineSweepRunner(context=context).run(n_runs=offline_runs)
