"""Candidate model and sweep search-space factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.stats import randint, reciprocal
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cap3_classification.modeling.torch_estimator import TorchMLPClassifier


@dataclass(frozen=True)
class SearchSpace:
    wandb: dict[str, Any]
    offline: dict[str, Any]


class ModelFactory:
    """Create competition models and their tuning spaces."""

    @staticmethod
    def create_candidate(model_name: str, random_state: int, params: dict[str, Any] | None = None):
        params = params or {}
        if model_name == "dummy_most_frequent":
            return DummyClassifier(strategy="most_frequent")
        if model_name == "sgd_classifier":
            defaults = {
                "loss": "log_loss",
                "alpha": 1e-4,
                "penalty": "l2",
                "max_iter": 1500,
                "tol": 1e-3,
            }
            defaults.update(params)
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SGDClassifier(random_state=random_state, **defaults)),
                ]
            )
        if model_name == "linear_svc":
            defaults = {"C": 1.0}
            defaults.update(params)
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LinearSVC(random_state=random_state, **defaults)),
                ]
            )
        if model_name == "knn":
            defaults = {"n_neighbors": 5, "weights": "distance", "p": 2}
            defaults.update(params)
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(**defaults)),
                ]
            )
        if model_name == "random_forest":
            defaults = {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "n_jobs": -1,
            }
            defaults.update(params)
            return RandomForestClassifier(random_state=random_state, **defaults)
        if model_name == "torch_mlp":
            defaults = {
                "hidden_layer_sizes": (256, 128),
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 64,
                "epochs": 12,
                "weight_decay": 1e-5,
            }
            defaults.update(params)
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        TorchMLPClassifier(random_state=random_state, **defaults),
                    ),
                ]
            )
        raise ValueError(f"Unsupported model candidate: {model_name}")

    @staticmethod
    def default_search_space(model_name: str, metric_name: str, goal: str) -> SearchSpace:
        metric = {"name": metric_name, "goal": goal}
        spaces = {
            "sgd_classifier": SearchSpace(
                wandb={
                    "method": "bayes",
                    "metric": metric,
                    "parameters": {
                        "loss": {"values": ["hinge", "log_loss", "modified_huber"]},
                        "alpha": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
                        "penalty": {"values": ["l2", "l1", "elasticnet"]},
                        "max_iter": {"values": [1000, 1500, 2000]},
                    },
                },
                offline={
                    "loss": ["hinge", "log_loss", "modified_huber"],
                    "alpha": reciprocal(1e-5, 1e-2),
                    "penalty": ["l2", "l1", "elasticnet"],
                    "max_iter": [1000, 1500, 2000],
                },
            ),
            "linear_svc": SearchSpace(
                wandb={
                    "method": "bayes",
                    "metric": metric,
                    "parameters": {
                        "C": {"distribution": "log_uniform_values", "min": 1e-3, "max": 10.0},
                    },
                },
                offline={"C": reciprocal(1e-3, 10.0)},
            ),
            "knn": SearchSpace(
                wandb={
                    "method": "bayes",
                    "metric": metric,
                    "parameters": {
                        "n_neighbors": {"distribution": "int_uniform", "min": 3, "max": 15},
                        "weights": {"values": ["uniform", "distance"]},
                        "p": {"values": [1, 2]},
                    },
                },
                offline={
                    "n_neighbors": randint(3, 16),
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
            ),
            "random_forest": SearchSpace(
                wandb={
                    "method": "bayes",
                    "metric": metric,
                    "parameters": {
                        "n_estimators": {"distribution": "int_uniform", "min": 100, "max": 400},
                        "max_depth": {"distribution": "int_uniform", "min": 4, "max": 20},
                        "min_samples_split": {"distribution": "int_uniform", "min": 2, "max": 12},
                        "min_samples_leaf": {"distribution": "int_uniform", "min": 1, "max": 6},
                        "max_features": {"values": ["sqrt", "log2", 0.5]},
                    },
                },
                offline={
                    "n_estimators": randint(100, 401),
                    "max_depth": randint(4, 21),
                    "min_samples_split": randint(2, 13),
                    "min_samples_leaf": randint(1, 7),
                    "max_features": ["sqrt", "log2", 0.5],
                },
            ),
            "torch_mlp": SearchSpace(
                wandb={
                    "method": "bayes",
                    "metric": metric,
                    "parameters": {
                        "hidden_layer_sizes": {
                            "values": ["256,128", "256,128,64", "512,256"],
                        },
                        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.4},
                        "learning_rate": {
                            "distribution": "log_uniform_values",
                            "min": 1e-4,
                            "max": 1e-2,
                        },
                        "batch_size": {"values": [32, 64, 128]},
                        "epochs": {"values": [8, 12, 16]},
                        "weight_decay": {
                            "distribution": "log_uniform_values",
                            "min": 1e-6,
                            "max": 1e-3,
                        },
                    },
                },
                offline={
                    "hidden_layer_sizes": ["256,128", "256,128,64", "512,256"],
                    "dropout": [0.0, 0.1, 0.2, 0.3],
                    "learning_rate": reciprocal(1e-4, 1e-2),
                    "batch_size": [32, 64, 128],
                    "epochs": [8, 12, 16],
                    "weight_decay": reciprocal(1e-6, 1e-3),
                },
            ),
        }
        if model_name not in spaces:
            raise ValueError(f"No sweep space defined for {model_name}")
        return spaces[model_name]

    @staticmethod
    def sanitize_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Normalize serialized sweep parameters into model-ready values."""
        normalized = dict(params)
        if model_name == "torch_mlp" and isinstance(normalized.get("hidden_layer_sizes"), str):
            normalized["hidden_layer_sizes"] = tuple(
                int(part.strip()) for part in normalized["hidden_layer_sizes"].split(",")
            )
        return normalized
