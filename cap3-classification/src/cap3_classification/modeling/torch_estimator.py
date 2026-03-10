"""Torch-based digit classifier with a scikit-learn compatible interface."""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """A small fully connected neural net that behaves like a sklearn estimator."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 12,
        weight_decay: float = 0.0,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    def _set_seeds(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_network(self, input_dim: int, n_classes: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(previous, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous = hidden_dim
        layers.append(nn.Linear(previous, n_classes))
        return nn.Sequential(*layers)

    def fit(self, x, y):
        self._set_seeds()
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.int64)

        self.classes_, encoded_targets = np.unique(y_array, return_inverse=True)
        self.input_dim_ = int(x_array.shape[1])
        self.device_ = self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu"
        self.model_ = self._build_network(self.input_dim_, len(self.classes_)).to(self.device_)

        dataset = TensorDataset(
            torch.from_numpy(x_array),
            torch.from_numpy(encoded_targets.astype(np.int64)),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.epochs):
            for features, target in loader:
                features = features.to(self.device_)
                target = target.to(self.device_)
                optimizer.zero_grad()
                logits = self.model_(features)
                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def predict_proba(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Estimator must be fitted before calling predict_proba")
        x_array = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(x_array).to(self.device_)
            logits = self.model_(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def predict(self, x) -> np.ndarray:
        probabilities = self.predict_proba(x)
        encoded = np.argmax(probabilities, axis=1)
        return self.classes_[encoded]
