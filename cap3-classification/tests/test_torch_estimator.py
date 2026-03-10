"""Tests for the Torch-based estimator."""

from __future__ import annotations

import joblib
from sklearn.datasets import load_digits

from cap3_classification.modeling.torch_estimator import TorchMLPClassifier


def test_torch_estimator_fits_predicts_and_roundtrips(tmp_path):
    dataset = load_digits()
    x_train = dataset.data[:120]
    y_train = dataset.target[:120]
    x_test = dataset.data[120:130]

    model = TorchMLPClassifier(
        hidden_layer_sizes=(32,),
        epochs=2,
        batch_size=32,
        learning_rate=1e-3,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    model_path = tmp_path / "torch_mlp.joblib"
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)

    assert len(predictions) == len(x_test)
    assert probabilities.shape == (len(x_test), len(model.classes_))
    assert len(restored.predict(x_test)) == len(x_test)
