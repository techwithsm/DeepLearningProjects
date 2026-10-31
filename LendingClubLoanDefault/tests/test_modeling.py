import numpy as np
import pytest

from loan_default.config import ModelConfig
from loan_default.evaluation import choose_threshold, compute_metrics
from loan_default.features import FeaturePipeline
from loan_default.modeling import build_model, compute_class_weights, set_seed, train_model
from loan_default.predictor import LoanDefaultPredictor

from .conftest import TARGET


def test_compute_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
    assert m["roc_auc"] == 1.0 and m["f1"] == 1.0 and m["fp"] == 0 and m["fn"] == 0


def test_choose_threshold_returns_value_from_grid():
    y = np.array([0, 0, 0, 1, 1])
    search = choose_threshold(y, np.array([0.1, 0.3, 0.45, 0.55, 0.9]), 0.2, 0.8, 0.05)
    assert search.best in search.thresholds
    assert 0.45 < search.best <= 0.55


def test_class_weights_upweight_minority():
    weights = compute_class_weights(np.array([0] * 8 + [1] * 2))
    assert weights[1] > weights[0]


def test_train_save_load_roundtrip(loans, tmp_path):
    set_seed(0)
    pipe = FeaturePipeline(TARGET)
    X = pipe.fit_transform(loans)
    y = loans[TARGET].to_numpy()
    cfg = ModelConfig(hidden_units=(8,), epochs=2, batch_size=64)
    model = build_model(X.shape[1], cfg)
    history = train_model(model, X, y, X, y, cfg)
    assert len(history["loss"]) == 2

    predictor = LoanDefaultPredictor(model, pipe, threshold=0.4)
    predictor.save(tmp_path)
    loaded = LoanDefaultPredictor.load(tmp_path)

    sample = loans.drop(columns=TARGET).head(10)
    assert loaded.threshold == 0.4
    assert np.allclose(loaded.predict_proba(sample), predictor.predict_proba(sample), atol=1e-6)
    assert list(loaded.predict(sample).columns) == ["default_probability", "predicted_default"]


def test_load_missing_model_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        LoanDefaultPredictor.load(tmp_path)
