"""End-to-end training pipeline used by the CLI."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from . import evaluation as ev
from .config import Config
from .data import DataSplits, load_data, split_data, validate_data
from .eda import save_eda_figures
from .features import FeaturePipeline
from .logging_utils import log_step
from .modeling import build_model, compute_class_weights, set_seed, train_baseline, train_model
from .predictor import LoanDefaultPredictor

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    predictor: LoanDefaultPredictor
    history: dict
    splits: DataSplits
    threshold_search: ev.ThresholdSearch
    test_metrics: dict[str, dict]
    test_proba: dict[str, np.ndarray]


def run_eda(cfg: Config) -> None:
    with log_step(logger, "Exploratory data analysis"):
        df = load_data(cfg.data_path)
        validate_data(df, cfg.target)
        save_eda_figures(df, cfg.figures_dir / "eda", cfg.target, cfg.categorical)


def run_training(cfg: Config, save: bool = True, make_plots: bool = True) -> TrainingResult:
    logger.debug("Configuration: %s", cfg.to_dict())

    with log_step(logger, "Load and validate data"):
        df = load_data(cfg.data_path)
        validate_data(df, cfg.target)

    with log_step(logger, "Stratified train/validation/test split"):
        splits = split_data(df, cfg.target, cfg.val_size, cfg.test_size, cfg.seed)
        y_train, y_val, y_test = (part[cfg.target].to_numpy() for _, part in splits.items())

    with log_step(logger, "Feature transformation and engineering"):
        features = FeaturePipeline(cfg.target, cfg.categorical, cfg.log_transform, cfg.corr_threshold)
        X_train = features.fit_transform(splits.train)
        X_val, X_test = features.transform(splits.val), features.transform(splits.test)

    with log_step(logger, "Logistic-regression baseline"):
        baseline = train_baseline(X_train, y_train, cfg.seed)

    with log_step(logger, "Train Keras neural network"):
        set_seed(cfg.seed)
        model = build_model(X_train.shape[1], cfg.model)
        history = train_model(model, X_train, y_train, X_val, y_val, cfg.model, compute_class_weights(y_train))

    with log_step(logger, "Choose decision threshold on validation set"):
        search = ev.choose_threshold(y_val, model.predict(X_val, verbose=0).ravel(), *cfg.threshold_grid)

    with log_step(logger, "Evaluate on held-out test set"):
        test_proba = {
            "Logistic Regression": baseline.predict_proba(X_test)[:, 1],
            "Neural Network": model.predict(X_test, verbose=0).ravel(),
        }
        test_metrics = {
            "Logistic Regression": ev.compute_metrics(y_test, test_proba["Logistic Regression"], 0.5),
            "Neural Network": ev.compute_metrics(y_test, test_proba["Neural Network"], search.best),
        }
        for name, metrics in test_metrics.items():
            ev.log_metrics(name, metrics)
        logger.info("Test-set comparison:\n%s", ev.comparison_table(test_metrics).to_string())

    predictor = LoanDefaultPredictor(model, features, search.best)
    result = TrainingResult(predictor, history, splits, search, test_metrics, test_proba)

    if save:
        with log_step(logger, "Save artefacts"):
            predictor.save(cfg.artifacts_dir, {"test_metrics": test_metrics["Neural Network"]})
            cfg.reports_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = cfg.reports_dir / "metrics.json"
            metrics_path.write_text(json.dumps(test_metrics, indent=2, default=float))
            logger.info("Wrote %s", metrics_path)

    if make_plots:
        with log_step(logger, "Save evaluation figures"):
            _save_evaluation_figures(result, y_test, cfg)

    return result


def _save_evaluation_figures(result: TrainingResult, y_test: np.ndarray, cfg: Config) -> None:
    out_dir = cfg.figures_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "training_history": ev.plot_training_history(result.history),
        "threshold_search": ev.plot_threshold_search(result.threshold_search),
        "roc_pr_curves": ev.plot_roc_pr(y_test, result.test_proba),
        "confusion_matrix": ev.plot_confusion_matrix(y_test, result.test_proba["Neural Network"],
                                                     result.threshold_search.best, "Neural network"),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved %d evaluation figures to %s", len(figures), out_dir)
