"""Metrics, decision-threshold selection and evaluation plots."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)

logger = logging.getLogger(__name__)


@dataclass
class ThresholdSearch:
    best: float
    thresholds: np.ndarray
    scores: np.ndarray


def choose_threshold(y_true, proba, start: float = 0.2, stop: float = 0.8, step: float = 0.01) -> ThresholdSearch:
    """Pick the cutoff that maximises F1 on the positive class. Use validation data, never test."""
    thresholds = np.round(np.arange(start, stop + step / 2, step), 4)
    scores = np.array([f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in thresholds])
    best = float(thresholds[int(np.argmax(scores))])
    logger.info("Best threshold %.2f (validation F1=%.4f; F1 at 0.50=%.4f)",
                best, scores.max(), f1_score(y_true, (proba >= 0.5).astype(int), zero_division=0))
    return ThresholdSearch(best, thresholds, scores)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def log_metrics(name: str, metrics: dict) -> None:
    logger.info("%s @ %.2f | ROC-AUC %.4f | PR-AUC %.4f | precision %.3f | recall %.3f | F1 %.3f | accuracy %.3f",
                name, metrics["threshold"], metrics["roc_auc"], metrics["pr_auc"], metrics["precision"],
                metrics["recall"], metrics["f1"], metrics["accuracy"])
    logger.debug("%s confusion matrix: TN=%d FP=%d FN=%d TP=%d",
                 name, metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])


def comparison_table(metrics_by_model: dict[str, dict]) -> pd.DataFrame:
    cols = ["threshold", "roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]
    return pd.DataFrame(metrics_by_model).T[cols].astype(float).round(4)


def plot_training_history(history: dict, metrics=("loss", "auc", "recall")) -> plt.Figure:
    best = int(np.argmax(history["val_auc"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        ax.plot(history[m], label="train")
        ax.plot(history[f"val_{m}"], label="validation")
        ax.axvline(best, ls="--", c="grey", lw=1)
        ax.set(title=m, xlabel="epoch")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_threshold_search(search: ThresholdSearch) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(search.thresholds, search.scores)
    ax.axvline(search.best, ls="--", c="C3", label=f"best = {search.best:.2f}")
    ax.set(title="Threshold selection on validation set", xlabel="threshold", ylabel="F1 (default class)")
    ax.legend()
    return fig


def plot_roc_pr(y_true, probas: dict[str, np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, p):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, p)
        axes[1].plot(rec, prec, label=f"{name} (AP = {average_precision_score(y_true, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(np.mean(y_true), c="k", ls="--", lw=1, label="random")
    axes[1].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, proba, threshold: float, title: str = "Confusion matrix") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, (proba >= threshold).astype(int)),
                           display_labels=["Fully paid", "Default"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title} (threshold {threshold:.2f})")
    ax.grid(False)
    return fig
