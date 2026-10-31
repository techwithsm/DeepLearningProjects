"""Exploratory data analysis plots (Task 2). Each function returns a matplotlib Figure."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .features import learn_categories, one_hot_encode

logger = logging.getLogger(__name__)

TARGET = "not.fully.paid"
NUMERIC_COLUMNS = ["int.rate", "installment", "log.annual.inc", "dti", "fico",
                   "days.with.cr.line", "revol.bal", "revol.util"]
COUNT_COLUMNS = ["inq.last.6mths", "delinq.2yrs", "pub.rec"]

sns.set_theme(style="whitegrid", palette="deep")


def default_rate_by(df: pd.DataFrame, column: str, target: str = TARGET) -> pd.Series:
    return df.groupby(column)[target].mean().sort_values(ascending=False)


def plot_target_distribution(df: pd.DataFrame, target: str = TARGET) -> plt.Figure:
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target, ax=ax)
    for i, v in enumerate(counts):
        ax.text(i, v + 60, f"{v} ({v / len(df):.1%})", ha="center")
    ax.set_xticks([0, 1], ["Fully paid (0)", "Not fully paid / default (1)"])
    ax.set_title("Target distribution")
    ax.set_xlabel("")
    return fig


def plot_purpose(df: pd.DataFrame, target: str = TARGET) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=df, y="purpose", hue=target, order=df["purpose"].value_counts().index, ax=axes[0])
    axes[0].set_title("Loan count by purpose")
    rate = default_rate_by(df, "purpose", target)
    sns.barplot(x=rate.values, y=rate.index, ax=axes[1], color="C3")
    axes[1].axvline(df[target].mean(), ls="--", c="k", label="overall rate")
    axes[1].set(title="Default rate by purpose", xlabel="default rate", ylabel="")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_credit_policy(df: pd.DataFrame, target: str = TARGET) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="credit.policy", hue=target, ax=axes[0])
    axes[0].set_title("Credit policy vs. outcome")
    sns.barplot(data=df, x="credit.policy", y=target, ax=axes[1], color="C3")
    axes[1].set(title="Default rate by credit policy", ylabel="default rate")
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, columns=NUMERIC_COLUMNS, target: str = TARGET) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, col in zip(axes.ravel(), columns):
        sns.histplot(data=df, x=col, hue=target, bins=40, stat="density",
                     common_norm=False, element="step", ax=ax)
        ax.set_title(col)
    fig.tight_layout()
    return fig


def plot_numeric_boxplots(df: pd.DataFrame, columns=NUMERIC_COLUMNS, target: str = TARGET) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, col in zip(axes.ravel(), columns):
        sns.boxplot(data=df, x=target, y=col, ax=ax, showfliers=False)
        ax.set_title(col)
    fig.tight_layout()
    return fig


def plot_count_features(df: pd.DataFrame, columns=COUNT_COLUMNS, target: str = TARGET) -> plt.Figure:
    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4))
    for ax, col in zip(np.atleast_1d(axes), columns):
        clipped = df[col].clip(upper=int(df[col].quantile(0.99)))  # int keeps count categories whole
        sns.barplot(x=clipped, y=df[target], ax=ax, color="C3", errorbar=None)
        ax.set(title=f"Default rate vs. {col} (clipped at 99th pct)", ylabel="default rate")
    fig.tight_layout()
    return fig


def plot_rate_vs_fico(df: pd.DataFrame, target: str = TARGET) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="fico", y="int.rate", hue=target, alpha=0.4, s=15, ax=ax)
    ax.set_title("Interest rate vs. FICO score")
    return fig


def plot_correlation_heatmap(encoded: pd.DataFrame) -> plt.Figure:
    corr = encoded.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation matrix (encoded features)")
    return fig


def plot_target_correlation(encoded: pd.DataFrame, target: str = TARGET) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    encoded.corr(numeric_only=True)[target].drop(target).sort_values().plot.barh(ax=ax, color="C0")
    ax.set_title(f"Correlation of each feature with {target}")
    return fig


def plot_log_transform(df: pd.DataFrame, column: str = "revol.bal") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[column], bins=50, ax=axes[0])
    axes[0].set_title(f"{column} (raw)")
    sns.histplot(np.log1p(df[column].clip(lower=0)), bins=50, ax=axes[1])
    axes[1].set_title(f"log1p({column})")
    fig.tight_layout()
    return fig


def save_eda_figures(df: pd.DataFrame, out_dir: Path | str, target: str = TARGET,
                     categorical=("purpose",)) -> list[Path]:
    """Render every EDA plot to PNG files in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = one_hot_encode(df, learn_categories(df, categorical))

    figures = {
        "01_target_distribution": lambda: plot_target_distribution(df, target),
        "02_purpose": lambda: plot_purpose(df, target),
        "03_credit_policy": lambda: plot_credit_policy(df, target),
        "04_numeric_distributions": lambda: plot_numeric_distributions(df, target=target),
        "05_numeric_boxplots": lambda: plot_numeric_boxplots(df, target=target),
        "06_count_features": lambda: plot_count_features(df, target=target),
        "07_rate_vs_fico": lambda: plot_rate_vs_fico(df, target),
        "08_correlation_heatmap": lambda: plot_correlation_heatmap(encoded),
        "09_target_correlation": lambda: plot_target_correlation(encoded, target),
        "10_log_transform": lambda: plot_log_transform(df),
    }
    paths = []
    for name, make in figures.items():
        fig = make()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved %s", path)
        paths.append(path)
    logger.info("Saved %d EDA figures to %s", len(paths), out_dir)
    return paths
