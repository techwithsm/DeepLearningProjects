"""Feature transformation (Task 1) and feature engineering (Task 3).

``FeaturePipeline`` learns everything it needs (category levels, correlated features to drop,
imputation medians, scaling) from the training split only, and replays exactly the same steps
on validation, test and new loans.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def learn_categories(df: pd.DataFrame, columns) -> dict[str, list[str]]:
    return {col: sorted(df[col].astype(str).unique()) for col in columns}


def one_hot_encode(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """One-hot encode nominal columns, dropping the first (reference) level of each.

    Levels not present in ``categories`` end up as all zeros, i.e. treated like the reference level.
    """
    out = df.copy()
    for col, levels in categories.items():
        values = out.pop(col).astype(str)
        unseen = set(values.unique()) - set(levels)
        if unseen:
            logger.warning("Column '%s' has unseen levels %s; encoding them as the reference level '%s'",
                           col, sorted(unseen), levels[0])
        for level in levels[1:]:
            out[f"{col}_{level}"] = (values == level).astype(int)
    return out


def find_correlated_features(
    df: pd.DataFrame, target: str, threshold: float = 0.7
) -> tuple[list[tuple[str, str, float]], list[str]]:
    """Find feature pairs with |r| > threshold and choose one feature of each pair to drop.

    From each pair the feature less correlated with the target is dropped.
    Returns ``(pairs, to_drop)``.
    """
    corr = df.corr(numeric_only=True)
    target_corr = corr[target].abs()
    features = corr.drop(index=target, columns=target)
    upper = features.where(np.triu(np.ones(features.shape, dtype=bool), k=1)).stack()
    upper = upper.reindex(upper.abs().sort_values(ascending=False).index)

    pairs, to_drop = [], []
    for (a, b), r in upper.items():
        if abs(r) <= threshold:
            break
        pairs.append((a, b, float(r)))
        if a in to_drop or b in to_drop:
            continue
        weaker, stronger = (a, b) if target_corr[a] < target_corr[b] else (b, a)
        to_drop.append(weaker)
        logger.info("%s vs %s: r=%.3f -> drop '%s' (|r with target| %.3f < %.3f)",
                    a, b, r, weaker, target_corr[weaker], target_corr[stronger])

    if not pairs:
        logger.info("No feature pairs above |r| = %.2f", threshold)
    return pairs, to_drop


class FeaturePipeline:
    def __init__(self, target: str, categorical=("purpose",), log_transform=("revol.bal",), corr_threshold=0.7):
        self.target = target
        self.categorical = tuple(categorical)
        self.log_transform = tuple(log_transform)
        self.corr_threshold = corr_threshold

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        self.categories_ = learn_categories(df, self.categorical)
        encoded = one_hot_encode(df, self.categories_)
        self.correlated_pairs_, self.dropped_features_ = find_correlated_features(
            encoded, self.target, self.corr_threshold)

        X = self._engineer(encoded.drop(columns=self.target))
        self.feature_names_ = list(X.columns)
        self.medians_ = X.median()
        self.scaler_ = StandardScaler().fit(X.fillna(self.medians_))

        logger.info("Feature pipeline fitted on %d rows: %d features, dropped %s",
                    len(df), len(self.feature_names_), self.dropped_features_ or "none")
        logger.debug("Feature names: %s", self.feature_names_)
        return self

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineered (unscaled) features, in training column order."""
        self._check_fitted()
        encoded = one_hot_encode(df.drop(columns=self.target, errors="ignore"), self.categories_)
        X = self._engineer(encoded)
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing features: {missing}")
        X = X[self.feature_names_]
        n_missing = int(X.isna().sum().sum())
        if n_missing:
            logger.warning("Imputing %d missing values with training medians", n_missing)
            X = X.fillna(self.medians_)
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self.transform_frame(df)
        return self.scaler_.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=[c for c in self.dropped_features_ if c in X.columns])
        for col in self.log_transform:
            if col in X.columns:
                if (X[col] < 0).any():
                    logger.warning("Negative values in '%s' clipped to 0 before log transform", col)
                X[f"log.{col}"] = np.log1p(X.pop(col).clip(lower=0))
        return X

    def _check_fitted(self) -> None:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("FeaturePipeline is not fitted; call fit() first")
