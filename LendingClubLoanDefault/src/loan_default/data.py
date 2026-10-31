"""Data loading, validation and train/validation/test splitting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "credit.policy", "purpose", "int.rate", "installment", "log.annual.inc", "dti", "fico",
    "days.with.cr.line", "revol.bal", "revol.util", "inq.last.6mths", "delinq.2yrs", "pub.rec",
]


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def items(self):
        return [("train", self.train), ("val", self.val), ("test", self.test)]


def load_data(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def validate_data(df: pd.DataFrame, target: str | None = None) -> None:
    """Check the schema and log data-quality issues. Pass ``target`` to also validate the label."""
    required = FEATURE_COLUMNS + ([target] if target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_missing = df[required].isna().sum()
    if n_missing.any():
        logger.warning("Missing values found: %s", n_missing[n_missing > 0].to_dict())
    else:
        logger.info("No missing values")

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        logger.warning("%d duplicate rows found", n_dupes)

    if target:
        labels = set(df[target].unique())
        if not labels <= {0, 1}:
            raise ValueError(f"Target '{target}' must be binary 0/1, found {sorted(labels)}")
        rate = df[target].mean()
        logger.info("Target '%s': %.1f%% positive (imbalance %.1f : 1)", target, 100 * rate, (1 - rate) / rate)


def split_data(
    df: pd.DataFrame, target: str, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> DataSplits:
    """Stratified split so every partition keeps the same default rate."""
    holdout = val_size + test_size
    train, temp = train_test_split(df, test_size=holdout, stratify=df[target], random_state=seed)
    val, test = train_test_split(temp, test_size=test_size / holdout, stratify=temp[target], random_state=seed)
    splits = DataSplits(train, val, test)
    for name, part in splits.items():
        logger.info("%-5s: %5d rows, positive rate %.3f", name, len(part), part[target].mean())
    return splits
