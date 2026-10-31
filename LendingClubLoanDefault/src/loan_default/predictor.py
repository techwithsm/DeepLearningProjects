"""Inference wrapper bundling the trained model, feature pipeline and decision threshold."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .features import FeaturePipeline

logger = logging.getLogger(__name__)

MODEL_FILE = "model.keras"
FEATURES_FILE = "features.joblib"
METADATA_FILE = "metadata.json"


class LoanDefaultPredictor:
    def __init__(self, model: keras.Model, features: FeaturePipeline, threshold: float, metadata: dict | None = None):
        self.model = model
        self.features = features
        self.threshold = threshold
        self.metadata = metadata or {}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.features.transform(df), verbose=0).ravel()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(df)
        result = pd.DataFrame({"default_probability": proba.round(4),
                               "predicted_default": (proba >= self.threshold).astype(int)}, index=df.index)
        logger.info("Scored %d loans: %d flagged as likely default (threshold %.2f)",
                    len(df), int(result["predicted_default"].sum()), self.threshold)
        return result

    def save(self, directory: Path | str, extra_metadata: dict | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / MODEL_FILE)
        joblib.dump(self.features, directory / FEATURES_FILE)
        self.metadata = {
            **self.metadata,
            **(extra_metadata or {}),
            "threshold": self.threshold,
            "features": self.features.feature_names_,
            "dropped_correlated": self.features.dropped_features_,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "tensorflow_version": tf.__version__,
        }
        (directory / METADATA_FILE).write_text(json.dumps(self.metadata, indent=2, default=float))
        logger.info("Saved model artefacts to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "LoanDefaultPredictor":
        directory = Path(directory)
        if not (directory / MODEL_FILE).exists():
            raise FileNotFoundError(f"No trained model in {directory}; run training first")
        model = keras.models.load_model(directory / MODEL_FILE)
        features = joblib.load(directory / FEATURES_FILE)
        metadata = json.loads((directory / METADATA_FILE).read_text())
        logger.info("Loaded model from %s (saved %s, threshold %.2f)",
                    directory, metadata.get("saved_at"), metadata["threshold"])
        return cls(model, features, metadata["threshold"], metadata)
