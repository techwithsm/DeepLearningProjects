"""Model construction and training (Task 4): Keras MLP plus a logistic-regression baseline."""
from __future__ import annotations

import logging
import time

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from .config import ModelConfig

logger = logging.getLogger(__name__)
tf.get_logger().setLevel("ERROR")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and TensorFlow; optionally force deterministic TF ops."""
    keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    logger.debug("Random seed set to %d (deterministic ops=%s)", seed, deterministic)


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    weights = dict(zip(classes.tolist(), compute_class_weight("balanced", classes=classes, y=y).tolist()))
    logger.info("Class weights: %s", {k: round(v, 3) for k, v in weights.items()})
    return weights


def build_model(n_features: int, cfg: ModelConfig) -> keras.Model:
    """Dense -> BatchNorm -> Dropout blocks with L2, sigmoid output for P(default)."""
    reg = keras.regularizers.l2(cfg.l2)
    model = keras.Sequential(name="loan_default_mlp")
    model.add(layers.Input(shape=(n_features,)))
    for units in cfg.hidden_units:
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(cfg.dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"),
                 keras.metrics.AUC(name="pr_auc", curve="PR"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    logger.info("Built model: %d inputs, hidden units %s, %d parameters",
                n_features, list(cfg.hidden_units), model.count_params())
    model.summary(print_fn=lambda line, *args, **kwargs: logger.debug(line))
    return model


class EpochLogger(keras.callbacks.Callback):
    """Routes Keras training progress into the logging system instead of stdout."""

    def __init__(self, every: int = 10):
        super().__init__()
        self.every = max(1, every)

    def on_train_begin(self, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        metrics = " ".join(f"{k}={v:.4g}" for k, v in (logs or {}).items())
        level = logging.INFO if epoch == 0 or (epoch + 1) % self.every == 0 else logging.DEBUG
        logger.log(level, "Epoch %3d | %s", epoch + 1, metrics)

    def on_train_end(self, logs=None):
        logger.info("Training loop took %.1fs", time.perf_counter() - self._start)


def train_model(model: keras.Model, X_train, y_train, X_val, y_val, cfg: ModelConfig,
                class_weight: dict[int, float] | None = None) -> dict[str, list[float]]:
    """Fit with early stopping on validation AUC. Returns the Keras history dict."""
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=cfg.early_stopping_patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=cfg.reduce_lr_factor,
                                          patience=cfg.reduce_lr_patience, min_lr=cfg.min_lr),
        EpochLogger(cfg.log_every),
    ]
    logger.info("Training for up to %d epochs (batch size %d, lr %.1e)", cfg.epochs, cfg.batch_size, cfg.learning_rate)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=cfg.epochs,
                        batch_size=cfg.batch_size, class_weight=class_weight, callbacks=callbacks, verbose=0).history

    best = int(np.argmax(history["val_auc"]))
    logger.info("Stopped after %d epochs; restored best epoch %d (val_auc=%.4f, val_loss=%.4f)",
                len(history["loss"]), best + 1, history["val_auc"][best], history["val_loss"][best])
    return history


def train_baseline(X, y, seed: int = 42) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(X, y)
    logger.info("Trained logistic-regression baseline on %d rows", len(y))
    return model
