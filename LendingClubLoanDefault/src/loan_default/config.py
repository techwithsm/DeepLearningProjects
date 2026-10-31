"""Central configuration: paths, preprocessing settings and model hyperparameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    """Hyperparameters for the Keras network and its training loop.

    Defaults were chosen by comparing a handful of configurations on validation AUC.
    """

    hidden_units: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.3
    l2: float = 1e-4
    learning_rate: float = 3e-4
    batch_size: int = 128
    epochs: int = 300
    early_stopping_patience: int = 25
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    log_every: int = 10  # epochs between INFO-level progress lines (every epoch goes to DEBUG)


@dataclass
class Config:
    data_path: Path = PROJECT_ROOT / "DataSet" / "loan_data.csv"
    artifacts_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    log_dir: Path = PROJECT_ROOT / "logs"

    target: str = "not.fully.paid"
    categorical: tuple[str, ...] = ("purpose",)
    log_transform: tuple[str, ...] = ("revol.bal",)
    corr_threshold: float = 0.7

    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    # Decision-threshold search grid (start, stop, step), evaluated on the validation set.
    threshold_grid: tuple[float, float, float] = (0.20, 0.80, 0.01)

    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
