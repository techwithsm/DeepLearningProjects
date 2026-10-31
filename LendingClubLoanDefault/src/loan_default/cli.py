"""Command-line interface.

    python -m loan_default train   [--epochs N] [--data PATH] [--no-plots]
    python -m loan_default eda     [--data PATH]
    python -m loan_default predict --input loans.csv [--output scored.csv]

Global options (before the sub-command): --log-level, --log-dir.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are only written to disk

from .config import Config  # noqa: E402
from .logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("loan_default.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loan_default", description="Lending Club loan default prediction")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="console log level (the log file always records DEBUG)")
    parser.add_argument("--log-dir", type=Path, default=Config().log_dir, help="directory for log files")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run the full training pipeline")
    train.add_argument("--data", type=Path, help="path to loan_data.csv")
    train.add_argument("--epochs", type=int, help="maximum training epochs")
    train.add_argument("--no-plots", action="store_true", help="skip saving evaluation figures")
    train.add_argument("--eda", action="store_true", help="also save EDA figures")

    eda = sub.add_parser("eda", help="save exploratory data analysis figures")
    eda.add_argument("--data", type=Path, help="path to loan_data.csv")

    predict = sub.add_parser("predict", help="score loans with a trained model")
    predict.add_argument("--input", type=Path, required=True, help="CSV with the 13 feature columns")
    predict.add_argument("--output", type=Path, help="where to write predictions (default: print)")
    predict.add_argument("--model-dir", type=Path, help="trained model directory (default: models/)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_file = setup_logging(args.log_level, args.log_dir, run_name=args.command)
    logger.info("Command: %s (log file: %s)", args.command, log_file)

    cfg = Config()
    if getattr(args, "data", None):
        cfg.data_path = args.data

    try:
        if args.command == "train":
            from .pipeline import run_eda, run_training
            if args.epochs:
                cfg.model.epochs = args.epochs
            if args.eda:
                run_eda(cfg)
            run_training(cfg, make_plots=not args.no_plots)
        elif args.command == "eda":
            from .pipeline import run_eda
            run_eda(cfg)
        elif args.command == "predict":
            _predict(args, cfg)
    except Exception:
        logger.exception("Command '%s' failed", args.command)
        return 1
    logger.info("Done")
    return 0


def _predict(args: argparse.Namespace, cfg: Config) -> None:
    from .data import load_data, validate_data
    from .predictor import LoanDefaultPredictor

    predictor = LoanDefaultPredictor.load(args.model_dir or cfg.artifacts_dir)
    df = load_data(args.input)
    validate_data(df)
    scored = df.join(predictor.predict(df))
    if args.output:
        scored.to_csv(args.output, index=False)
        logger.info("Wrote predictions to %s", args.output)
    else:
        print(scored.to_string())


if __name__ == "__main__":
    sys.exit(main())
