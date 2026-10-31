"""Logging setup shared by the CLI, the notebook and the tests.

Every module logs through ``logging.getLogger(__name__)``; all of them sit under the
``loan_default`` logger, which this module configures with a console handler (user-chosen
level) and a timestamped file handler that always records DEBUG detail.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO, Iterator

PACKAGE_LOGGER = "loan_default"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    log_dir: Path | str | None = None,
    run_name: str = "run",
    stream: IO[str] | None = None,
) -> Path | None:
    """Configure the package logger. Safe to call repeatedly (e.g. re-running a notebook cell).

    Returns the path of the log file, or None when ``log_dir`` is not given.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(stream or sys.stderr)
    console.setLevel(level.upper() if isinstance(level, str) else level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging initialised (console level=%s, file=%s)", logging.getLevelName(console.level), log_path)
    return log_path


@contextmanager
def log_step(logger: logging.Logger, description: str) -> Iterator[None]:
    """Log the start, duration and any failure of a pipeline step."""
    logger.info("Starting: %s", description)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("Failed: %s (after %.2fs)", description, time.perf_counter() - start)
        raise
    logger.info("Finished: %s (%.2fs)", description, time.perf_counter() - start)
