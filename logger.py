"""
Logging configuration for the Leap Gesture Recognition project.
Provides structured logging with file and console handlers.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "gesture_recognition",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (None for no file logging)
        level: Logging level
        console: Whether to add console handler

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    return logger


class ProgressLogger:
    """Context manager for logging progress of long operations."""

    def __init__(self, logger: logging.Logger, operation: str, total: Optional[int] = None):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.start_time: Optional[datetime] = None

    def __enter__(self):
        self.start_time = datetime.now()
        msg = f"Starting {self.operation}"
        if self.total:
            msg += f" ({self.total} items)"
        self.logger.info(msg)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type:
                self.logger.error(f"{self.operation} failed after {duration:.2f}s: {exc_val}")
            else:
                self.logger.info(f"Completed {self.operation} in {duration:.2f}s")

    def log_progress(self, current: int, message: str = ""):
        """Log progress update."""
        if self.total and current % max(1, self.total // 10) == 0:
            pct = (current / self.total) * 100
            msg = f"{self.operation} progress: {current}/{self.total} ({pct:.1f}%)"
            if message:
                msg += f" - {message}"
            self.logger.info(msg)


# Default logger instance
logger = setup_logger()
