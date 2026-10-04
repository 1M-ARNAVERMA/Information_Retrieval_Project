"""Centralized logger configuration for FEVER-X."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_loggers = {}


def get_logger(name: str = "fever_x", log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Retrieves or creates a named logger with formatted console and optional file handlers.

    Args:
        name: Subsystem or module name.
        log_file: Optional path to append logs into.
        level: Minimum log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Formatter
        format_str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        # Console Handler with color if available
        try:
            import colorlog

            color_format = (
                "%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
            )
            console_formatter = colorlog.ColoredFormatter(
                color_format,
                datefmt=date_format,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        except ImportError:
            console_formatter = logging.Formatter(format_str, datefmt=date_format)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler (if requested or default logs dir configured)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_formatter = logging.Formatter(format_str, datefmt=date_format)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
