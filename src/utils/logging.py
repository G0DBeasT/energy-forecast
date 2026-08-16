"""
Centralized Logging Utility.

Configures structured logging for all pipeline stages, model training,
and forecasting execution.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a structured console logger.

    Args:
        name: Module name (__name__).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
