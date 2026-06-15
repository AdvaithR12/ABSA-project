"""
Centralized logging setup for the ABSA system.
Provides consistent logging across all modules.
"""

import logging
import os


def setup_logger(name, log_file=None, level=logging.INFO):
    """Create and configure a logger instance.
    
    Args:
        name: Logger name (typically __name__ of the calling module).
        log_file: Optional file path to write logs to.
        level: Logging level (default: INFO).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default application logger
app_logger = setup_logger('absa', log_file='logs/absa_inference.log')
