"""
Colored logger configuration with traffic light colors
Author: Carlos Daniel Jiménez
Date: 2025-01-13

Colors:
- Green: INFO (everything is fine)
- Yellow: WARNING (caution needed)
- Red: ERROR/CRITICAL (something went wrong)
"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to log messages based on level."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green (traffic light: go)
        'WARNING': '\033[33m',    # Yellow (traffic light: caution)
        'ERROR': '\033[31m',      # Red (traffic light: stop)
        'CRITICAL': '\033[1;31m'  # Bold Red (traffic light: danger)
    }

    RESET = '\033[0m'

    def format(self, record):
        """Format the log record with color based on level."""
        log_color = self.COLORS.get(record.levelname, self.RESET)

        # Add color to the entire log message
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        record.msg = f"{log_color}{record.msg}{self.RESET}"

        return super().format(record)


def setup_colored_logger(
    name: str = None,
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Sets up a logger with colored output.

    Args:
        name: Logger name (default: root logger)
        level: Logging level (default: INFO)
        format_string: Log message format

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set level
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create and set colored formatter
    formatter = ColoredFormatter(format_string)
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Gets or creates a colored logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return setup_colored_logger(name)
