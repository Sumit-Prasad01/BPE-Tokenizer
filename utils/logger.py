"""Structured logging system with colored console formatting and rotating file handlers."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Color mapping for terminal formatting
LOG_COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",      # Reset
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding ANSI colors for console output."""

    def format(self, record: logging.LogRecord) -> str:
        color = LOG_COLORS.get(record.levelname, LOG_COLORS["RESET"])
        reset = LOG_COLORS["RESET"]
        record.levelname = f"{color}{record.levelname:<8}{reset}"
        return super().format(record)


def setup_logger(
    name: str = "bpe_tokenizer",
    log_file: str | Path | None = "logs/tokenizer.log",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configures and returns a logger instance with console and optional rotating file handlers.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to rotating log file.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        max_bytes: Max file size in bytes before rotation.
        backup_count: Number of rotated log files to retain.
    """
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers if already initialized
    if logger.handlers:
        return logger

    # Ensure stdout/stderr handles UTF-8 emojis on Windows without cp1252 charmap errors
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    console_handler.setFormatter(ColoredFormatter(console_format, datefmt=date_format))
    logger.addHandler(console_handler)

    # Rotating file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format, datefmt=date_format))
        logger.addHandler(file_handler)

    return logger


def add_file_handler(logger: logging.Logger, log_file: str | Path) -> None:
    """Adds an additional file handler (e.g. for a specific experiment run)."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    file_handler.setFormatter(logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)


# Default module-level logger instance
logger = setup_logger()
