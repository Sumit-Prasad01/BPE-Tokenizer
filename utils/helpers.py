"""Helper utilities for YAML/JSON I/O, byte formatting, timing, and hashing."""

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator
import yaml

from utils.custom_exception import ConfigValidationError
from utils.logger import logger


def ensure_dir(path: str | Path) -> Path:
    """Ensures that a directory exists, creating parent directories if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Loads and parses a YAML file safely.
    
    Args:
        file_path: Path to the YAML file.
        
    Returns:
        Parsed dictionary.
        
    Raises:
        ConfigValidationError: If file not found or invalid YAML syntax.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ConfigValidationError(f"Configuration file not found: {path.resolve()}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
            return content
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Error parsing YAML file {path}: {exc}") from exc


def dump_yaml(data: dict[str, Any], file_path: str | Path) -> None:
    """Saves dictionary data to a formatted YAML file."""
    path = Path(file_path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_json(file_path: str | Path) -> Any:
    """Loads data from a JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str | Path, indent: int = 2) -> None:
    """Saves data to a JSON file with indentation."""
    path = Path(file_path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def format_bytes(byte_count: int | float) -> str:
    """Converts a raw byte count into a human-readable string (B, KB, MB, GB)."""
    bytes_val = float(byte_count)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0 or unit == "TB":
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{byte_count} B"


def format_number(num: int | float) -> str:
    """Formats a number with thousand separators (e.g. 1,234,567)."""
    if isinstance(num, int):
        return f"{num:,}"
    return f"{num:,.4f}"


def compute_file_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Calculates the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path}")
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


@contextmanager
def Timer(task_name: str = "Task", log_level: str = "INFO") -> Generator[dict[str, float], None, None]:
    """Context manager that measures wall time and logs execution duration.
    
    Yields:
        Dictionary {'elapsed': seconds} populated upon exit.
    """
    start_time = time.perf_counter()
    logger.info(f"⏳ Starting: {task_name}")
    metrics = {"elapsed": 0.0}
    try:
        yield metrics
    finally:
        elapsed = time.perf_counter() - start_time
        metrics["elapsed"] = elapsed
        msg = f"✅ Completed: {task_name} in {elapsed:.2f}s"
        if log_level.upper() == "DEBUG":
            logger.debug(msg)
        else:
            logger.info(msg)
