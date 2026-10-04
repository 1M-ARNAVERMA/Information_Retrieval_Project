"""Configuration loader with validation and environment overrides."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.common.exceptions import ConfigurationError


def _find_project_root() -> Path:
    """Heuristic to locate the repository root containing config/."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config" / "config.yaml").exists():
            return parent
    return Path.cwd()


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Loads and parses a YAML file safely.

    Args:
        file_path: Absolute or relative Path.

    Returns:
        Parsed configuration dictionary.
    """
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception as e:
        raise ConfigurationError(f"Failed to parse YAML file {file_path}: {e}") from e


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads system config.yaml.

    Args:
        config_path: Optional direct path to config.yaml.

    Returns:
        Configuration dictionary.
    """
    if config_path:
        path = Path(config_path)
    else:
        root = _find_project_root()
        path = root / "config" / "config.yaml"

    cfg = load_yaml(path)

    # Optional environment overrides
    if "LOG_LEVEL" in os.environ:
        cfg.setdefault("app", {})["log_level"] = os.environ["LOG_LEVEL"]
    if "DATA_RAW_DIR" in os.environ:
        cfg.setdefault("paths", {})["raw_data_dir"] = os.environ["DATA_RAW_DIR"]
    if "DATA_PROCESSED_DIR" in os.environ:
        cfg.setdefault("paths", {})["processed_data_dir"] = os.environ["DATA_PROCESSED_DIR"]
    if "DATA_INDEXES_DIR" in os.environ:
        cfg.setdefault("paths", {})["indexes_dir"] = os.environ["DATA_INDEXES_DIR"]

    return cfg


def load_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads model_config.yaml.

    Args:
        config_path: Optional direct path to model_config.yaml.

    Returns:
        Model configuration dictionary.
    """
    if config_path:
        path = Path(config_path)
    else:
        root = _find_project_root()
        path = root / "config" / "model_config.yaml"

    cfg = load_yaml(path)

    if "DEVICE" in os.environ:
        cfg.setdefault("runtime", {})["device"] = os.environ["DEVICE"]

    return cfg
