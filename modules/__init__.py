"""
Modules package for the Hallucination-Aware Adaptive LLM System.

Provides shared utilities (load_config) and all core modules:
  - QueryAnalyzer
  - HallucinationPredictor
  - StrategySelector
  - GenerationModule
  - RAGModule
  - VerificationModule
"""

from pathlib import Path
from typing import Dict, Any

import yaml


def _find_project_root() -> Path:
    """Walk up from this file to find the hallucination_aware project root."""
    current = Path(__file__).resolve().parent.parent
    return current


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        config_path: Optional explicit path to config.yaml.
                     If None, resolves to <project_root>/config/config.yaml.

    Returns:
        A dictionary with the full configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is malformed.
    """
    if config_path is None:
        project_root = _find_project_root()
        config_path = str(project_root / "config" / "config.yaml")

    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path_obj, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    return config
