import json
import os
from pathlib import Path
from typing import Any, Dict
import yaml

def find_project_root(marker_files: tuple=('main.py', 'requirements.txt', 'README.md')) -> Path:
    """Find project root by looking for marker files."""
    current = Path(__file__).resolve()
    
    # Walk up the directory tree
    for parent in current.parents:
        if any((parent / marker).exists() for marker in marker_files):
            return parent

    # Fallback: assume current file is in src/semantic_video_compressor/
    return current.parent.parent

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def load_config(config_path: str | os.PathLike[str] = "config.yaml") -> Dict[str, Any]:
    """
    Load a config file (YAML preferred, JSON fallback).

    If *config_path* is relative, it’s considered relative to the detected project root.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = find_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yaml(path)
    elif path.suffix.lower() == ".json":
        return load_json(path)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

def get_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Convenience accessor that never raises KeyError."""
    return config.get(section, {})