import json
import os
from pathlib import Path

def find_project_root(marker_files=('config.json', 'requirements.txt', 'README.md')):
    """Find project root by looking for marker files."""
    current = Path(__file__).resolve()
    
    # Walk up the directory tree
    for parent in current.parents:
        if any((parent / marker).exists() for marker in marker_files):
            return parent

    # Fallback: assume current file is in src/semantic_video_compressor/
    return current.parent.parent

def load_config(config_path="config.json"):
    if not os.path.isabs(config_path):
        project_root = find_project_root()
        config_path = project_root / config_path
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_config_section(config, section):
    return config.get(section, {})