"""
Configuration files for VisProbe.

This package contains:
- presets/: YAML preset configuration files
- preset_schema.json: JSON Schema for preset validation

For the presets API, import from visprobe.presets:

    from visprobe.presets import load_preset, save_preset
"""

from pathlib import Path

# Config directory path
CONFIG_DIR = Path(__file__).parent

# Presets directory path
PRESETS_DIR = CONFIG_DIR / "presets"

# Schema file path
SCHEMA_PATH = CONFIG_DIR / "preset_schema.json"

__all__ = [
    "CONFIG_DIR",
    "PRESETS_DIR",
    "SCHEMA_PATH",
]
