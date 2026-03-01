"""
YAML preset loading logic.

Handles discovering, loading, and saving preset configurations.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# =============================================================================
# Path Configuration
# =============================================================================

# Built-in presets directory (relative to this file's parent)
BUILTIN_PRESETS_DIR = Path(__file__).parent.parent / "config" / "presets"

# User presets directory (can be overridden via environment variable)
USER_PRESETS_DIR = Path(os.environ.get(
    "VISPROBE_PRESETS_DIR",
    Path.home() / ".visprobe" / "presets"
))


# =============================================================================
# Discovery
# =============================================================================

def _discover_presets(directory: Path) -> Dict[str, Path]:
    """
    Discover all preset YAML files in a directory.

    Args:
        directory: Directory to search

    Returns:
        Dict mapping preset names to file paths
    """
    presets = {}
    if directory.exists():
        for file in directory.glob("*.yaml"):
            preset_name = file.stem
            presets[preset_name] = file
        for file in directory.glob("*.yml"):
            preset_name = file.stem
            if preset_name not in presets:
                presets[preset_name] = file
    return presets


def list_available(include_user: bool = True) -> List[str]:
    """
    List all available preset names.

    Args:
        include_user: Whether to include user-defined presets

    Returns:
        Sorted list of preset names
    """
    presets = set(_discover_presets(BUILTIN_PRESETS_DIR).keys())
    if include_user:
        presets.update(_discover_presets(USER_PRESETS_DIR).keys())
    return sorted(presets)


# =============================================================================
# Loading
# =============================================================================

def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load(
    name: str,
    validate: bool = True,
    search_user_dir: bool = True,
) -> Dict[str, Any]:
    """
    Load a preset by name.

    Searches in order:
    1. User presets directory (~/.visprobe/presets/)
    2. Built-in presets directory

    Args:
        name: Preset name (without .yaml extension)
        validate: Whether to validate against JSON Schema
        search_user_dir: Whether to search user directory first

    Returns:
        Preset configuration dictionary

    Raises:
        ValueError: If preset is not found
        jsonschema.ValidationError: If validation fails
    """
    # Search for preset file
    preset_path = None

    if search_user_dir:
        user_presets = _discover_presets(USER_PRESETS_DIR)
        if name in user_presets:
            preset_path = user_presets[name]

    if preset_path is None:
        builtin_presets = _discover_presets(BUILTIN_PRESETS_DIR)
        if name in builtin_presets:
            preset_path = builtin_presets[name]

    if preset_path is None:
        available = list_available()
        raise ValueError(
            f"Preset '{name}' not found. Available presets: {', '.join(available)}"
        )

    # Load preset
    preset = _load_yaml_file(preset_path)

    # Validate if requested
    if validate:
        from .validator import validate_preset
        validate_preset(preset, raise_on_error=True)

    # Warn about legacy presets
    if preset.get("legacy"):
        hint = preset.get("migration_hint", "")
        warnings.warn(
            f"Preset '{name}' is legacy. {hint}",
            DeprecationWarning,
            stacklevel=2,
        )

    return preset


def load_from_file(
    path: str,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Load a preset from a specific file path.

    Args:
        path: Path to the YAML preset file
        validate: Whether to validate against JSON Schema

    Returns:
        Preset configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        jsonschema.ValidationError: If validation fails
    """
    preset_path = Path(path)
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")

    preset = _load_yaml_file(preset_path)

    if validate:
        from .validator import validate_preset
        validate_preset(preset, raise_on_error=True)

    return preset


# =============================================================================
# Listing
# =============================================================================

def list_presets(include_legacy: bool = False) -> List[Tuple[str, str, str]]:
    """
    List all available presets with descriptions and threat models.

    Args:
        include_legacy: If True, include legacy presets

    Returns:
        List of (name, description, threat_model) tuples
    """
    results = []
    for name in list_available():
        try:
            preset = load(name, validate=False)
            if not include_legacy and preset.get("legacy"):
                continue
            results.append((
                name,
                preset.get("description", ""),
                preset.get("threat_model", "unknown"),
            ))
        except Exception:
            continue
    return results


def list_threat_aware() -> List[Tuple[str, str]]:
    """
    List only non-legacy presets.

    Returns:
        List of (name, description) tuples
    """
    threat_aware = ["natural", "adversarial", "realistic_attack", "comprehensive"]
    results = []
    for name in threat_aware:
        try:
            preset = load(name, validate=False)
            results.append((name, preset.get("description", "")))
        except ValueError:
            continue
    return results


# =============================================================================
# Information
# =============================================================================

def get_info(name: str) -> str:
    """
    Get detailed information about a preset.

    Args:
        name: Preset name

    Returns:
        Formatted string with preset details
    """
    preset = load(name, validate=False)

    lines = [
        f"Preset: {preset['name']}",
        f"Description: {preset['description']}",
        "",
        f"Threat Model: {preset.get('threat_model', 'unknown')}",
        f"  {preset.get('threat_model_description', '')}",
        "",
    ]

    if "novelty" in preset:
        lines.extend([
            "Key Insight:",
            f"  {preset['novelty']}",
            "",
        ])

    lines.extend([
        f"Compute Cost: {preset.get('compute_cost', 'unknown')}",
        f"Estimated Time: {preset.get('estimated_time', 'unknown')}",
        f"Search Budget: {preset.get('search_budget', 'N/A')} queries",
        "",
        "Use Cases:",
    ])

    for use_case in preset.get('use_cases', []):
        lines.append(f"  - {use_case}")

    lines.extend([
        "",
        f"Strategies: {len(preset['strategies'])} perturbations",
    ])

    strategy_types = []
    for strat in preset['strategies']:
        if strat['type'] == 'compositional':
            strategy_types.append(f"  - {strat.get('name', 'compositional')} (compositional)")
        else:
            strategy_types.append(f"  - {strat['type']}")

    lines.extend(strategy_types[:10])
    if len(strategy_types) > 10:
        lines.append(f"  ... and {len(strategy_types) - 10} more")

    if "requires" in preset:
        lines.extend([
            "",
            "Dependencies:",
        ])
        for dep in preset["requires"]:
            lines.append(f"  - {dep}")

    return "\n".join(lines)


# =============================================================================
# User Preset Management
# =============================================================================

def create_user_dir() -> Path:
    """
    Create the user presets directory if it doesn't exist.

    Returns:
        Path to the user presets directory
    """
    USER_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return USER_PRESETS_DIR


def save(
    name: str,
    preset: Dict[str, Any],
    overwrite: bool = False,
    validate: bool = True,
) -> Path:
    """
    Save a preset to the user presets directory.

    Args:
        name: Preset name (will be saved as {name}.yaml)
        preset: Preset configuration dictionary
        overwrite: Whether to overwrite existing preset
        validate: Whether to validate before saving

    Returns:
        Path to the saved preset file

    Raises:
        FileExistsError: If preset exists and overwrite is False
        jsonschema.ValidationError: If validation fails
    """
    if validate:
        from .validator import validate_preset
        validate_preset(preset, raise_on_error=True)

    create_user_dir()
    preset_path = USER_PRESETS_DIR / f"{name}.yaml"

    if preset_path.exists() and not overwrite:
        raise FileExistsError(
            f"Preset '{name}' already exists. Use overwrite=True to replace."
        )

    with open(preset_path, "w") as f:
        yaml.dump(preset, f, default_flow_style=False, sort_keys=False)

    return preset_path


def delete(name: str) -> bool:
    """
    Delete a user-defined preset.

    Args:
        name: Preset name to delete

    Returns:
        True if deleted, False if not found

    Raises:
        ValueError: If trying to delete a built-in preset
    """
    user_presets = _discover_presets(USER_PRESETS_DIR)
    if name not in user_presets:
        builtin_presets = _discover_presets(BUILTIN_PRESETS_DIR)
        if name in builtin_presets:
            raise ValueError(f"Cannot delete built-in preset '{name}'")
        return False

    user_presets[name].unlink()
    return True


# =============================================================================
# Strategy Utilities
# =============================================================================

def get_strategies_by_category(preset_name: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get strategies from a preset grouped by threat category.

    Useful for comprehensive preset to compute per-threat-model scores.

    Args:
        preset_name: Name of preset (typically "comprehensive")

    Returns:
        Dict mapping category names to strategy configs
    """
    preset = load(preset_name, validate=False)
    categorized: Dict[str, List[Dict[str, Any]]] = {
        "natural": [],
        "adversarial": [],
        "realistic_attack": [],
        "uncategorized": [],
    }

    for strat in preset["strategies"]:
        category = strat.get("category", "uncategorized")
        if category in categorized:
            categorized[category].append(strat)
        else:
            categorized["uncategorized"].append(strat)

    return {k: v for k, v in categorized.items() if v}
