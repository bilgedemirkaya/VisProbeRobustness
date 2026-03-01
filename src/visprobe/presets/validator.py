"""
JSON Schema validation for preset configurations.

Validates preset YAML files against a defined schema to ensure
they have the correct structure and valid values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional JSON Schema validation
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# =============================================================================
# Schema Path
# =============================================================================

SCHEMA_PATH = Path(__file__).parent.parent / "config" / "preset_schema.json"


# =============================================================================
# Schema Cache
# =============================================================================

_schema_cache: Optional[Dict[str, Any]] = None


def _load_schema() -> Dict[str, Any]:
    """Load and cache the JSON Schema for preset validation."""
    global _schema_cache
    if _schema_cache is None:
        with open(SCHEMA_PATH, "r") as f:
            _schema_cache = json.load(f)
    return _schema_cache


def get_schema() -> Dict[str, Any]:
    """
    Get the JSON Schema for preset validation.

    Returns:
        The schema as a dictionary
    """
    return _load_schema()


# =============================================================================
# Validation
# =============================================================================

def validate_preset(
    preset: Dict[str, Any],
    raise_on_error: bool = True,
) -> List[str]:
    """
    Validate a preset configuration against the JSON Schema.

    Args:
        preset: Preset configuration dictionary
        raise_on_error: If True, raise ValidationError on failure

    Returns:
        List of validation error messages (empty if valid)

    Raises:
        jsonschema.ValidationError: If raise_on_error is True and validation fails
        ImportError: If jsonschema is not installed and raise_on_error is True
    """
    if not HAS_JSONSCHEMA:
        if raise_on_error:
            raise ImportError(
                "jsonschema is required for preset validation. "
                "Install with: pip install jsonschema"
            )
        return ["jsonschema not installed - validation skipped"]

    schema = _load_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(preset))

    if errors and raise_on_error:
        raise jsonschema.ValidationError(
            f"Preset validation failed: {errors[0].message}",
            path=errors[0].path,
        )

    return [str(e.message) for e in errors]


def is_valid(preset: Dict[str, Any]) -> bool:
    """
    Check if a preset is valid without raising exceptions.

    Args:
        preset: Preset configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    errors = validate_preset(preset, raise_on_error=False)
    return len(errors) == 0


def check_jsonschema_available() -> bool:
    """
    Check if jsonschema is available for validation.

    Returns:
        True if jsonschema is installed
    """
    return HAS_JSONSCHEMA


# =============================================================================
# Detailed Validation
# =============================================================================

def validate_strategies(strategies: List[Dict[str, Any]]) -> List[str]:
    """
    Validate the strategies list of a preset.

    Args:
        strategies: List of strategy configurations

    Returns:
        List of validation error messages
    """
    errors = []

    if not strategies:
        errors.append("Preset must have at least one strategy")
        return errors

    for i, strat in enumerate(strategies):
        if "type" not in strat:
            errors.append(f"Strategy {i}: missing 'type' field")
            continue

        strat_type = strat["type"]

        # Check compositional strategies have components
        if strat_type == "compositional":
            if "components" not in strat or not strat["components"]:
                errors.append(f"Strategy {i} ({strat_type}): missing 'components'")

        # Check level ranges are valid
        if "min_level" in strat and "max_level" in strat:
            if strat["min_level"] > strat["max_level"]:
                errors.append(
                    f"Strategy {i} ({strat_type}): min_level > max_level"
                )

        if "level_range" in strat:
            level_range = strat["level_range"]
            if len(level_range) != 2:
                errors.append(
                    f"Strategy {i} ({strat_type}): level_range must have 2 elements"
                )
            elif level_range[0] > level_range[1]:
                errors.append(
                    f"Strategy {i} ({strat_type}): level_range[0] > level_range[1]"
                )

    return errors


def validate_full(preset: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Perform full validation including detailed strategy checks.

    Args:
        preset: Preset configuration dictionary

    Returns:
        Dict with 'schema_errors' and 'strategy_errors' lists
    """
    result = {
        "schema_errors": validate_preset(preset, raise_on_error=False),
        "strategy_errors": [],
    }

    if "strategies" in preset:
        result["strategy_errors"] = validate_strategies(preset["strategies"])

    return result
