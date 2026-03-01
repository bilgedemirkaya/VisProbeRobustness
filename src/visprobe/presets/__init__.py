"""
Preset configurations for VisProbe robustness testing.

Presets are loaded from YAML configuration files, allowing users to:
- Create custom presets without modifying code
- Store presets in version control with clear diffs
- Validate presets with JSON Schema

Presets are stored in:
- Built-in: src/visprobe/config/presets/
- User-defined: ~/.visprobe/presets/ (or VISPROBE_PRESETS_DIR env var)

Example usage:
    >>> from visprobe.presets import load_preset, list_available_presets
    >>>
    >>> # List all available presets
    >>> presets = list_available_presets()
    >>>
    >>> # Load a preset
    >>> preset = load_preset('natural')
    >>>
    >>> # Create and save a custom preset
    >>> from visprobe.presets import save_preset
    >>> my_preset = {
    ...     'name': 'My Preset',
    ...     'description': 'Custom robustness test',
    ...     'strategies': [{'type': 'brightness', 'min_level': 0.5, 'max_level': 1.5}],
    ...     'property': 'label_constant'
    ... }
    >>> save_preset('my_custom', my_preset)
"""

from __future__ import annotations

# Import from submodules
from .loader import (
    BUILTIN_PRESETS_DIR,
    USER_PRESETS_DIR,
    create_user_dir,
    delete,
    get_info,
    get_strategies_by_category,
    list_available,
    list_presets,
    list_threat_aware,
    load,
    load_from_file,
    save,
)
from .threat_models import (
    PRESET_CATEGORIES,
    THREAT_MODELS,
    get_preset_by_threat_model,
    get_threat_model_info,
    is_adversarial_preset,
    is_legacy_preset,
    list_threat_models,
    requires_art,
)
from .validator import (
    check_jsonschema_available,
    get_schema,
    is_valid,
    validate_full,
    validate_preset,
    validate_strategies,
)

# =============================================================================
# Public API (with cleaner names)
# =============================================================================

# Loading
load_preset = load
load_preset_from_file = load_from_file
list_available_presets = list_available
list_threat_aware_presets = list_threat_aware
get_preset_info = get_info

# User presets
save_preset = save
delete_user_preset = delete
create_user_presets_dir = create_user_dir


# =============================================================================
# Normalization Stats
# =============================================================================

# Default normalization stats (ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CIFAR-10 normalization stats
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Loading
    "load_preset",
    "load_preset_from_file",
    "list_available_presets",
    "list_presets",
    "list_threat_aware_presets",
    "get_preset_info",
    "get_strategies_by_category",
    # Validation
    "validate_preset",
    "validate_strategies",
    "validate_full",
    "is_valid",
    "get_schema",
    "check_jsonschema_available",
    # User presets
    "save_preset",
    "delete_user_preset",
    "create_user_presets_dir",
    # Threat models
    "THREAT_MODELS",
    "PRESET_CATEGORIES",
    "get_preset_by_threat_model",
    "get_threat_model_info",
    "is_adversarial_preset",
    "is_legacy_preset",
    "requires_art",
    "list_threat_models",
    # Normalization stats
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "CIFAR_MEAN",
    "CIFAR_STD",
    # Paths
    "BUILTIN_PRESETS_DIR",
    "USER_PRESETS_DIR",
]
