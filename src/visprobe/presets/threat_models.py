"""
Threat model definitions for VisProbe robustness testing.

Threat models categorize the type of adversary or perturbation source:
- environmental: Natural/environmental perturbations without active adversary
- adversarial: Gradient-based attacks with full model access
- combined: Adversary exploiting suboptimal environmental conditions
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# =============================================================================
# Threat Model Definitions
# =============================================================================

THREAT_MODELS: Dict[str, Dict[str, Any]] = {
    "environmental": {
        "name": "Environmental (No Adversary)",
        "description": "Natural perturbations without active adversary",
        "examples": ["Weather changes", "Sensor noise", "Compression artifacts"],
        "compute_requirements": "low",
        "requires_gradients": False,
    },
    "adversarial": {
        "name": "Adversarial (White-Box)",
        "description": "Gradient-based attacks with full model access",
        "examples": ["FGSM", "PGD", "DeepFool"],
        "compute_requirements": "high",
        "requires_gradients": True,
    },
    "combined": {
        "name": "Combined (Adversarial + Environmental)",
        "description": "Adversary exploiting suboptimal environmental conditions",
        "examples": ["Low-light + small FGSM", "Blur + tiny PGD"],
        "compute_requirements": "high",
        "requires_gradients": True,
    },
}


# =============================================================================
# Preset Categories
# =============================================================================

PRESET_CATEGORIES: Dict[str, list] = {
    "threat_aware": ["natural", "adversarial", "realistic_attack", "comprehensive"],
    "legacy": ["standard", "lighting", "blur", "corruption"],
    "requires_art": ["adversarial", "realistic_attack", "comprehensive"],
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_threat_model_info(threat_model: str) -> Dict[str, Any]:
    """
    Get information about a threat model.

    Args:
        threat_model: One of "environmental", "adversarial", "combined"

    Returns:
        Dict with name, description, examples, and requirements
    """
    return THREAT_MODELS.get(threat_model, {
        "name": "Unknown",
        "description": "Unknown threat model",
        "examples": [],
        "compute_requirements": "unknown",
        "requires_gradients": False,
    })


def get_preset_by_threat_model(threat_model: str) -> Optional[str]:
    """
    Get the recommended preset for a threat model.

    Args:
        threat_model: One of "environmental", "adversarial", "combined", "all"

    Returns:
        Preset name or None if not found
    """
    mapping = {
        "environmental": "natural",
        "adversarial": "adversarial",
        "combined": "realistic_attack",
        "all": "comprehensive",
    }
    return mapping.get(threat_model)


def is_adversarial_preset(preset_name: str) -> bool:
    """
    Check if a preset requires adversarial attack capabilities.

    Args:
        preset_name: Name of preset

    Returns:
        True if the preset uses adversarial strategies
    """
    return preset_name in PRESET_CATEGORIES.get("requires_art", [])


def requires_art(preset_name: str) -> bool:
    """
    Check if a preset requires the Adversarial Robustness Toolbox.

    Args:
        preset_name: Name of preset

    Returns:
        True if ART is required
    """
    return preset_name in PRESET_CATEGORIES.get("requires_art", [])


def is_legacy_preset(preset_name: str) -> bool:
    """
    Check if a preset is a legacy preset.

    Args:
        preset_name: Name of preset

    Returns:
        True if preset is legacy
    """
    return preset_name in PRESET_CATEGORIES.get("legacy", [])


def list_threat_models() -> list:
    """
    List all available threat models.

    Returns:
        List of threat model names
    """
    return list(THREAT_MODELS.keys())
