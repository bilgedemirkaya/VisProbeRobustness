"""
VisProbe: Find robustness failures in your vision models in 5 minutes.

VisProbe tests **accuracy preservation** under perturbation: models should maintain
correct predictions when inputs are perturbed with realistic noise, blur, lighting
changes, etc. Only correctly-classified samples are tested (samples where
model_prediction == true_label).

Quick Start:
    >>> from visprobe import search, list_presets
    >>> # See available presets
    >>> list_presets()
    >>> # Run multi-perturbation test
    >>> report = search(model, data, preset="natural")
    >>> report.show()

Single perturbation testing:
    >>> from visprobe import search, list_perturbations, Perturbation
    >>> # See available perturbations
    >>> list_perturbations()
    >>> # Test a specific perturbation
    >>> report = search(model, data, perturbation="gaussian_blur")
    >>> # Or use named constants (IDE autocomplete!)
    >>> report = search(model, data, perturbation=Perturbation.GAUSSIAN_NOISE)

Advanced usage (full control):
    >>> from visprobe.strategies.image import GaussianNoiseStrategy
    >>> report = search(model, data, strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    ...                 level_lo=0.0, level_hi=0.15)

Search methods:
    >>> # Adaptive search (default) - fast, good for unknown ranges
    >>> report = search(model, data, perturbation="gaussian_noise", search_method="adaptive")
    >>> # Binary search - efficient for known ranges
    >>> report = search(model, data, perturbation="gaussian_noise", search_method="binary")
    >>> # Bayesian optimization - query-efficient, provides confidence intervals
    >>> report = search(model, data, perturbation="gaussian_noise", search_method="bayesian")
"""

__version__ = "0.2.0"

# Primary API
from .api import search
from .report import Report, generate_insights
from .core.normalization import NormalizationHandler, NORMALIZATION_PRESETS
from .perturbations import list_perturbations, get_perturbation, Perturbation
from .presets import list_presets as _list_presets_raw, get_preset_info

# Modules
from . import presets, properties, strategies, analysis


# Preset discovery wrapper for cleaner output
def list_presets(verbose: bool = False):
    """
    List all available presets for multi-perturbation testing.

    Args:
        verbose: If True, print detailed information for each preset.
                 If False, return dict of {name: description}.

    Returns:
        Dict mapping preset name to description (if verbose=False)
        None (if verbose=True, prints to stdout)

    Example:
        >>> from visprobe import list_presets
        >>> presets = list_presets()
        >>> print(presets.keys())
        dict_keys(['natural', 'lighting', 'weather', 'geometric', ...])

        >>> list_presets(verbose=True)
        natural: Natural image perturbations (blur, noise, brightness)
          Threat Model: natural_perturbations
        lighting: Lighting and color variations
          Threat Model: environmental_variation
        ...
    """
    preset_list = _list_presets_raw(include_legacy=False)

    if not verbose:
        # Return simple dict of {name: description}
        return {name: desc for name, desc, _ in preset_list}
    else:
        # Print detailed information
        print("Available Presets:")
        print("=" * 70)
        for name, description, threat_model in preset_list:
            print(f"\n{name}:")
            print(f"  {description}")
            print(f"  Threat Model: {threat_model}")
        print("\n" + "=" * 70)
        print(f"Total: {len(preset_list)} presets")
        print("\nUsage:")
        print("  search(model, data, preset='<name>')")
        print("\nFor detailed info on a specific preset:")
        print("  get_preset_info('<name>')")
        return None

__all__ = [
    # Primary API
    "search",               # Unified search function (preset or single perturbation)
    "Report",               # Report class
    "generate_insights",    # AI-style insights from multiple reports
    # Perturbations
    "list_perturbations",   # List available perturbations
    "get_perturbation",     # Get perturbation spec (advanced)
    "Perturbation",         # Named constants for perturbations
    # Presets
    "list_presets",         # List available presets
    "get_preset_info",      # Get detailed info on a preset
    # Normalization
    "NormalizationHandler", # Normalization handler class
    "NORMALIZATION_PRESETS",# Available normalization presets
    # Modules
    "properties",
    "strategies",
    "presets",
    "analysis",
]
