"""
The `visprobe.strategies` module provides a collection of perturbation
strategies that can be used in robustness testing.

All perturbation types inherit from the Strategy base class:
- Adversarial attacks (FGSM, PGD, BIM, APGD, Square)
- Noise perturbations (Gaussian, salt-pepper, uniform, speckle)
- Blur effects (Gaussian, motion, defocus, box)
- Lighting changes (brightness, contrast, gamma, low-light, saturation)
- Spatial transformations (rotation, scale, translation, shear, elastic deformation)
- Composition and blending of strategies
"""

from __future__ import annotations

# Base class
from .base import Strategy

# Adversarial strategies
from .adversarial import (
    APGDStrategy,
    BIMStrategy,
    FGSMStrategy,
    PGDStrategy,
    SquareAttackStrategy,
)

# Legacy image strategies (for backward compatibility)
from .image import (
    BrightnessStrategy,
    ContrastStrategy,
    GaussianBlurStrategy,
    GaussianNoiseStrategy,
    RotateStrategy,
)

# Noise strategies
from .noise import GaussianNoise, SaltPepperNoise, SpeckleNoise, UniformNoise

# Blur strategies
from .blur import BoxBlur, DefocusBlur, GaussianBlur, MotionBlur

# Lighting strategies
from .lighting import Brightness, Contrast, Gamma, LowLight, Saturation

# Spatial strategies
from .spatial import ElasticDeform, Rotation, Scale, Shear, Translation

# Composition strategies
from .composition import Blend, Compose, LevelTransform, LowLightBlur, RandomChoice


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base class
    "Strategy",
    # Adversarial strategies
    "FGSMStrategy",
    "PGDStrategy",
    "BIMStrategy",
    "APGDStrategy",
    "SquareAttackStrategy",
    # Legacy image strategies
    "GaussianNoiseStrategy",
    "GaussianBlurStrategy",
    "BrightnessStrategy",
    "ContrastStrategy",
    "RotateStrategy",
    # Noise strategies
    "GaussianNoise",
    "SaltPepperNoise",
    "UniformNoise",
    "SpeckleNoise",
    # Blur strategies
    "GaussianBlur",
    "MotionBlur",
    "DefocusBlur",
    "BoxBlur",
    # Lighting strategies
    "Brightness",
    "Contrast",
    "Gamma",
    "LowLight",
    "Saturation",
    # Spatial strategies
    "Rotation",
    "Scale",
    "Translation",
    "Shear",
    "ElasticDeform",
    # Composition strategies
    "Compose",
    "RandomChoice",
    "Blend",
    "LowLightBlur",
    "LevelTransform",
]
