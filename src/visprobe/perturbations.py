"""
Perturbation catalog with sensible defaults.

Provides a high-level API for common perturbations without requiring users
to know parameter ranges, strategy construction, or normalization details.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from .strategies.base import Strategy
from .strategies.image import (
    BrightnessStrategy,
    ContrastStrategy,
    GammaStrategy,
    GaussianBlurStrategy,
    GaussianNoiseStrategy,
    JPEGCompressionStrategy,
    MotionBlurStrategy,
    RotateStrategy,
)


@dataclass
class PerturbationSpec:
    """Specification for a perturbation type."""

    name: str
    """Human-readable name"""

    strategy_class: type
    """Strategy class to instantiate"""

    param_name: str
    """Parameter name that varies with level"""

    default_range: Tuple[float, float]
    """Default (level_lo, level_hi) for search"""

    imagenet_range: Optional[Tuple[float, float]] = None
    """ImageNet-specific range (if different from default)"""

    cifar_range: Optional[Tuple[float, float]] = None
    """CIFAR-specific range (if different from default)"""

    description: str = ""
    """Brief description of the perturbation"""

    def get_range(self, preset: Optional[str] = None) -> Tuple[float, float]:
        """Get appropriate range based on dataset preset."""
        if preset == "imagenet" and self.imagenet_range:
            return self.imagenet_range
        elif preset in ("cifar10", "cifar100") and self.cifar_range:
            return self.cifar_range
        return self.default_range

    def create_factory(self) -> Callable[[float], Strategy]:
        """Create a strategy factory function."""
        param_name = self.param_name
        strategy_class = self.strategy_class

        def factory(level: float) -> Strategy:
            return strategy_class(**{param_name: level})

        return factory


# Perturbation catalog with sensible defaults
PERTURBATIONS: Dict[str, PerturbationSpec] = {
    # Noise perturbations
    "gaussian_noise": PerturbationSpec(
        name="Gaussian Noise",
        strategy_class=GaussianNoiseStrategy,
        param_name="std_dev",
        default_range=(0.0, 0.15),
        imagenet_range=(0.0, 0.12),
        cifar_range=(0.0, 0.15),
        description="Additive Gaussian noise in pixel space",
    ),

    # Blur perturbations
    "gaussian_blur": PerturbationSpec(
        name="Gaussian Blur",
        strategy_class=GaussianBlurStrategy,
        param_name="sigma",
        default_range=(0.0, 3.0),
        imagenet_range=(0.0, 10.0),  # Expanded to find actual failure point
        cifar_range=(0.0, 2.5),
        description="Gaussian blur filter",
    ),

    "motion_blur": PerturbationSpec(
        name="Motion Blur",
        strategy_class=MotionBlurStrategy,
        param_name="kernel_size",
        default_range=(3.0, 25.0),
        imagenet_range=(3.0, 31.0),
        cifar_range=(3.0, 21.0),
        description="Directional motion blur",
    ),

    # Brightness perturbations
    "brightness_increase": PerturbationSpec(
        name="Brightness (Brighter)",
        strategy_class=BrightnessStrategy,
        param_name="brightness_factor",
        default_range=(1.0, 2.0),
        imagenet_range=(1.0, 2.5),
        cifar_range=(1.0, 2.0),
        description="Increase image brightness",
    ),

    "brightness_decrease": PerturbationSpec(
        name="Brightness (Darker)",
        strategy_class=BrightnessStrategy,
        param_name="brightness_factor",
        default_range=(0.2, 1.0),
        imagenet_range=(0.1, 1.0),
        cifar_range=(0.3, 1.0),
        description="Decrease image brightness",
    ),

    # Contrast perturbations
    "contrast_increase": PerturbationSpec(
        name="Contrast (Higher)",
        strategy_class=ContrastStrategy,
        param_name="contrast_factor",
        default_range=(1.0, 2.5),
        imagenet_range=(1.0, 3.0),
        cifar_range=(1.0, 2.0),
        description="Increase image contrast",
    ),

    "contrast_decrease": PerturbationSpec(
        name="Contrast (Lower)",
        strategy_class=ContrastStrategy,
        param_name="contrast_factor",
        default_range=(0.2, 1.0),
        imagenet_range=(0.1, 1.0),
        cifar_range=(0.3, 1.0),
        description="Decrease image contrast",
    ),

    # Gamma correction
    "gamma_bright": PerturbationSpec(
        name="Gamma (Brighter)",
        strategy_class=GammaStrategy,
        param_name="gamma",
        default_range=(0.3, 1.0),
        imagenet_range=(0.2, 1.0),
        cifar_range=(0.4, 1.0),
        description="Gamma correction (darker to neutral)",
    ),

    "gamma_dark": PerturbationSpec(
        name="Gamma (Darker)",
        strategy_class=GammaStrategy,
        param_name="gamma",
        default_range=(1.0, 3.0),
        imagenet_range=(1.0, 3.5),
        cifar_range=(1.0, 2.5),
        description="Gamma correction (neutral to darker)",
    ),

    # Geometric perturbations
    "rotation": PerturbationSpec(
        name="Rotation",
        strategy_class=RotateStrategy,
        param_name="angle",
        default_range=(0.0, 45.0),
        imagenet_range=(0.0, 60.0),
        cifar_range=(0.0, 45.0),
        description="Rotation in degrees",
    ),

    # Compression artifacts
    "jpeg_compression": PerturbationSpec(
        name="JPEG Compression",
        strategy_class=JPEGCompressionStrategy,
        param_name="quality",
        default_range=(10.0, 100.0),
        imagenet_range=(5.0, 100.0),
        cifar_range=(10.0, 100.0),
        description="JPEG compression quality degradation",
    ),
}


def list_perturbations() -> Dict[str, str]:
    """
    Get available perturbations with descriptions.

    Returns:
        Dict mapping perturbation ID to description
    """
    return {
        pert_id: f"{spec.name} - {spec.description}"
        for pert_id, spec in PERTURBATIONS.items()
    }


def get_perturbation(
    perturbation: str,
    preset: Optional[str] = None,
) -> Tuple[Callable[[float], Strategy], float, float, str]:
    """
    Get a perturbation with appropriate ranges.

    Args:
        perturbation: Perturbation ID (e.g., "gaussian_noise")
        preset: Dataset preset for range selection ("imagenet", "cifar10", etc.)

    Returns:
        Tuple of (strategy_factory, level_lo, level_hi, name)

    Raises:
        ValueError: If perturbation ID is not recognized

    Example:
        >>> factory, lo, hi, name = get_perturbation("gaussian_noise", preset="imagenet")
        >>> report = search(model, data, strategy=factory, level_lo=lo, level_hi=hi)
    """
    if perturbation not in PERTURBATIONS:
        available = ", ".join(PERTURBATIONS.keys())
        raise ValueError(
            f"Unknown perturbation: '{perturbation}'. "
            f"Available: {available}"
        )

    spec = PERTURBATIONS[perturbation]
    level_lo, level_hi = spec.get_range(preset)
    factory = spec.create_factory()

    return factory, level_lo, level_hi, spec.name


# =============================================================================
# Named Constants for IDE Autocomplete
# =============================================================================

class Perturbation:
    """
    Named constants for perturbation types.

    Use these for IDE autocomplete and type safety instead of string literals.

    Example:
        >>> from visprobe.perturbations import Perturbation as P
        >>> perturbations = [P.GAUSSIAN_NOISE, P.GAUSSIAN_BLUR, P.ROTATION]
        >>> report = search(model, data, perturbation=P.GAUSSIAN_NOISE)
    """
    # Noise
    GAUSSIAN_NOISE = "gaussian_noise"

    # Blur
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"

    # Brightness
    BRIGHTNESS_INCREASE = "brightness_increase"
    BRIGHTNESS_DECREASE = "brightness_decrease"

    # Contrast
    CONTRAST_INCREASE = "contrast_increase"
    CONTRAST_DECREASE = "contrast_decrease"

    # Gamma
    GAMMA_BRIGHT = "gamma_bright"
    GAMMA_DARK = "gamma_dark"

    # Geometric
    ROTATION = "rotation"

    # Compression
    JPEG_COMPRESSION = "jpeg_compression"

    @classmethod
    def all(cls) -> list[str]:
        """Get all available perturbation names."""
        return [
            cls.GAUSSIAN_NOISE,
            cls.GAUSSIAN_BLUR,
            cls.MOTION_BLUR,
            cls.BRIGHTNESS_INCREASE,
            cls.BRIGHTNESS_DECREASE,
            cls.CONTRAST_INCREASE,
            cls.CONTRAST_DECREASE,
            cls.GAMMA_BRIGHT,
            cls.GAMMA_DARK,
            cls.ROTATION,
            cls.JPEG_COMPRESSION,
        ]
