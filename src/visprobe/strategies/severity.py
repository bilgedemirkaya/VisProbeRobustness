"""
Severity-to-level mapping utilities for common perturbation patterns.

Provides convenient factory functions to create strategies with common
severity transformations for experimental workflows.
"""

from __future__ import annotations

from typing import Callable, Optional

from .base import Strategy
from .composition import LevelTransform

__all__ = [
    "linear_scale",
    "brightness_reduction",
    "with_severity",
]


def linear_scale(max_value: float) -> Callable[[Optional[float]], Optional[float]]:
    """
    Create a linear scaling transform: severity -> max_value * severity.

    Args:
        max_value: Maximum value at severity=1.0

    Returns:
        Transform function mapping severity [0,1] to [0, max_value]

    Example:
        >>> from visprobe.strategies import GaussianBlur, with_severity
        >>> from visprobe.strategies.severity import linear_scale
        >>>
        >>> # Blur with sigma ranging from 0 to 3.0
        >>> blur = with_severity(GaussianBlur(), linear_scale(3.0))
        >>> result = blur.generate(images, level=0.5)  # sigma=1.5
    """
    def transform(severity: Optional[float]) -> Optional[float]:
        if severity is None:
            return None
        return max_value * severity
    return transform


def brightness_reduction(max_reduction: float = 0.7) -> Callable[[Optional[float]], Optional[float]]:
    """
    Create brightness reduction transform for low-light simulation.

    Maps severity to brightness factor:
    - severity=0.0 -> brightness=1.0 (no reduction)
    - severity=1.0 -> brightness=(1.0 - max_reduction)

    Args:
        max_reduction: Maximum brightness reduction (default: 0.7 -> 30% brightness at s=1)

    Returns:
        Transform function mapping severity to brightness factor

    Example:
        >>> from visprobe.strategies import Brightness, with_severity
        >>> from visprobe.strategies.severity import brightness_reduction
        >>>
        >>> # Brightness reduction: s=0 -> 100%, s=1 -> 30%
        >>> lowlight = with_severity(Brightness(), brightness_reduction(0.7))
        >>> result = lowlight.generate(images, level=0.5)  # 65% brightness
    """
    def transform(severity: Optional[float]) -> Optional[float]:
        if severity is None:
            return None
        return 1.0 - max_reduction * severity
    return transform


def with_severity(
    strategy: Strategy,
    transform: Callable[[Optional[float]], Optional[float]],
) -> LevelTransform:
    """
    Convenience wrapper to apply severity transformation to a strategy.

    This is a shorthand for LevelTransform(strategy, transform).

    Args:
        strategy: The base strategy to wrap
        transform: Severity-to-level transformation function

    Returns:
        LevelTransform strategy with the specified mapping

    Example:
        >>> from visprobe.strategies import GaussianBlur, GaussianNoise, Brightness
        >>> from visprobe.strategies.severity import with_severity, linear_scale, brightness_reduction
        >>>
        >>> # Create severity-mapped strategies
        >>> blur = with_severity(GaussianBlur(), linear_scale(3.0))
        >>> noise = with_severity(GaussianNoise(seed=42), linear_scale(0.1))
        >>> lowlight = with_severity(Brightness(), brightness_reduction(0.7))
        >>>
        >>> # Use with uniform severity parameter
        >>> blurred = blur.generate(images, level=0.5)    # sigma=1.5
        >>> noisy = noise.generate(images, level=0.5)     # std=0.05
        >>> dark = lowlight.generate(images, level=0.5)   # brightness=0.65
    """
    return LevelTransform(strategy, transform)


# Common pre-configured transforms for experiments
def gaussian_blur_severity(sigma_max: float = 3.0) -> LevelTransform:
    """
    Create Gaussian blur with linear severity mapping.

    Args:
        sigma_max: Maximum blur sigma at severity=1.0

    Returns:
        GaussianBlur strategy with severity mapping

    Example:
        >>> from visprobe.strategies.severity import gaussian_blur_severity
        >>> blur = gaussian_blur_severity(sigma_max=3.0)
        >>> result = blur.generate(images, level=0.5)  # sigma=1.5
    """
    from .blur import GaussianBlur
    return with_severity(GaussianBlur(), linear_scale(sigma_max))


def gaussian_noise_severity(std_max: float = 0.1, seed: Optional[int] = None) -> LevelTransform:
    """
    Create Gaussian noise with linear severity mapping.

    Args:
        std_max: Maximum noise std at severity=1.0
        seed: Random seed for reproducibility

    Returns:
        GaussianNoise strategy with severity mapping

    Example:
        >>> from visprobe.strategies.severity import gaussian_noise_severity
        >>> noise = gaussian_noise_severity(std_max=0.1, seed=42)
        >>> result = noise.generate(images, level=0.5)  # std=0.05
    """
    from .noise import GaussianNoise
    return with_severity(GaussianNoise(seed=seed), linear_scale(std_max))


def lowlight_severity(max_reduction: float = 0.7) -> LevelTransform:
    """
    Create brightness reduction (low-light) with severity mapping.

    Args:
        max_reduction: Maximum brightness reduction at severity=1.0

    Returns:
        Brightness strategy with severity mapping

    Example:
        >>> from visprobe.strategies.severity import lowlight_severity
        >>> lowlight = lowlight_severity(max_reduction=0.7)
        >>> result = lowlight.generate(images, level=0.5)  # brightness=0.65
    """
    from .lighting import Brightness
    return with_severity(Brightness(), brightness_reduction(max_reduction))
