"""
Lighting perturbations for robustness testing.

Provides various lighting/color adjustments:
- Brightness
- Contrast
- Gamma correction
- Low light simulation
- Saturation
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import Strategy

__all__ = [
    "Brightness",
    "Contrast",
    "Gamma",
    "LowLight",
    "Saturation",
]


class Brightness(Strategy):
    """
    Brightness adjustment perturbation.

    Adjusts image brightness by multiplying pixel values.

    Args:
        brightness_factor: Brightness multiplier (1.0 = unchanged, <1 = darker, >1 = brighter).
                          If specified, this value is used instead of level parameter.
        clamp: Whether to clamp output to [0, 1] range.

    Example:
        >>> # Option 1: Fixed brightness at construction
        >>> brightness = Brightness(brightness_factor=0.5)
        >>> darker = brightness(images)  # 50% brightness

        >>> # Option 2: Runtime brightness via level
        >>> brightness = Brightness()
        >>> darker = brightness(images, level=0.5)   # 50% brightness
        >>> brighter = brightness(images, level=1.5) # 150% brightness
    """

    name = "brightness"

    def __init__(self, brightness_factor: Optional[float] = None, clamp: bool = True) -> None:
        super().__init__()
        self.brightness_factor = brightness_factor
        self.clamp = clamp

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply brightness adjustment.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Brightness factor (1.0 = unchanged, <1 = darker, >1 = brighter).
                   Ignored if brightness_factor was set in __init__.

        Returns:
            Brightness-adjusted images
        """
        # Use init brightness_factor if provided, otherwise use level
        factor = self.brightness_factor if self.brightness_factor is not None else level

        if factor is None or factor == 1.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        output = imgs * factor

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Brightness(brightness_factor={self.brightness_factor}, clamp={self.clamp})"


class Contrast(Strategy):
    """
    Contrast adjustment perturbation.

    Adjusts contrast by scaling deviation from mean.

    Example:
        >>> contrast = Contrast()
        >>> low_contrast = contrast(images, level=0.5)  # 50% contrast
        >>> high_contrast = contrast(images, level=2.0) # 200% contrast
    """

    name = "contrast"

    def __init__(self, clamp: bool = True) -> None:
        super().__init__()
        self.clamp = clamp

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply contrast adjustment.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Contrast factor (1.0 = unchanged, <1 = less contrast, >1 = more)

        Returns:
            Contrast-adjusted images
        """
        if level is None or level == 1.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)

        # Compute mean per image (keep channel dimension for broadcasting)
        mean = imgs.mean(dim=(-2, -1), keepdim=True)

        # Adjust contrast: output = mean + level * (input - mean)
        output = mean + level * (imgs - mean)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Contrast(clamp={self.clamp})"


class Gamma(Strategy):
    """
    Gamma correction perturbation.

    Applies power-law transformation: output = input^gamma

    Args:
        gain: Multiplier applied before gamma correction (default: 1.0)

    Example:
        >>> gamma = Gamma()
        >>> lighter = gamma(images, level=0.5)  # gamma < 1 brightens
        >>> darker = gamma(images, level=2.0)   # gamma > 1 darkens
    """

    name = "gamma"

    def __init__(self, gain: float = 1.0, clamp: bool = True) -> None:
        super().__init__()
        self.gain = gain
        self.clamp = clamp

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply gamma correction.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Gamma value (1.0 = unchanged)

        Returns:
            Gamma-corrected images
        """
        if level is None:
            level = 1.0

        if level == 1.0 and self.gain == 1.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)

        # Ensure non-negative for power operation
        imgs_clamped = torch.clamp(imgs, 0.0, None)

        # Apply gamma: output = gain * input^gamma
        output = self.gain * torch.pow(imgs_clamped + 1e-8, level)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Gamma(gain={self.gain}, clamp={self.clamp})"


class LowLight(Strategy):
    """
    Low-light simulation perturbation.

    Combines reduced brightness with increased noise to simulate
    low-light conditions.

    Args:
        noise_factor: How much noise to add relative to darkening (default: 0.5)
        seed: Random seed for reproducibility

    Example:
        >>> lowlight = LowLight(noise_factor=0.3)
        >>> dim = lowlight(images, level=0.3)  # 30% brightness with noise
    """

    name = "low_light"

    def __init__(
        self,
        noise_factor: float = 0.5,
        seed: int = None,
    ) -> None:
        super().__init__()
        self.noise_factor = noise_factor
        self.seed = seed

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply low-light simulation.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Brightness factor (0.0-1.0, lower = darker)

        Returns:
            Low-light simulated images
        """
        if level is None or level >= 1.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)

        # Reduce brightness
        output = imgs * level

        # Add noise proportional to darkening
        noise_std = self.noise_factor * (1.0 - level)
        if noise_std > 0:
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=imgs.device)
                generator.manual_seed(self.seed)

            noise = torch.randn(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=generator,
            )
            output = output + noise_std * noise

        output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"LowLight(noise_factor={self.noise_factor}, seed={self.seed})"


class Saturation(Strategy):
    """
    Saturation adjustment perturbation.

    Adjusts color saturation (0 = grayscale, 1 = original, >1 = oversaturated).

    Example:
        >>> sat = Saturation()
        >>> gray = sat(images, level=0.0)     # Grayscale
        >>> vivid = sat(images, level=2.0)    # Double saturation
    """

    name = "saturation"

    def __init__(self, clamp: bool = True) -> None:
        super().__init__()
        self.clamp = clamp

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply saturation adjustment.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W) - assumes RGB
            model: Unused (for API compatibility)
            level: Saturation factor (0 = grayscale, 1 = unchanged)

        Returns:
            Saturation-adjusted images
        """
        if level is None or level == 1.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)

        # Convert to grayscale using luminance weights
        # ITU-R BT.601: Y = 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=imgs.device, dtype=imgs.dtype)
        weights = weights.view(1, 3, 1, 1)

        gray = (imgs * weights).sum(dim=1, keepdim=True)
        gray = gray.expand_as(imgs)

        # Interpolate between grayscale and original
        output = gray + level * (imgs - gray)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Saturation(clamp={self.clamp})"
