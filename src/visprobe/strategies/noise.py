"""
Noise perturbations for robustness testing.

Provides various noise injection methods:
- Gaussian noise
- Salt and pepper noise
- Uniform noise
- Speckle noise
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import Strategy

__all__ = [
    "GaussianNoise",
    "SaltPepperNoise",
    "UniformNoise",
    "SpeckleNoise",
]


class GaussianNoise(Strategy):
    """
    Additive Gaussian noise perturbation.

    Applies noise as: clip(x + std_dev * N(0, 1), 0, 1)

    Args:
        std_dev: Noise standard deviation. If specified, this value is used
                instead of level parameter.
        seed: Random seed for reproducibility (optional)
        clamp: Whether to clamp output to [0, 1] (default: True)

    Example:
        >>> # Option 1: Fixed std_dev at construction
        >>> noise = GaussianNoise(std_dev=0.1)
        >>> perturbed = noise(images)  # Add noise with std=0.1

        >>> # Option 2: Runtime std_dev via level
        >>> noise = GaussianNoise(seed=42)
        >>> perturbed = noise(images, level=0.1)  # Add noise with std=0.1
    """

    name = "gaussian_noise"

    def __init__(
        self,
        std_dev: Optional[float] = None,
        seed: Optional[int] = None,
        clamp: bool = True,
    ) -> None:
        super().__init__()
        self.std_dev = std_dev
        self.seed = seed
        self.clamp = clamp
        self._generator: Optional[torch.Generator] = None

    def _get_generator(self, device: torch.device) -> Optional[torch.Generator]:
        """Get or create random generator for reproducibility."""
        if self.seed is None:
            return None
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        return self._generator

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply Gaussian noise to images.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Noise standard deviation (0.0 = no noise).
                   Ignored if std_dev was set in __init__.

        Returns:
            Noisy images
        """
        # Use init std_dev if provided, otherwise use level
        effective_std = self.std_dev if self.std_dev is not None else level

        if effective_std is None or effective_std == 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        generator = self._get_generator(imgs.device)

        # Generate noise
        noise = torch.randn(
            imgs.shape,
            device=imgs.device,
            dtype=imgs.dtype,
            generator=generator,
        )

        # Apply noise
        output = imgs + effective_std * noise

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"GaussianNoise(std_dev={self.std_dev}, seed={self.seed}, clamp={self.clamp})"


class SaltPepperNoise(Strategy):
    """
    Salt and pepper noise perturbation.

    Randomly sets pixels to 0 (pepper) or 1 (salt).

    Args:
        salt_ratio: Ratio of salt to pepper (default: 0.5 = equal)
        seed: Random seed for reproducibility
    """

    name = "salt_pepper_noise"

    def __init__(
        self,
        salt_ratio: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.salt_ratio = salt_ratio
        self.seed = seed

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply salt and pepper noise.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Fraction of pixels to corrupt (0.0 to 1.0)

        Returns:
            Noisy images
        """
        if level is None or level == 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)

        # Create generator for reproducibility
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=imgs.device)
            generator.manual_seed(self.seed)

        output = imgs.clone()
        n, c, h, w = imgs.shape

        # Generate random mask for affected pixels
        rand_mask = torch.rand(n, 1, h, w, device=imgs.device, generator=generator)

        # Pixels to change
        change_mask = rand_mask < level

        # Salt or pepper decision
        salt_mask = torch.rand(n, 1, h, w, device=imgs.device, generator=generator) < self.salt_ratio

        # Apply salt (1.0) where change_mask AND salt_mask
        salt_pixels = change_mask & salt_mask
        output = torch.where(salt_pixels.expand_as(output), torch.ones_like(output), output)

        # Apply pepper (0.0) where change_mask AND NOT salt_mask
        pepper_pixels = change_mask & ~salt_mask
        output = torch.where(pepper_pixels.expand_as(output), torch.zeros_like(output), output)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"SaltPepperNoise(salt_ratio={self.salt_ratio}, seed={self.seed})"


class UniformNoise(Strategy):
    """
    Additive uniform noise perturbation.

    Applies noise as: clip(x + level * U(-1, 1), 0, 1)

    Args:
        seed: Random seed for reproducibility
        clamp: Whether to clamp output to [0, 1]
    """

    name = "uniform_noise"

    def __init__(
        self,
        seed: Optional[int] = None,
        clamp: bool = True,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.clamp = clamp

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply uniform noise to images.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Noise magnitude (max deviation from original)

        Returns:
            Noisy images
        """
        if level is None or level == 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=imgs.device)
            generator.manual_seed(self.seed)

        # Uniform noise in [-1, 1]
        noise = 2 * torch.rand(imgs.shape, device=imgs.device, dtype=imgs.dtype, generator=generator) - 1

        output = imgs + level * noise

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"UniformNoise(seed={self.seed}, clamp={self.clamp})"


class SpeckleNoise(Strategy):
    """
    Multiplicative speckle noise perturbation.

    Applies noise as: clip(x * (1 + level * N(0, 1)), 0, 1)
    Common in radar/ultrasound images.

    Args:
        seed: Random seed for reproducibility
        clamp: Whether to clamp output to [0, 1]
    """

    name = "speckle_noise"

    def __init__(
        self,
        seed: Optional[int] = None,
        clamp: bool = True,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.clamp = clamp

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply speckle noise to images.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Noise intensity multiplier

        Returns:
            Noisy images
        """
        if level is None or level == 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=imgs.device)
            generator.manual_seed(self.seed)

        noise = torch.randn(imgs.shape, device=imgs.device, dtype=imgs.dtype, generator=generator)

        # Multiplicative noise
        output = imgs * (1 + level * noise)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"SpeckleNoise(seed={self.seed}, clamp={self.clamp})"
