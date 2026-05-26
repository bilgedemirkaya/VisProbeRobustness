"""
Environmental perturbations for vision robustness testing.

Each perturbation is a callable: ``perturbation(images, severity) -> images``
where severity is in [0, 1]. severity=0 is a no-op.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Callable
import numpy as np


class GaussianBlur:
    """Gaussian blur perturbation."""

    def __init__(self, sigma_max: float = 3.0):
        """
        Args:
            sigma_max: Sigma value at severity=1.0.
        """
        self.sigma_max = sigma_max

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        if severity <= 0:
            return images

        sigma = severity * self.sigma_max

        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=images.dtype, device=images.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        channels = images.shape[1]
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

        padding = kernel_size // 2
        return F.conv2d(images, kernel_2d, padding=padding, groups=channels)


class GaussianNoise:
    """Additive Gaussian noise perturbation."""

    def __init__(self, std_max: float = 0.1, seed: Optional[int] = None):
        """
        Args:
            std_max: Noise standard deviation at severity=1.0.
            seed: Optional RNG seed for reproducibility.
        """
        self.std_max = std_max
        self.seed = seed
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        if severity <= 0:
            return images

        std = severity * self.std_max
        if self.generator is not None:
            noise = torch.randn(
                images.shape,
                dtype=images.dtype,
                device=images.device,
                generator=self.generator,
            ) * std
        else:
            noise = torch.randn_like(images) * std

        return torch.clamp(images + noise, 0, 1)


class Brightness:
    """Brightness shift perturbation."""

    def __init__(self, delta_max: float = 0.3):
        """
        Args:
            delta_max: Brightness offset at severity=1.0. Use negative severity to darken.
        """
        self.delta_max = delta_max

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        if abs(severity) < 1e-10:
            return images
        delta = severity * self.delta_max
        return torch.clamp(images + delta, 0, 1)


class LowLight:
    """Low-light simulation via gamma correction (higher severity = darker)."""

    def __init__(self, gamma_max: float = 5.0):
        """
        Args:
            gamma_max: Gamma value at severity=1.0.
        """
        self.gamma_max = gamma_max

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        if severity <= 0:
            return images
        gamma = 1.0 + severity * (self.gamma_max - 1.0)
        return torch.clamp(torch.pow(images, gamma), 0, 1)


def get_standard_perturbations() -> Dict[str, Callable]:
    """Standard 4-perturbation set covering blur, noise, lighting."""
    return {
        "blur": GaussianBlur(sigma_max=3.0),
        "noise": GaussianNoise(std_max=0.1),
        "brightness": Brightness(delta_max=0.3),
        "lowlight": LowLight(gamma_max=5.0),
    }
