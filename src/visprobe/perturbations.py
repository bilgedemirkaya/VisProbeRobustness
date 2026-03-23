"""
Simplified environmental perturbations for vision models.
Clean, transparent implementations without hidden complexity.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Callable
import numpy as np


class GaussianBlur:
    """Gaussian blur perturbation."""

    def __init__(self, sigma_max: float = 3.0):
        """
        Initialize Gaussian blur.

        Args:
            sigma_max: Maximum sigma value at severity=1.0
        """
        self.sigma_max = sigma_max

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Apply Gaussian blur.

        Args:
            images: Input images (B, C, H, W)
            severity: Severity level in [0, 1]

        Returns:
            Blurred images
        """
        if severity <= 0:
            return images

        sigma = severity * self.sigma_max

        # Create Gaussian kernel
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Generate 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=images.dtype, device=images.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D kernel
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        # Apply blur per channel
        channels = images.shape[1]
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

        # Apply convolution with padding
        padding = kernel_size // 2
        blurred = F.conv2d(
            images,
            kernel_2d,
            padding=padding,
            groups=channels
        )

        return blurred


class GaussianNoise:
    """Gaussian noise perturbation."""

    def __init__(self, std_max: float = 0.1, seed: Optional[int] = None):
        """
        Initialize Gaussian noise.

        Args:
            std_max: Maximum standard deviation at severity=1.0
            seed: Random seed for reproducibility
        """
        self.std_max = std_max
        self.seed = seed
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Add Gaussian noise.

        Args:
            images: Input images (B, C, H, W)
            severity: Severity level in [0, 1]

        Returns:
            Noisy images
        """
        if severity <= 0:
            return images

        std = severity * self.std_max

        # Generate noise
        if self.generator is not None:
            noise = torch.randn(
                images.shape,
                dtype=images.dtype,
                device=images.device,
                generator=self.generator
            ) * std
        else:
            noise = torch.randn_like(images) * std

        # Add noise and clamp
        noisy = images + noise
        noisy = torch.clamp(noisy, 0, 1)

        return noisy


class Brightness:
    """Brightness adjustment perturbation."""

    def __init__(self, delta_max: float = 0.3):
        """
        Initialize brightness adjustment.

        Args:
            delta_max: Maximum brightness change at severity=1.0
        """
        self.delta_max = delta_max

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Adjust brightness.

        Args:
            images: Input images (B, C, H, W)
            severity: Severity level in [0, 1] (negative for darkening)

        Returns:
            Brightness-adjusted images
        """
        if abs(severity) < 1e-10:
            return images

        delta = severity * self.delta_max

        # Apply brightness adjustment
        adjusted = images + delta
        adjusted = torch.clamp(adjusted, 0, 1)

        return adjusted


class Contrast:
    """Contrast adjustment perturbation."""

    def __init__(self, factor_range: tuple = (0.5, 1.5)):
        """
        Initialize contrast adjustment.

        Args:
            factor_range: (min_factor, max_factor) for contrast
        """
        self.factor_min, self.factor_max = factor_range

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Adjust contrast.

        Args:
            images: Input images (B, C, H, W)
            severity: Severity level in [0, 1]

        Returns:
            Contrast-adjusted images
        """
        if severity <= 0:
            return images

        # Map severity to contrast factor
        # severity=0 -> factor=1 (no change)
        # severity=1 -> factor=factor_min or factor_max
        if severity <= 0.5:
            # Decrease contrast
            factor = 1.0 - 2 * severity * (1.0 - self.factor_min)
        else:
            # Increase contrast
            factor = 1.0 + 2 * (severity - 0.5) * (self.factor_max - 1.0)

        # Apply contrast adjustment
        mean = images.mean(dim=(2, 3), keepdim=True)
        adjusted = (images - mean) * factor + mean
        adjusted = torch.clamp(adjusted, 0, 1)

        return adjusted


class LowLight:
    """Low-light condition simulation using gamma correction."""

    def __init__(self, gamma_max: float = 5.0):
        """
        Initialize low-light simulation.

        Args:
            gamma_max: Maximum gamma value at severity=1.0 (higher = darker)
        """
        self.gamma_max = gamma_max

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Apply low-light condition.

        Args:
            images: Input images (B, C, H, W)
            severity: Severity level in [0, 1]

        Returns:
            Darkened images
        """
        if severity <= 0:
            return images

        # Map severity to gamma
        # severity=0 -> gamma=1 (no change)
        # severity=1 -> gamma=gamma_max (darkest)
        gamma = 1.0 + severity * (self.gamma_max - 1.0)

        # Apply gamma correction
        adjusted = torch.pow(images, gamma)
        adjusted = torch.clamp(adjusted, 0, 1)

        return adjusted


class MotionBlur:
    """Motion blur perturbation."""

    def __init__(self, kernel_size_max: int = 15):
        """
        Initialize motion blur.

        Args:
            kernel_size_max: Maximum kernel size at severity=1.0
        """
        self.kernel_size_max = kernel_size_max

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Apply motion blur.

        Args:
            images: Input images (B, C, H, W)
            severity: Severity level in [0, 1]

        Returns:
            Motion-blurred images
        """
        if severity <= 0:
            return images

        # Calculate kernel size
        kernel_size = int(3 + severity * (self.kernel_size_max - 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create motion blur kernel (horizontal)
        kernel = torch.zeros(kernel_size, kernel_size, dtype=images.dtype, device=images.device)
        kernel[kernel_size // 2, :] = 1.0
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        # Apply per channel
        channels = images.shape[1]
        kernel = kernel.repeat(channels, 1, 1, 1)

        # Apply convolution
        padding = kernel_size // 2
        blurred = F.conv2d(
            images,
            kernel,
            padding=padding,
            groups=channels
        )

        return blurred


class SaltPepperNoise:
    """Salt and pepper noise perturbation."""

    def __init__(self, prob_max: float = 0.05, seed: Optional[int] = None):
        """
        Initialize salt and pepper noise.

        Args:
            prob_max: Maximum probability at severity=1.0
            seed: Random seed
        """
        self.prob_max = prob_max
        self.seed = seed
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Add salt and pepper noise.

        Args:
            images: Input images (B, C, H, W)
            severity: Severity level in [0, 1]

        Returns:
            Noisy images
        """
        if severity <= 0:
            return images

        prob = severity * self.prob_max

        # Generate random mask
        if self.generator is not None:
            mask = torch.rand(
                images.shape,
                dtype=images.dtype,
                device=images.device,
                generator=self.generator
            )
        else:
            mask = torch.rand_like(images)

        # Apply salt and pepper
        noisy = images.clone()
        noisy[mask < prob / 2] = 0  # Pepper
        noisy[mask > 1 - prob / 2] = 1  # Salt

        return noisy


class Compose:
    """Compose multiple perturbations sequentially."""

    def __init__(self, perturbations: list):
        """
        Initialize composition.

        Args:
            perturbations: List of perturbation functions
        """
        self.perturbations = perturbations

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """
        Apply all perturbations in sequence.

        Args:
            images: Input images
            severity: Severity level applied to all perturbations

        Returns:
            Perturbed images
        """
        result = images
        for perturbation in self.perturbations:
            result = perturbation(result, severity)
        return result


def get_standard_perturbations() -> Dict[str, Callable]:
    """
    Get standard set of environmental perturbations.

    Returns:
        Dictionary of perturbation name to function
    """
    return {
        "blur": GaussianBlur(sigma_max=3.0),
        "motion_blur": MotionBlur(kernel_size_max=15),
        "noise": GaussianNoise(std_max=0.1),
        "salt_pepper": SaltPepperNoise(prob_max=0.05),
        "brightness": Brightness(delta_max=0.3),
        "contrast": Contrast(factor_range=(0.5, 1.5)),
        "lowlight": LowLight(gamma_max=5.0),
    }


def get_minimal_perturbations() -> Dict[str, Callable]:
    """
    Get minimal set of key perturbations.

    Returns:
        Dictionary of perturbation name to function
    """
    return {
        "blur": GaussianBlur(sigma_max=3.0),
        "noise": GaussianNoise(std_max=0.1),
        "lowlight": LowLight(gamma_max=5.0),
    }