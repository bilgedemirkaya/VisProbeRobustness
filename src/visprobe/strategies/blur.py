"""
Blur perturbations for robustness testing.

Provides various blur methods:
- Gaussian blur
- Motion blur
- Defocus blur
- Box blur
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import Strategy

__all__ = [
    "GaussianBlur",
    "MotionBlur",
    "DefocusBlur",
    "BoxBlur",
]


def _get_gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _get_gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Create 2D Gaussian kernel via outer product."""
    kernel_1d = _get_gaussian_kernel_1d(kernel_size, sigma, device)
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    return kernel_2d


class GaussianBlur(Strategy):
    """
    Gaussian blur perturbation.

    Applies Gaussian smoothing with specified kernel size and sigma.

    Args:
        kernel_size: Size of blur kernel (must be odd). If None, computed from sigma.
        sigma: Blur strength (standard deviation). If specified, this value is used
               instead of the level parameter in generate().

    Example:
        >>> # Option 1: Fixed sigma at construction
        >>> blur = GaussianBlur(sigma=2.0)
        >>> perturbed = blur(images)  # Uses sigma=2.0

        >>> # Option 2: Runtime sigma via level
        >>> blur = GaussianBlur()
        >>> perturbed = blur(images, level=2.0)  # Uses sigma=2.0
    """

    name = "gaussian_blur"

    def __init__(self, kernel_size: Optional[int] = None, sigma: Optional[float] = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._kernel_cache: dict = {}

    def _get_kernel_size(self, sigma: float) -> int:
        """Compute kernel size from sigma if not specified."""
        if self.kernel_size is not None:
            return self.kernel_size
        # Rule of thumb: kernel_size = ceil(6*sigma) | 1 (ensure odd)
        k = int(math.ceil(6 * sigma))
        k = k + 1 if k % 2 == 0 else k
        return max(3, k)

    def _get_kernel(self, sigma: float, device: torch.device) -> torch.Tensor:
        """Get cached or create Gaussian kernel."""
        kernel_size = self._get_kernel_size(sigma)
        cache_key = (kernel_size, sigma, device)

        if cache_key not in self._kernel_cache:
            kernel = _get_gaussian_kernel_2d(kernel_size, sigma, device)
            self._kernel_cache[cache_key] = kernel

        return self._kernel_cache[cache_key]

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply Gaussian blur.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Blur sigma (0.0 = no blur). Ignored if sigma was set in __init__.

        Returns:
            Blurred images
        """
        # Use init sigma if provided, otherwise use level
        effective_sigma = self.sigma if self.sigma is not None else level

        if effective_sigma is None or effective_sigma <= 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        kernel = self._get_kernel(effective_sigma, imgs.device)
        k_size = kernel.shape[0]
        padding = k_size // 2

        # Expand kernel for depthwise convolution: (1, 1, K, K) -> (C, 1, K, K)
        kernel = kernel.view(1, 1, k_size, k_size).expand(c, 1, k_size, k_size)

        # Apply depthwise convolution (separate kernel per channel)
        output = F.conv2d(
            imgs,
            kernel.to(dtype=imgs.dtype),
            padding=padding,
            groups=c,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"GaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma})"


class MotionBlur(Strategy):
    """
    Motion blur perturbation.

    Simulates camera motion by applying directional blur.

    Args:
        angle: Motion direction in degrees (0 = horizontal)
        kernel_size: Fixed kernel size (if None, derived from level)

    Example:
        >>> blur = MotionBlur(angle=45)
        >>> perturbed = blur(images, level=15)  # 15-pixel motion
    """

    name = "motion_blur"

    def __init__(
        self,
        angle: float = 0.0,
        kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.angle = angle
        self.kernel_size = kernel_size
        self._kernel_cache: dict = {}

    def _create_motion_kernel(
        self,
        size: int,
        angle: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Create motion blur kernel."""
        # Create kernel with line at specified angle
        kernel = torch.zeros((size, size), device=device, dtype=torch.float32)

        # Convert angle to radians
        angle_rad = math.radians(angle)
        center = size // 2

        # Draw line through center at specified angle
        for i in range(size):
            x = int(center + (i - center) * math.cos(angle_rad))
            y = int(center + (i - center) * math.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0

        # Normalize
        kernel = kernel / (kernel.sum() + 1e-8)
        return kernel

    def _get_kernel(self, size: int, angle: float, device: torch.device) -> torch.Tensor:
        """Get cached or create motion kernel."""
        cache_key = (size, angle, device)

        if cache_key not in self._kernel_cache:
            kernel = self._create_motion_kernel(size, angle, device)
            self._kernel_cache[cache_key] = kernel

        return self._kernel_cache[cache_key]

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply motion blur.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Kernel size / motion extent (in pixels)

        Returns:
            Motion-blurred images
        """
        if level is None:
            level = 3

        # Use level as kernel size
        kernel_size = self.kernel_size or max(3, int(level))

        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        if kernel_size < 3:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        kernel = self._get_kernel(kernel_size, self.angle, imgs.device)
        padding = kernel_size // 2

        # Expand kernel for depthwise convolution
        kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(c, 1, kernel_size, kernel_size)

        output = F.conv2d(
            imgs,
            kernel.to(dtype=imgs.dtype),
            padding=padding,
            groups=c,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"MotionBlur(angle={self.angle}, kernel_size={self.kernel_size})"


class DefocusBlur(Strategy):
    """
    Defocus (out-of-focus) blur perturbation.

    Simulates optical defocus using a disk-shaped kernel.

    Args:
        None - radius is controlled by level parameter

    Example:
        >>> blur = DefocusBlur()
        >>> perturbed = blur(images, level=5)  # 5-pixel radius defocus
    """

    name = "defocus_blur"

    def __init__(self) -> None:
        super().__init__()
        self._kernel_cache: dict = {}

    def _create_disk_kernel(self, radius: int, device: torch.device) -> torch.Tensor:
        """Create disk-shaped defocus kernel."""
        size = 2 * radius + 1
        y, x = torch.meshgrid(
            torch.arange(size, device=device) - radius,
            torch.arange(size, device=device) - radius,
            indexing='ij',
        )

        # Disk mask
        mask = (x ** 2 + y ** 2 <= radius ** 2).float()

        # Normalize
        kernel = mask / (mask.sum() + 1e-8)
        return kernel

    def _get_kernel(self, radius: int, device: torch.device) -> torch.Tensor:
        """Get cached or create disk kernel."""
        cache_key = (radius, device)

        if cache_key not in self._kernel_cache:
            kernel = self._create_disk_kernel(radius, device)
            self._kernel_cache[cache_key] = kernel

        return self._kernel_cache[cache_key]

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply defocus blur.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Defocus radius in pixels

        Returns:
            Defocused images
        """
        if level is None:
            level = 1

        radius = max(1, int(level))

        if radius < 1:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        kernel = self._get_kernel(radius, imgs.device)
        k_size = kernel.shape[0]
        padding = k_size // 2

        # Expand kernel for depthwise convolution
        kernel = kernel.view(1, 1, k_size, k_size).expand(c, 1, k_size, k_size)

        output = F.conv2d(
            imgs,
            kernel.to(dtype=imgs.dtype),
            padding=padding,
            groups=c,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return "DefocusBlur()"


class BoxBlur(Strategy):
    """
    Box (average) blur perturbation.

    Simple uniform averaging filter.

    Example:
        >>> blur = BoxBlur()
        >>> perturbed = blur(images, level=5)  # 5x5 box filter
    """

    name = "box_blur"

    def __init__(self) -> None:
        super().__init__()
        self._kernel_cache: dict = {}

    def _get_kernel(self, size: int, device: torch.device) -> torch.Tensor:
        """Get cached or create box kernel."""
        cache_key = (size, device)

        if cache_key not in self._kernel_cache:
            kernel = torch.ones((size, size), device=device, dtype=torch.float32)
            kernel = kernel / kernel.sum()
            self._kernel_cache[cache_key] = kernel

        return self._kernel_cache[cache_key]

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply box blur.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Kernel size

        Returns:
            Blurred images
        """
        if level is None:
            level = 3

        kernel_size = max(1, int(level))

        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        if kernel_size < 3:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        kernel = self._get_kernel(kernel_size, imgs.device)
        padding = kernel_size // 2

        # Expand kernel for depthwise convolution
        kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(c, 1, kernel_size, kernel_size)

        output = F.conv2d(
            imgs,
            kernel.to(dtype=imgs.dtype),
            padding=padding,
            groups=c,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return "BoxBlur()"
