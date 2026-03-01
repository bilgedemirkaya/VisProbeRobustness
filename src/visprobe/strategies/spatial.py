"""
Spatial perturbations for robustness testing.

Provides various geometric transformations:
- Rotation
- Scale
- Translation
- Affine transformations
- Elastic deformation
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import Strategy

__all__ = [
    "Rotation",
    "Scale",
    "Translation",
    "Shear",
    "ElasticDeform",
]


class Rotation(Strategy):
    """
    Rotation perturbation.

    Rotates images around the center.

    Args:
        fill: Value to fill empty areas (default: 0.0)
        mode: Interpolation mode ('bilinear' or 'nearest')

    Example:
        >>> rotate = Rotation()
        >>> rotated = rotate(images, level=15)  # Rotate 15 degrees
    """

    name = "rotation"

    def __init__(
        self,
        fill: float = 0.0,
        mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.fill = fill
        self.mode = mode

    def _get_rotation_matrix(
        self,
        angle_degrees: float,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create 2x3 affine rotation matrix."""
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Rotation matrix (2x3 for affine_grid)
        # This rotates around the center of the image
        matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
        ], device=device, dtype=dtype)

        return matrix

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply rotation.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            Rotated images
        """
        if level is None or level == 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        # Get rotation matrix
        matrix = self._get_rotation_matrix(level, h, w, imgs.device, imgs.dtype)
        matrix = matrix.unsqueeze(0).expand(n, -1, -1)

        # Create sampling grid
        grid = F.affine_grid(matrix, imgs.shape, align_corners=False)

        # Sample with interpolation
        output = F.grid_sample(
            imgs,
            grid,
            mode=self.mode,
            padding_mode="zeros",
            align_corners=False,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Rotation(fill={self.fill}, mode={self.mode})"


class Scale(Strategy):
    """
    Scale (zoom) perturbation.

    Scales images from the center.

    Args:
        mode: Interpolation mode ('bilinear' or 'nearest')

    Example:
        >>> scale = Scale()
        >>> zoomed_in = scale(images, level=1.2)   # 120% zoom
        >>> zoomed_out = scale(images, level=0.8)  # 80% zoom
    """

    name = "scale"

    def __init__(self, mode: str = "bilinear") -> None:
        super().__init__()
        self.mode = mode

    def _get_scale_matrix(
        self,
        scale_factor: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create 2x3 affine scale matrix."""
        # Scale matrix (inverse scale for grid_sample)
        inv_scale = 1.0 / scale_factor if scale_factor != 0 else 1.0
        matrix = torch.tensor([
            [inv_scale, 0, 0],
            [0, inv_scale, 0],
        ], device=device, dtype=dtype)

        return matrix

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply scaling.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Scale factor (1.0 = unchanged, >1 = zoom in, <1 = zoom out)

        Returns:
            Scaled images
        """
        if level is None or level == 1.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        matrix = self._get_scale_matrix(level, imgs.device, imgs.dtype)
        matrix = matrix.unsqueeze(0).expand(n, -1, -1)

        grid = F.affine_grid(matrix, imgs.shape, align_corners=False)

        output = F.grid_sample(
            imgs,
            grid,
            mode=self.mode,
            padding_mode="zeros",
            align_corners=False,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Scale(mode={self.mode})"


class Translation(Strategy):
    """
    Translation perturbation.

    Shifts images horizontally and/or vertically.

    Args:
        direction: Direction of translation ('horizontal', 'vertical', or 'both')
        mode: Interpolation mode

    Example:
        >>> translate = Translation(direction='horizontal')
        >>> shifted = translate(images, level=0.1)  # Shift 10% of width
    """

    name = "translation"

    def __init__(
        self,
        direction: str = "both",
        mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if direction not in ("horizontal", "vertical", "both"):
            raise ValueError(f"direction must be 'horizontal', 'vertical', or 'both', got {direction}")
        self.direction = direction
        self.mode = mode

    def _get_translation_matrix(
        self,
        tx: float,
        ty: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create 2x3 affine translation matrix."""
        matrix = torch.tensor([
            [1, 0, tx],
            [0, 1, ty],
        ], device=device, dtype=dtype)

        return matrix

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply translation.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Translation amount as fraction of image size (-1 to 1)

        Returns:
            Translated images
        """
        if level is None or level == 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        # Determine translation direction
        if self.direction == "horizontal":
            tx, ty = level, 0.0
        elif self.direction == "vertical":
            tx, ty = 0.0, level
        else:  # both
            tx, ty = level, level

        matrix = self._get_translation_matrix(tx, ty, imgs.device, imgs.dtype)
        matrix = matrix.unsqueeze(0).expand(n, -1, -1)

        grid = F.affine_grid(matrix, imgs.shape, align_corners=False)

        output = F.grid_sample(
            imgs,
            grid,
            mode=self.mode,
            padding_mode="zeros",
            align_corners=False,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Translation(direction={self.direction}, mode={self.mode})"


class Shear(Strategy):
    """
    Shear perturbation.

    Applies shear transformation to images.

    Args:
        axis: Shear axis ('horizontal' or 'vertical')
        mode: Interpolation mode

    Example:
        >>> shear = Shear(axis='horizontal')
        >>> sheared = shear(images, level=0.2)  # 20% horizontal shear
    """

    name = "shear"

    def __init__(
        self,
        axis: str = "horizontal",
        mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if axis not in ("horizontal", "vertical"):
            raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis}")
        self.axis = axis
        self.mode = mode

    def _get_shear_matrix(
        self,
        shear: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create 2x3 affine shear matrix."""
        if self.axis == "horizontal":
            matrix = torch.tensor([
                [1, shear, 0],
                [0, 1, 0],
            ], device=device, dtype=dtype)
        else:
            matrix = torch.tensor([
                [1, 0, 0],
                [shear, 1, 0],
            ], device=device, dtype=dtype)

        return matrix

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply shear transformation.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Shear factor

        Returns:
            Sheared images
        """
        if level is None or level == 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        matrix = self._get_shear_matrix(level, imgs.device, imgs.dtype)
        matrix = matrix.unsqueeze(0).expand(n, -1, -1)

        grid = F.affine_grid(matrix, imgs.shape, align_corners=False)

        output = F.grid_sample(
            imgs,
            grid,
            mode=self.mode,
            padding_mode="zeros",
            align_corners=False,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Shear(axis={self.axis}, mode={self.mode})"


class ElasticDeform(Strategy):
    """
    Elastic deformation perturbation.

    Applies smooth random displacements to simulate elastic distortion.

    Args:
        grid_size: Size of displacement grid (smaller = smoother)
        seed: Random seed for reproducibility

    Example:
        >>> elastic = ElasticDeform(grid_size=4)
        >>> deformed = elastic(images, level=0.1)  # 10% max displacement
    """

    name = "elastic_deform"

    def __init__(
        self,
        grid_size: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.seed = seed

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply elastic deformation.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Unused (for API compatibility)
            level: Maximum displacement as fraction of image size

        Returns:
            Deformed images
        """
        if level is None or level == 0.0:
            return imgs

        imgs, was_3d = self._ensure_4d(imgs)
        n, c, h, w = imgs.shape

        # Create random displacement field on coarse grid
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=imgs.device)
            generator.manual_seed(self.seed)

        # Random displacements on coarse grid
        coarse_dx = torch.randn(
            n, 1, self.grid_size, self.grid_size,
            device=imgs.device,
            dtype=imgs.dtype,
            generator=generator,
        ) * level

        coarse_dy = torch.randn(
            n, 1, self.grid_size, self.grid_size,
            device=imgs.device,
            dtype=imgs.dtype,
            generator=generator,
        ) * level

        # Upsample to image size (smooth interpolation)
        dx = F.interpolate(coarse_dx, size=(h, w), mode="bilinear", align_corners=False)
        dy = F.interpolate(coarse_dy, size=(h, w), mode="bilinear", align_corners=False)

        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=imgs.device),
            torch.linspace(-1, 1, w, device=imgs.device),
            indexing='ij',
        )

        # Add displacements
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(n, 1, h, w) + dx
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(n, 1, h, w) + dy

        # Combine into sampling grid (N, H, W, 2)
        grid = torch.cat([grid_x, grid_y], dim=1)  # (N, 2, H, W)
        grid = grid.permute(0, 2, 3, 1)  # (N, H, W, 2)

        # Sample with deformed grid
        output = F.grid_sample(
            imgs,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"ElasticDeform(grid_size={self.grid_size}, seed={self.seed})"
