"""
Normalization handling for VisProbe robustness testing.

This module provides centralized normalization/denormalization for perturbation
strategies. The key insight is that perturbations should be applied in [0,1]
pixel space, but models expect normalized inputs.

Workflow:
    normalized_input → denormalize → apply_perturbation → renormalize → model

Presets:
    - 'imagenet': Standard ImageNet normalization (ResNet, VGG, etc.)
    - 'cifar10': CIFAR-10 normalization
    - 'cifar100': CIFAR-100 normalization
    - 'mnist': MNIST/grayscale normalization
    - None: No normalization (data already in [0,1])

Usage:
    >>> handler = NormalizationHandler.from_preset('imagenet')
    >>> pixel_img = handler.denormalize(normalized_img)
    >>> perturbed_pixel = apply_perturbation(pixel_img)
    >>> normalized_perturbed = handler.normalize(perturbed_pixel)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# =============================================================================
# Normalization Presets
# =============================================================================

NORMALIZATION_PRESETS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    # ImageNet: Used by ResNet, VGG, EfficientNet, ViT, etc.
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    # CIFAR-10
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    # CIFAR-100
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    # MNIST / Grayscale
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    # Fashion-MNIST
    "fashion_mnist": {
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    # Standard [0,1] to [-1,1] normalization
    "standard": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
}

# Aliases for common names
NORMALIZATION_PRESETS["imagenet1k"] = NORMALIZATION_PRESETS["imagenet"]
NORMALIZATION_PRESETS["resnet"] = NORMALIZATION_PRESETS["imagenet"]
NORMALIZATION_PRESETS["vgg"] = NORMALIZATION_PRESETS["imagenet"]


@dataclass
class NormalizationStats:
    """Container for normalization statistics."""

    mean: Tuple[float, ...]
    std: Tuple[float, ...]

    @property
    def num_channels(self) -> int:
        """Number of channels these stats apply to."""
        return len(self.mean)

    def to_tensors(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors shaped for broadcasting (1, C, 1, 1)."""
        mean_t = torch.tensor(self.mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(self.std, device=device, dtype=dtype).view(1, -1, 1, 1)
        return mean_t, std_t


class NormalizationHandler:
    """
    Handles normalization/denormalization for robustness testing.

    This class centralizes the logic for converting between:
    - Normalized space (what models expect)
    - Pixel space [0,1] (where perturbations are applied)

    The handler is immutable after creation and caches tensors per device
    for efficient repeated operations.

    Args:
        mean: Channel means for normalization
        std: Channel standard deviations for normalization

    Example:
        >>> handler = NormalizationHandler.from_preset('imagenet')
        >>>
        >>> # Denormalize for perturbation
        >>> pixel_img = handler.denormalize(normalized_img)
        >>> perturbed = pixel_img + noise  # Apply perturbation in [0,1]
        >>> perturbed = torch.clamp(perturbed, 0, 1)
        >>>
        >>> # Renormalize for model
        >>> model_input = handler.normalize(perturbed)
        >>> output = model(model_input)
    """

    def __init__(
        self,
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
    ) -> None:
        if len(mean) != len(std):
            raise ValueError(
                f"mean and std must have same length: {len(mean)} vs {len(std)}"
            )
        if any(s <= 0 for s in std):
            raise ValueError("std values must be positive")

        self._stats = NormalizationStats(mean=mean, std=std)
        self._tensor_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def from_preset(cls, preset: str) -> "NormalizationHandler":
        """
        Create handler from a preset name.

        Args:
            preset: One of 'imagenet', 'cifar10', 'cifar100', 'mnist', 'standard'

        Returns:
            NormalizationHandler configured with preset stats

        Raises:
            ValueError: If preset name is not recognized
        """
        preset_lower = preset.lower().replace("-", "_").replace(" ", "_")
        if preset_lower not in NORMALIZATION_PRESETS:
            available = ", ".join(sorted(NORMALIZATION_PRESETS.keys()))
            raise ValueError(
                f"Unknown normalization preset: '{preset}'. "
                f"Available: {available}"
            )
        config = NORMALIZATION_PRESETS[preset_lower]
        return cls(mean=config["mean"], std=config["std"])

    @classmethod
    def from_config(
        cls,
        config: Union[str, Dict[str, Any], "NormalizationHandler", None],
    ) -> Optional["NormalizationHandler"]:
        """
        Create handler from various input formats.

        Args:
            config: One of:
                - str: Preset name (e.g., 'imagenet')
                - dict: {'mean': [...], 'std': [...]}
                - NormalizationHandler: Returns as-is
                - None: Returns None (no normalization)

        Returns:
            NormalizationHandler or None

        Example:
            >>> handler = NormalizationHandler.from_config('imagenet')
            >>> handler = NormalizationHandler.from_config({'mean': [0.5], 'std': [0.5]})
            >>> handler = NormalizationHandler.from_config(None)  # Returns None
        """
        if config is None:
            return None

        if isinstance(config, cls):
            return config

        if isinstance(config, str):
            return cls.from_preset(config)

        if isinstance(config, dict):
            mean = config.get("mean")
            std = config.get("std")
            if mean is None or std is None:
                raise ValueError(
                    "Dict config must have 'mean' and 'std' keys. "
                    f"Got: {list(config.keys())}"
                )
            return cls(mean=tuple(mean), std=tuple(std))

        raise TypeError(
            f"Invalid normalization config type: {type(config)}. "
            "Expected str, dict, NormalizationHandler, or None."
        )

    @classmethod
    def detect_from_data(
        cls,
        sample: torch.Tensor,
        tolerance: float = 0.15,
    ) -> Optional["NormalizationHandler"]:
        """
        Attempt to detect normalization from data statistics.

        Compares sample statistics against known presets and returns
        the best match if within tolerance.

        Args:
            sample: Sample tensor (C,H,W) or (N,C,H,W)
            tolerance: Maximum deviation from preset to accept match

        Returns:
            NormalizationHandler if a preset matches, None otherwise

        Note:
            This is a heuristic and may not always be accurate.
            Prefer explicit configuration when possible.
        """
        if sample.dim() == 3:
            sample = sample.unsqueeze(0)

        # Compute per-channel statistics
        sample_mean = sample.mean(dim=[0, 2, 3]).tolist()
        sample_std = sample.std(dim=[0, 2, 3]).tolist()

        best_match: Optional[str] = None
        best_distance = float("inf")

        for preset_name, preset_stats in NORMALIZATION_PRESETS.items():
            # Skip aliases
            if preset_name in ("imagenet1k", "resnet", "vgg"):
                continue

            preset_mean = preset_stats["mean"]
            preset_std = preset_stats["std"]

            # Check channel count matches
            if len(preset_mean) != len(sample_mean):
                continue

            # Compute distance
            # For normalized data, mean should be ~0 and std ~1
            # We compute what the data stats would be if normalized with this preset
            expected_mean = [
                (m - pm) / ps for m, pm, ps in zip(sample_mean, preset_mean, preset_std)
            ]
            expected_std = [s / ps for s, ps in zip(sample_std, preset_std)]

            # Distance from expected normalized stats (mean≈0, std≈1)
            mean_dist = sum(abs(m) for m in expected_mean) / len(expected_mean)
            std_dist = sum(abs(s - 1.0) for s in expected_std) / len(expected_std)
            distance = mean_dist + std_dist

            if distance < best_distance:
                best_distance = distance
                best_match = preset_name

        if best_match is not None and best_distance < tolerance:
            return cls.from_preset(best_match)

        return None

    def _get_tensors(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached mean/std tensors for device."""
        if device not in self._tensor_cache:
            self._tensor_cache[device] = self._stats.to_tensors(device, dtype)
        return self._tensor_cache[device]

    @property
    def mean(self) -> Tuple[float, ...]:
        """Channel means."""
        return self._stats.mean

    @property
    def std(self) -> Tuple[float, ...]:
        """Channel standard deviations."""
        return self._stats.std

    @property
    def num_channels(self) -> int:
        """Number of channels."""
        return self._stats.num_channels

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize images from [0,1] pixel space to model space.

        Formula: normalized = (pixel - mean) / std

        Args:
            images: Images in [0,1] range, shape (N,C,H,W) or (C,H,W)

        Returns:
            Normalized images ready for model input
        """
        squeeze_back = images.dim() == 3
        if squeeze_back:
            images = images.unsqueeze(0)

        mean, std = self._get_tensors(images.device, images.dtype)
        normalized = (images - mean) / std

        return normalized.squeeze(0) if squeeze_back else normalized

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Denormalize images from model space to [0,1] pixel space.

        Formula: pixel = normalized * std + mean

        Args:
            images: Normalized images from model preprocessing

        Returns:
            Images in [0,1] pixel space for perturbation
        """
        squeeze_back = images.dim() == 3
        if squeeze_back:
            images = images.unsqueeze(0)

        mean, std = self._get_tensors(images.device, images.dtype)
        pixel = images * std + mean

        return pixel.squeeze(0) if squeeze_back else pixel

    def denormalize_perturb_normalize(
        self,
        images: torch.Tensor,
        perturbation_fn: Any,
        model: Any = None,
        level: Optional[float] = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        """
        Complete workflow: denormalize → perturb → renormalize.

        This is the recommended way to apply perturbations to normalized data.

        Args:
            images: Normalized input images
            perturbation_fn: Strategy or callable that applies perturbation
            model: Model (passed to perturbation_fn if needed)
            level: Perturbation level (passed to perturbation_fn if supported)
            clamp: Whether to clamp to [0,1] after perturbation (default True)

        Returns:
            Perturbed images, renormalized for model input
        """
        squeeze_back = images.dim() == 3
        if squeeze_back:
            images = images.unsqueeze(0)

        # Denormalize to pixel space
        pixel_images = self.denormalize(images)

        # Apply perturbation
        if hasattr(perturbation_fn, "generate"):
            perturbed = perturbation_fn.generate(pixel_images, model, level)
        elif callable(perturbation_fn):
            perturbed = perturbation_fn(pixel_images)
        else:
            raise TypeError(
                f"perturbation_fn must be a Strategy or callable, got {type(perturbation_fn)}"
            )

        # Clamp to valid pixel range
        if clamp:
            perturbed = torch.clamp(perturbed, 0.0, 1.0)

        # Renormalize for model
        result = self.normalize(perturbed)

        return result.squeeze(0) if squeeze_back else result

    def __repr__(self) -> str:
        return f"NormalizationHandler(mean={self.mean}, std={self.std})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormalizationHandler):
            return False
        return self._stats == other._stats


# =============================================================================
# Convenience Functions
# =============================================================================

def get_preset_names() -> List[str]:
    """Get list of available normalization preset names."""
    # Filter out aliases
    return sorted(
        name for name in NORMALIZATION_PRESETS.keys()
        if name not in ("imagenet1k", "resnet", "vgg")
    )


def get_preset_stats(preset: str) -> Dict[str, Tuple[float, ...]]:
    """Get mean/std for a preset."""
    preset_lower = preset.lower().replace("-", "_")
    if preset_lower not in NORMALIZATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    return NORMALIZATION_PRESETS[preset_lower].copy()
