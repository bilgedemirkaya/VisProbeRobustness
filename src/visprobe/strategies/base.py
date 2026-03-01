"""
Base Strategy class for all perturbation methods in VisProbe.

Provides a unified interface for both model-aware (adversarial) and
model-agnostic (natural) perturbations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch


class Strategy:
    """
    Base class for all perturbation strategies in VisProbe.

    A strategy defines how to modify an input to test a model's robustness.
    Supports:
    - Batched operations (process multiple images at once)
    - GPU acceleration (computations stay on device)
    - Level-based intensity control
    - Both model-aware and model-agnostic perturbations
    """

    name: str = "base"

    def __init__(self) -> None:
        self._device: Optional[torch.device] = None

    @classmethod
    def resolve(cls, perturb_spec: Any, *, level: Optional[float] = None) -> "Strategy":
        """
        Resolves a perturbation specification into a valid Strategy instance.

        Accepted forms:
        - Strategy instance (or any object with a `generate(imgs, model)` method)
        - Dict spec, e.g., {"type": "gaussian_noise", "std": 0.1}
        - Callable fn(level, imgs, model) -> imgs for ad-hoc perturbations
        - Sequence of any of the above -> compose sequentially
        """
        # 0) Sequence -> compose
        if isinstance(perturb_spec, (list, tuple)):
            strategies: List[Strategy] = [cls.resolve(s, level=level) for s in perturb_spec]
            return CompositeStrategy(strategies)

        # 1) Already a Strategy-like object
        if isinstance(perturb_spec, cls) or callable(getattr(perturb_spec, "generate", None)):
            return perturb_spec

        # 2) Dict-based specification
        if isinstance(perturb_spec, dict):
            if "type" not in perturb_spec:
                raise ValueError("Dict spec must include a 'type' field.")
            spec_type = perturb_spec["type"]
            params: Dict[str, Any] = {k: v for k, v in perturb_spec.items() if k != "type"}

            # Lazy import to avoid circular dependencies
            if spec_type in {"gaussian_noise", "brightness", "contrast", "rotate", "gamma",
                            "gaussian_blur", "motion_blur", "jpeg_compression"}:
                from .image import (
                    BrightnessStrategy,
                    ContrastStrategy,
                    GammaStrategy,
                    GaussianBlurStrategy,
                    GaussianNoiseStrategy,
                    JPEGCompressionStrategy,
                    MotionBlurStrategy,
                    RotateStrategy,
                )

                mapping: Dict[str, Callable[..., Strategy]] = {
                    "gaussian_noise": GaussianNoiseStrategy,
                    "brightness": BrightnessStrategy,
                    "contrast": ContrastStrategy,
                    "rotate": RotateStrategy,
                    "gamma": GammaStrategy,
                    "gaussian_blur": GaussianBlurStrategy,
                    "motion_blur": MotionBlurStrategy,
                    "jpeg_compression": JPEGCompressionStrategy,
                }
                return mapping[spec_type](**params)

            if spec_type in {"fgsm", "pgd", "bim", "apgd", "square"}:
                from .adversarial import (
                    APGDStrategy,
                    BIMStrategy,
                    FGSMStrategy,
                    PGDStrategy,
                    SquareAttackStrategy,
                )

                mapping: Dict[str, Callable[..., Strategy]] = {
                    "fgsm": FGSMStrategy,
                    "pgd": PGDStrategy,
                    "bim": BIMStrategy,
                    "apgd": APGDStrategy,
                    "square": SquareAttackStrategy,
                }
                return mapping[spec_type](**params)

            raise ValueError(f"Unknown strategy type in dict spec: {spec_type}")

        # 3) Ad-hoc callable: fn(level, imgs, model) -> imgs
        if callable(perturb_spec):
            return _CallableStrategy(perturb_spec, level=level)

        raise ValueError(f"Unknown perturbation specification: {perturb_spec}.")

    def generate(
        self, imgs: torch.Tensor, model: Any = None, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generates a perturbed version of the input images.

        This is the primary method for applying perturbations. Model-agnostic
        strategies can ignore the model parameter.

        Args:
            imgs: Input images to perturb (N, C, H, W) or (C, H, W)
            model: The model being tested (optional for non-adversarial)
            level: Perturbation intensity level

        Returns:
            Perturbed images
        """
        raise NotImplementedError

    def apply(
        self, imgs: torch.Tensor, model: Any = None, level: Optional[float] = None
    ) -> torch.Tensor:
        """Alias for generate() for backward compatibility."""
        return self.generate(imgs, model, level)

    def __call__(
        self, imgs: torch.Tensor, level: Optional[float] = None, **kwargs: Any
    ) -> torch.Tensor:
        """Shorthand for generate() without model parameter."""
        return self.generate(imgs, model=None, level=level)

    def query_cost(self) -> int:
        """
        Returns the number of additional model queries used by the strategy.

        Returns:
            Number of extra model forward passes required (0 for non-adversarial)
        """
        return 0

    def to(self, device: Union[str, torch.device]) -> "Strategy":
        """Move strategy computations to specified device."""
        self._device = torch.device(device) if isinstance(device, str) else device
        return self

    @property
    def device(self) -> Optional[torch.device]:
        """Get current device."""
        return self._device

    def _ensure_4d(self, images: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Ensure images are 4D tensor (N, C, H, W).

        Returns:
            Tuple of (4D tensor, was_3d_flag)
        """
        if images.dim() == 3:
            return images.unsqueeze(0), True
        elif images.dim() == 4:
            return images, False
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {images.dim()}D")

    def _restore_dims(self, images: torch.Tensor, was_3d: bool) -> torch.Tensor:
        """Restore original dimensions if needed."""
        return images.squeeze(0) if was_3d else images

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        return self.__str__()


class _CallableStrategy(Strategy):
    """Adapter for user-provided functions: fn(level, imgs, model) -> imgs."""

    def __init__(
        self, fn: Callable[[Optional[float], Any, Any], Any], *, level: Optional[float] = None
    ):
        super().__init__()
        self._fn = fn
        self.level = level

    def generate(
        self, imgs: torch.Tensor, model: Any = None, level: Optional[float] = None
    ) -> torch.Tensor:
        """Apply the callable perturbation function."""
        return self._fn(level or self.level, imgs, model)


class CompositeStrategy(Strategy):
    """Composes multiple strategies sequentially: s_n(...s_2(s_1(x)))."""

    def __init__(self, strategies: Sequence[Strategy]):
        super().__init__()
        self.strategies: List[Strategy] = list(strategies)

    def generate(
        self, imgs: torch.Tensor, model: Any = None, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply all strategies sequentially.

        Args:
            imgs: Input images
            model: The model being tested
            level: Optional perturbation level passed to all strategies

        Returns:
            Images after all perturbations applied
        """
        out = imgs
        for strat in self.strategies:
            try:
                out = strat.generate(out, model=model, level=level)
            except TypeError:
                out = strat.generate(out, model=model)
        return out

    def query_cost(self) -> int:
        """Sum of inner strategy query costs."""
        total = 0
        for strat in self.strategies:
            try:
                total += int(strat.query_cost())
            except Exception:
                pass
        return total
