"""
Perturbation composition utilities.

Provides ways to combine multiple perturbations:
- Sequential composition (apply one after another)
- Parallel composition (apply all and blend)
- Random selection
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Union

import torch

from .base import Strategy

__all__ = [
    "Compose",
    "RandomChoice",
    "Blend",
    "LowLightBlur",
    "LevelTransform",
]


class LevelTransform(Strategy):
    """
    Wrapper strategy that transforms the level parameter before passing to wrapped strategy.

    This enables composing strategies with different level mappings from a global severity.

    Args:
        strategy: The strategy to wrap
        transform_fn: Function that maps input level to strategy-specific level
                     Signature: (float) -> float

    Example:
        >>> # Brightness that goes from 1.0 (s=0) to 0.3 (s=1)
        >>> brightness_transform = LevelTransform(
        ...     Brightness(),
        ...     transform_fn=lambda s: 0.3 + 0.7 * (1 - s)
        ... )
        >>>
        >>> # PGD that scales eps linearly
        >>> def pgd_factory(max_eps=0.01, max_iter=10):
        ...     def transform(s):
        ...         return s * max_eps
        ...     pgd = PGDStrategy(eps=max_eps, eps_step=max_eps/10, max_iter=max_iter)
        ...     return LevelTransform(pgd, transform)
        >>>
        >>> # Compose them
        >>> composed = Compose([
        ...     brightness_transform,
        ...     pgd_factory(max_eps=0.01, max_iter=10)
        ... ])
        >>> result = composed(images, model, level=0.5)
    """

    name = "level_transform"

    def __init__(
        self,
        strategy: Strategy,
        transform_fn: Callable[[Optional[float]], Optional[float]],
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.transform_fn = transform_fn

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply transformation to level, then generate perturbation.

        Args:
            imgs: Input images
            model: Model (passed through)
            level: Input severity level

        Returns:
            Perturbed images with transformed level
        """
        transformed_level = self.transform_fn(level) if level is not None else None
        return self.strategy.generate(imgs, model, transformed_level)

    def __repr__(self) -> str:
        return f"LevelTransform({self.strategy}, transform_fn={self.transform_fn})"


class Compose(Strategy):
    """
    Sequential composition of strategies.

    Applies strategies one after another in order.

    Args:
        strategies: List of strategies to compose
        level_mode: How to distribute level across strategies
            - 'same': Use same level for all
            - 'split': Divide level equally
            - 'first': Apply level to first only

    Example:
        >>> composed = Compose([LowLight(), GaussianBlur()])
        >>> result = composed(images, level=0.5)
    """

    name = "compose"

    def __init__(
        self,
        strategies: Sequence[Strategy],
        level_mode: str = "same",
    ) -> None:
        super().__init__()
        self.strategies = list(strategies)
        if level_mode not in ("same", "split", "first"):
            raise ValueError(f"level_mode must be 'same', 'split', or 'first', got {level_mode}")
        self.level_mode = level_mode

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply all strategies sequentially.

        Args:
            imgs: Input images
            model: Model (passed to each strategy)
            level: Perturbation intensity. If None, each strategy uses its own
                   baked-in parameters (recommended for pre-configured compositions).

        Returns:
            Images after all strategies applied
        """
        output = imgs

        for i, strat in enumerate(self.strategies):
            # Determine level for this strategy
            # If level is None, pass None to let each strategy use its baked-in params
            if level is None:
                strat_level = None
            elif self.level_mode == "same":
                strat_level = level
            elif self.level_mode == "split":
                strat_level = level / len(self.strategies)
            else:  # first
                strat_level = level if i == 0 else None

            output = strat.generate(output, model, strat_level)

        return output

    def __repr__(self) -> str:
        strat_names = [s.name for s in self.strategies]
        return f"Compose({strat_names}, level_mode={self.level_mode})"


class RandomChoice(Strategy):
    """
    Randomly select one strategy to apply.

    Args:
        strategies: List of strategies to choose from
        weights: Optional weights for selection probabilities
        seed: Random seed for reproducibility

    Example:
        >>> random_strat = RandomChoice([GaussianNoise(), GaussianBlur()])
        >>> result = random_strat(images, level=0.5)
    """

    name = "random_choice"

    def __init__(
        self,
        strategies: Sequence[Strategy],
        weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.strategies = list(strategies)
        self.weights = weights
        self.seed = seed
        self._generator: Optional[torch.Generator] = None

    def _get_generator(self, device: torch.device) -> Optional[torch.Generator]:
        """Get random generator."""
        if self.seed is None:
            return None
        if self._generator is None:
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
        Apply a randomly selected strategy.

        Args:
            imgs: Input images
            model: Model (passed to selected strategy)
            level: Perturbation intensity

        Returns:
            Perturbed images
        """
        # Select strategy
        if self.weights is not None:
            weights_tensor = torch.tensor(self.weights, device=imgs.device)
            idx = torch.multinomial(
                weights_tensor,
                1,
                generator=self._get_generator(imgs.device),
            ).item()
        else:
            gen = self._get_generator(imgs.device)
            if gen is not None:
                idx = torch.randint(0, len(self.strategies), (1,), generator=gen).item()
            else:
                idx = torch.randint(0, len(self.strategies), (1,)).item()

        return self.strategies[idx].generate(imgs, model, level)

    def __repr__(self) -> str:
        strat_names = [s.name for s in self.strategies]
        return f"RandomChoice({strat_names}, weights={self.weights})"


class Blend(Strategy):
    """
    Blend multiple strategies together.

    Applies all strategies and blends results.

    Args:
        strategies: List of strategies to blend
        blend_weights: Weights for blending (default: equal weights)

    Example:
        >>> blended = Blend([GaussianNoise(), GaussianBlur()], blend_weights=[0.7, 0.3])
        >>> result = blended(images, level=0.5)
    """

    name = "blend"

    def __init__(
        self,
        strategies: Sequence[Strategy],
        blend_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.strategies = list(strategies)

        if blend_weights is None:
            blend_weights = [1.0 / len(strategies)] * len(strategies)
        elif len(blend_weights) != len(strategies):
            raise ValueError("blend_weights must match number of strategies")

        # Normalize weights
        total = sum(blend_weights)
        self.blend_weights = [w / total for w in blend_weights]

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply all strategies and blend results.

        Args:
            imgs: Input images
            model: Model (passed to each strategy)
            level: Perturbation intensity

        Returns:
            Blended result of all strategies
        """
        result = torch.zeros_like(imgs)

        for strat, weight in zip(self.strategies, self.blend_weights):
            perturbed = strat.generate(imgs, model, level)
            result = result + weight * perturbed

        return result

    def __repr__(self) -> str:
        strat_names = [s.name for s in self.strategies]
        return f"Blend({strat_names}, weights={self.blend_weights})"


class LowLightBlur(Strategy):
    """
    Combined low-light and blur strategy.

    Simulates challenging visibility conditions: dim lighting + focus blur.
    This is a common real-world scenario (e.g., night driving, indoor dim lighting).

    Args:
        noise_factor: Noise intensity in low-light simulation
        blur_kernel_size: Fixed blur kernel size (or None for auto)

    Example:
        >>> low_light_blur = LowLightBlur()
        >>> result = low_light_blur(images, level=0.5)
    """

    name = "low_light_blur"

    def __init__(
        self,
        noise_factor: float = 0.3,
        blur_kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.noise_factor = noise_factor
        self.blur_kernel_size = blur_kernel_size

        # Lazy init components
        self._low_light: Optional[Strategy] = None
        self._blur: Optional[Strategy] = None

    def _ensure_components(self) -> None:
        """Lazy initialization of component strategies."""
        if self._low_light is None:
            from .lighting import LowLight
            from .blur import GaussianBlur

            self._low_light = LowLight(noise_factor=self.noise_factor)
            self._blur = GaussianBlur(kernel_size=self.blur_kernel_size)

    def generate(
        self,
        imgs: torch.Tensor,
        model: Any = None,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply combined low-light and blur.

        Args:
            imgs: Input images
            model: Unused (for API compatibility)
            level: Combined intensity (0-1)
                - Low light: brightness = 1 - level * 0.7
                - Blur: sigma = level * 3

        Returns:
            Images with low-light and blur applied
        """
        if level is None:
            level = 0.5

        self._ensure_components()

        # Split level between effects
        brightness = max(0.1, 1.0 - level * 0.7)  # 1.0 -> 0.3
        blur_sigma = level * 3.0  # 0 -> 3

        # Apply low light first (brightness reduction + noise)
        output = self._low_light.generate(imgs, model, brightness)

        # Then apply blur
        if blur_sigma > 0.1:
            output = self._blur.generate(output, model, blur_sigma)

        return output

    def __repr__(self) -> str:
        return f"LowLightBlur(noise_factor={self.noise_factor}, blur_kernel_size={self.blur_kernel_size})"
