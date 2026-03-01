"""
VisProbe's high-level user-facing API decorators.
"""

from __future__ import annotations

import inspect
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..strategies.base import Strategy
from .config import (
    DEFAULT_NOISE_SWEEP,
    DEFAULT_RESOLUTIONS,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from .registry import TestRegistry
from .runner import TestRunner


# --- Public API Decorators ---
class VisProbeError(Exception):
    """Base exception for VisProbe decorator errors."""


class ValidationError(VisProbeError):
    """Exception raised when decorator parameters fail validation."""


def _validate_test_function(func: Callable, decorator_name: str) -> None:
    """
    Validate that the test function has the correct signature.

    Args:
        func: The function to validate
        decorator_name: Name of the decorator (for error messages)

    Raises:
        ValidationError: If the function signature is invalid
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < 2:
        raise ValidationError(
            f"Test function '{func.__name__}' decorated with @{decorator_name} "
            f"must accept at least 2 parameters (original, perturbed), "
            f"but found {len(params)} parameter(s)"
        )


def _validate_search_params(  # noqa: C901
    initial_level: float,
    step: float,
    min_step: float,
    max_queries: int,
    mode: str,
    level_lo: Optional[float],
    level_hi: Optional[float],
) -> None:
    """
    Validate search decorator parameters.

    Raises:
        ValidationError: If any parameter is invalid
    """
    # Validate numeric parameters
    if initial_level < 0:
        raise ValidationError(f"initial_level must be >= 0, got {initial_level}")

    if step <= 0:
        raise ValidationError(f"step must be > 0, got {step}")

    if min_step <= 0:
        raise ValidationError(f"min_step must be > 0, got {min_step}")

    if min_step > step:
        raise ValidationError(f"min_step ({min_step}) must be <= step ({step})")

    if max_queries < 1:
        raise ValidationError(f"max_queries must be >= 1, got {max_queries}")

    # Validate mode
    valid_modes = {"adaptive", "binary", "grid", "random"}
    if mode not in valid_modes:
        raise ValidationError(f"mode must be one of {valid_modes}, got '{mode}'")

    # Validate bounds
    if level_lo is not None and level_lo < 0:
        raise ValidationError(f"level_lo must be >= 0, got {level_lo}")

    if level_hi is not None and level_hi <= 0:
        raise ValidationError(f"level_hi must be > 0, got {level_hi}")

    if level_lo is not None and level_hi is not None:
        if level_lo >= level_hi:
            raise ValidationError(f"level_lo ({level_lo}) must be < level_hi ({level_hi})")

    # Warn about mode-specific parameter requirements
    if mode in {"binary", "grid", "random"} and (level_lo is None or level_hi is None):
        warnings.warn(
            f"Mode '{mode}' works best with explicit level_lo and level_hi bounds. "
            f"Consider setting these parameters for optimal performance.",
            UserWarning,
            stacklevel=3,
        )


def model(model_obj: Any, *, capture_intermediate_layers: Optional[List[str]] = None):
    """
    Attaches a model to a VisProbe test.

    Args:
        model_obj: The PyTorch model to test
        capture_intermediate_layers: Optional list of layer names to capture during forward pass

    Returns:
        Decorator function that attaches model to the test function

    Example:
        @model(my_resnet, capture_intermediate_layers=["layer4"])
        @given(strategy=GaussianNoiseStrategy(std=0.05))
        def test_robustness(original, perturbed):
            assert original["output"].argmax() == perturbed["output"].argmax()
    """

    def decorator(func: Callable) -> Callable:
        func._visprobe_model = model_obj
        func._visprobe_capture_intermediate_layers = capture_intermediate_layers
        return func

    return decorator


def data_source(
    data_obj: Any,
    *,
    collate_fn: Optional[Callable[[Any], Any]] = None,
    class_names: Optional[List[str]] = None,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
):
    """
    Provides the data source for a VisProbe test.

    Args:
        data_obj: Data source (tensor, dataset, or any object)
        collate_fn: Optional function to collate data into batches
        class_names: Optional list of class names for visualization
        mean: Channel means for denormalization (defaults to ImageNet means)
        std: Channel stds for denormalization (defaults to ImageNet stds)

    Returns:
        Decorator function that attaches data configuration to the test function

    Example:
        @data_source(test_images, class_names=["cat", "dog"], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        @model(my_model)
        @given(strategy=GaussianNoiseStrategy(std=0.05))
        def test_robustness(original, perturbed):
            assert original["output"].argmax() == perturbed["output"].argmax()
    """

    def decorator(func: Callable) -> Callable:
        func._visprobe_data = data_obj
        func._visprobe_collate = collate_fn  # may be None; runner handles identity
        func._visprobe_class_names = class_names
        func._visprobe_mean = mean if mean is not None else IMAGENET_DEFAULT_MEAN
        func._visprobe_std = std if std is not None else IMAGENET_DEFAULT_STD
        return func

    return decorator


def given(
    *,
    strategy: Strategy,
    vectorized: bool = False,
    noise_sweep: Optional[Dict[str, Any]] = DEFAULT_NOISE_SWEEP,
    resolutions: Optional[List[Tuple[int, int]]] = DEFAULT_RESOLUTIONS,
    top_k: Optional[int] = 5,
    property_name: Optional[str] = None,
):
    """
    Defines a test with a fixed perturbation.
    Users write assertions in the body:
        @given(strategy=GaussianNoiseStrategy(std=0.05))
        def test_margin_guard(original, perturbed):
            # original["output"] -> (logits, features|None)
            # perturbed["output"] -> (logits, features|None)
            assert ...
    """

    def decorator(user_func: Callable) -> Callable:
        # Validate test function signature at decoration time
        _validate_test_function(user_func, "given")

        @wraps(user_func)
        def wrapper(*args, **wrapper_kwargs):
            # Attach optional display name for property
            if property_name is not None:
                user_func._visprobe_property_name = property_name
            runner = TestRunner(
                user_func,
                "given",
                {
                    "strategy": strategy,
                    "vectorized": vectorized,
                    "noise_sweep": noise_sweep,
                    "resolutions": resolutions,
                    "top_k": top_k,
                },
            )
            return runner.run()

        # register at decoration time for discovery
        TestRegistry.register_given(wrapper)
        return wrapper

    return decorator


def search(
    *,
    strategy: Callable[[float], Strategy] | Strategy,
    initial_level: float = 0.001,
    step: float = 0.002,
    min_step: float = 1e-5,
    max_queries: int = 500,
    mode: str = "adaptive",
    level_lo: Optional[float] = None,
    level_hi: Optional[float] = None,
    resolutions: Optional[List[Tuple[int, int]]] = DEFAULT_RESOLUTIONS,
    noise_sweep: Optional[Dict[str, Any]] = DEFAULT_NOISE_SWEEP,
    top_k: Optional[int] = 5,
    reduce: Optional[str] = "all",
    property_name: Optional[str] = None,
):
    """
    Defines a search for a model's failure point.

    Args:
        strategy: Perturbation strategy or factory function
        initial_level: Starting level for search
        step: Step size for adaptive search
        min_step: Minimum step size before stopping
        max_queries: Maximum model queries allowed
        mode: Search mode - 'adaptive' (default), 'binary', 'grid', or 'random'
        level_lo: Lower bound for search (used by binary/grid/random)
        level_hi: Upper bound for search (used by binary/grid/random)
        resolutions: Resolutions to test for analysis
        noise_sweep: Parameters for noise sensitivity analysis
        top_k: Number of top predictions to analyze
        reduce: Aggregation method ('all', 'any', or 'frac>=X')
        property_name: Display name for the robustness property

    Returns:
        Decorated test function

    Example:
        @search(strategy=lambda l: FGSMStrategy(eps=l),
                initial_level=0.001,
                mode='binary',  # Use binary search for efficiency
                level_lo=0.0,
                level_hi=0.1)
        def test_fgsm_threshold(original, perturbed):
            return LabelConstant.evaluate(original, perturbed)
    """

    def decorator(user_func: Callable) -> Callable:
        # Validate test function signature at decoration time
        _validate_test_function(user_func, "search")

        # Validate search parameters
        _validate_search_params(
            initial_level=initial_level,
            step=step,
            min_step=min_step,
            max_queries=max_queries,
            mode=mode,
            level_lo=level_lo,
            level_hi=level_hi,
        )

        @wraps(user_func)
        def wrapper(*args, **wrapper_kwargs):
            if property_name is not None:
                user_func._visprobe_property_name = property_name
            params = {
                "strategy": strategy,
                "initial_level": initial_level,
                "step": step,
                "min_step": min_step,
                "max_queries": max_queries,
                "mode": mode,
                "level_lo": level_lo,
                "level_hi": level_hi,
                "resolutions": resolutions,
                "noise_sweep": noise_sweep,
                "top_k": top_k,
                "reduce": reduce,
            }
            runner = TestRunner(user_func, "search", params)
            return runner.run()

        # register at decoration time for discovery
        TestRegistry.register_search(wrapper)
        return wrapper

    return decorator
