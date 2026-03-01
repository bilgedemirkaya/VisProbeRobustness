"""
Strategy configuration and serialization utilities.

This module provides functions for configuring perturbation strategies
and serializing them to JSON-safe formats for reporting.
"""

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class StrategyConfig:
    """Handles strategy configuration and serialization."""

    @staticmethod
    def configure(strategy, mean, std, rng=None):
        """
        Configure strategy with normalization parameters and random seed.

        Args:
            strategy: Strategy object to configure
            mean: Normalization mean values
            std: Normalization std values
            rng: Optional random number generator

        Raises:
            AttributeError: If strategy doesn't support configuration
        """
        if hasattr(strategy, "configure"):
            # Modern strategy API
            strategy.configure(mean=mean, std=std, generator=rng)
        else:
            # Fallback for legacy strategies
            if hasattr(strategy, "mean"):
                strategy.mean = mean
            if hasattr(strategy, "std"):
                strategy.std = std
            if hasattr(strategy, "_rng") and rng is not None:
                strategy._rng = rng

    @staticmethod
    def serialize(strategy_obj) -> Tuple[str, Dict[str, Any]]:
        """
        Serialize strategy to JSON-safe format.

        Args:
            strategy_obj: Strategy object to serialize

        Returns:
            Tuple of (strategy_name, params_dict)
        """
        name = type(strategy_obj).__name__

        # Handle composite strategies (multiple strategies combined)
        if hasattr(strategy_obj, "strategies") and isinstance(
            getattr(strategy_obj, "strategies"), (list, tuple)
        ):
            components = []
            for s in getattr(strategy_obj, "strategies"):
                try:
                    cn, cp = StrategyConfig.serialize(s)
                    components.append({"name": cn, "params": cp})
                except Exception:
                    components.append({"name": type(s).__name__, "params": {}})
            return name, {"components": components}

        # Standard strategy: extract parameters
        raw = {}
        try:
            raw = dict(vars(strategy_obj))
        except Exception:
            raw = {}

        params = {k: StrategyConfig._to_safe_value(v) for k, v in raw.items() if not k.startswith("_")}
        return name, params

    @staticmethod
    def _to_safe_value(v):
        """Convert value to JSON-serializable format."""
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [StrategyConfig._to_safe_value(x) for x in v]
        if isinstance(v, dict):
            return {str(k): StrategyConfig._to_safe_value(val) for k, val in v.items()}

        # Handle PyTorch tensors
        try:
            import torch

            if isinstance(v, torch.Tensor):
                return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
        except Exception:
            pass

        # Handle callables
        if callable(v):
            return getattr(v, "__name__", "callable")

        # Fallback: type name
        return str(type(v).__name__)

    @staticmethod
    def infer_strength_units(strategy_obj) -> str:
        """
        Infer units for perturbation strength from strategy type.

        Args:
            strategy_obj: Strategy object

        Returns:
            String description of strength units
        """
        name = strategy_obj.__class__.__name__.lower()
        if "gaussiannoise" in name or "noise" in name:
            return "std of Gaussian"
        if "fgsm" in name or "pgd" in name:
            return "epsilon L∞"
        if "rotate" in name:
            return "degrees"
        if "brightness" in name:
            return "brightness factor"
        return "level"
