"""
Test context initialization and device management.

This module handles the setup of the test environment, including device selection,
model/data placement, and configuration of test parameters.
"""

import logging
import os
from typing import Any, Callable, Dict

import torch

from .config import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, configure_threading, get_default_device
from .model_wrap import _ModelWithIntermediateOutput

logger = logging.getLogger(__name__)


class TestContext:
    """Manages test context initialization and device placement."""

    @staticmethod
    def build(user_func: Callable) -> Dict[str, Any]:
        """
        Build test context from decorated user function.

        Args:
            user_func: The user's test function with VisProbe decorators

        Returns:
            Dictionary containing model, data, device, and configuration
        """
        # Configure threading for stability
        try:
            configure_threading()
        except Exception as e:
            logger.warning(f"Failed to configure threading: {e}")

        # Extract model and data from decorators
        model_obj = getattr(user_func, "_visprobe_model")
        capture_layers = getattr(user_func, "_visprobe_capture_intermediate_layers", None)
        wrapped_model = (
            _ModelWithIntermediateOutput(model_obj, capture_layers) if capture_layers else model_obj
        )

        data_obj = getattr(user_func, "_visprobe_data")
        collate_fn = getattr(user_func, "_visprobe_collate", None)
        batch_tensor = collate_fn(data_obj) if callable(collate_fn) else data_obj

        # Device selection and placement
        device = get_default_device()
        wrapped_model = TestContext._move_model_to_device(wrapped_model, device)
        batch_tensor = TestContext._move_data_to_device(batch_tensor, device)

        # Random seed for reproducibility
        seed = TestContext._get_seed()
        rng = torch.Generator()
        try:
            rng.manual_seed(seed)
        except Exception as e:
            logger.warning(f"Could not set random seed {seed}: {e}")

        # Batch size detection
        batch_size = TestContext._detect_batch_size(batch_tensor)

        return {
            "model": wrapped_model,
            "batch_tensor": batch_tensor,
            "batch_size": batch_size,
            "class_names": getattr(user_func, "_visprobe_class_names", None),
            "mean": getattr(user_func, "_visprobe_mean", IMAGENET_DEFAULT_MEAN),
            "std": getattr(user_func, "_visprobe_std", IMAGENET_DEFAULT_STD),
            "capture_layers": capture_layers,
            "device": str(device),
            "rng": rng,
            "seed": seed,
        }

    @staticmethod
    def _move_model_to_device(model, device):
        """Move model to specified device with error handling."""
        try:
            if hasattr(model, "to"):
                return model.to(device)
            elif hasattr(model, "model") and hasattr(model.model, "to"):
                # Handle wrapped models
                model.model = model.model.to(device)
                return model
        except RuntimeError as e:
            logger.warning(f"Could not move model to {device}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error moving model to {device}: {e}")
        return model

    @staticmethod
    def _move_data_to_device(data, device):
        """Move data to specified device with error handling."""
        try:
            if hasattr(data, "to"):
                return data.to(device)
        except RuntimeError as e:
            logger.warning(f"Could not move data to {device}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error moving data to {device}: {e}")
        return data

    @staticmethod
    def _get_seed() -> int:
        """Extract seed from environment variables."""
        try:
            return int(os.environ.get("VISPROBE_SEED", os.environ.get("RQ2_SEED", "1337")))
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid seed value in environment, using default 1337: {e}")
            return 1337

    @staticmethod
    def _detect_batch_size(batch_tensor) -> int:
        """Detect batch size from tensor shape."""
        try:
            return int(getattr(batch_tensor, "shape", [1])[0])
        except (AttributeError, IndexError, ValueError) as e:
            logger.warning(f"Could not detect batch size, defaulting to 1: {e}")
            return 1
