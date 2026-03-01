"""
The `visprobe.properties` module provides a collection of functions and
classes for asserting the robustness of a model's behavior.
"""

from .base import CompositeProperty, Property
from .classification import ConfidenceDrop, L2Distance, LabelConstant, TopKStability
from .helpers import get_top_prediction, get_topk_predictions

__all__ = [
    # Base classes
    "Property",
    "CompositeProperty",
    # Classification properties (class-based)
    "LabelConstant",
    "TopKStability",
    "ConfidenceDrop",
    "L2Distance",
    # Helper functions
    "get_top_prediction",
    "get_topk_predictions",
]
