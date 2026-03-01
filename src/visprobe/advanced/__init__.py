"""
Advanced decorator-based API for VisProbe (for power users).

See README.md for migration guide.

The decorator API remains available for power users who need fine-grained control.
"""

import warnings


# Import decorator API from main package
from ..api import (
    ImageData,
    PerturbationInfo,
    Report,
    data_source,
    given,
    model,
    search,
)

__all__ = [
    "given",
    "model",
    "data_source",
    "search",
    "Report",
    "ImageData",
    "PerturbationInfo",
]
