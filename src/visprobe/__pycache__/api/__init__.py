"""
This module exposes the primary user-facing API of VisProbe, including
decorators for defining tests and core data structures for reports.
"""

from .decorators import data_source, given, model, search
from .report import ImageData, PerturbationInfo, Report

__all__ = [
    "given",
    "model",
    "data_source",
    "search",
    "Report",
    "ImageData",
    "PerturbationInfo",
]
