"""
Advanced API for VisProbe (for power users).

This module provides advanced access to internal components for power users
who need fine-grained control over the testing process.
"""

# Re-export the main search function and Report
from ..api import search
from ..report import Report

# Re-export strategy and property classes for advanced usage
from ..strategies import *
from ..properties import *
from ..core.search_engine import SearchEngine
from ..core.search_strategies import AdaptiveSearchStrategy, BinarySearchStrategy

__all__ = [
    "search",
    "Report",
    "SearchEngine",
    "AdaptiveSearchStrategy",
    "BinarySearchStrategy",
]
