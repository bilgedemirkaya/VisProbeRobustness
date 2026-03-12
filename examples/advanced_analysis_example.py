#!/usr/bin/env python3
"""
Advanced Analysis Example for VisProbe

This example demonstrates all capabilities of the analysis module.
"""

import torch
import torchvision.models as models
import numpy as np

# Import VisProbe analysis module
from visprobe.analysis import (
    evaluate_detailed,
    bootstrap_accuracy,
    bootstrap_delta,
    find_crossover,
    disagreement_analysis,
    confidence_profile,
    class_vulnerability,
    systematic_failures,
)

print("VisProbe Advanced Analysis Example")
print("See the full example in the documentation")
print("This module provides:")
print("- Detailed per-sample tracking")
print("- Bootstrap confidence intervals")
print("- Crossover detection")
print("- Disagreement analysis")
print("- Confidence profiling")
print("- Vulnerability analysis")
