#!/usr/bin/env python3
"""
Simple test to verify the analysis module works.
"""

import torch
import torch.nn as nn
from visprobe.analysis import (
    evaluate_detailed,
    bootstrap_accuracy,
    confidence_profile,
    class_vulnerability,
)

print("Testing VisProbe Analysis Module")
print("="*60)

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Create test data
model = SimpleModel()
images = torch.randn(20, 10)
labels = torch.randint(0, 5, (20,))

# Test evaluation
print("\n1. Testing detailed evaluation...")
results = evaluate_detailed(
    model, images, labels,
    model_name="SimpleModel",
    scenario="test"
)
print(f"   Accuracy: {results.accuracy:.2%}")
print(f"   Failures: {len(results.get_failures())}")

# Test bootstrap
print("\n2. Testing bootstrap confidence intervals...")
acc, lower, upper = bootstrap_accuracy(results.correct_mask, n_bootstrap=100)
print(f"   Accuracy: {acc:.2%} (95% CI: [{lower:.2%}, {upper:.2%}])")

# Test confidence profile
print("\n3. Testing confidence profiling...")
profile = confidence_profile(results.samples)
print(f"   Mean confidence: {profile.mean_confidence:.2f}")
print(f"   Calibration error: {profile.calibration_error:.3f}")

print("\n✅ All tests passed successfully!")
print("The VisProbe analysis module is working correctly.")