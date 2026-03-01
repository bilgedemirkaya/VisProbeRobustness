#!/usr/bin/env python3
"""
Basic VisProbe Example - Minimal Working Example

This is the simplest possible use of VisProbe.
Tests a pretrained ResNet-18 model on random data.
"""

import torch
import torchvision.models as models
from visprobe import search

# 1. Load a model
print("Loading model...")
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# 2. Create some test data (10 random images)
print("Creating test data...")
test_images = torch.randn(10, 3, 224, 224)  # 10 images, 224x224
test_labels = torch.randint(0, 1000, (10,))  # Random ImageNet labels
test_data = [(img, int(label.item())) for img, label in zip(test_images, test_labels)]

# 3. Run robustness test
print("\nRunning robustness test...")
report = search(
    model=model,
    data=test_data,
    preset="lighting",  # Test lighting variations
    budget=100,         # Small budget for quick test
    device="auto"       # Auto-detect GPU/CPU
)

# 4. View results
print("\n" + "="*70)
report.show()
print("="*70)

# 5. Access results programmatically
print(f"\nRobustness Score: {report.score:.1%}")
print(f"Total Failures: {len(report.failures)}")
print(f"Runtime: {report.summary['runtime_sec']:.1f}s")

# 6. Export failures if any
if report.failures:
    export_path = report.export_failures(n=5)
    print(f"\n✅ Exported failures to: {export_path}")
