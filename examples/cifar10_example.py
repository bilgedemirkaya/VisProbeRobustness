#!/usr/bin/env python3
"""
CIFAR-10 Robustness Testing Example

This example shows how to test a model trained on CIFAR-10.
It demonstrates:
- Loading CIFAR-10 dataset
- Testing with different presets
- Comparing results across presets
"""

import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from visprobe import search

# CIFAR-10 normalization stats
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

print("="*70)
print("CIFAR-10 Robustness Testing")
print("="*70)

# 1. Load a pretrained model (ResNet-18)
print("\n1. Loading model...")
model = models.resnet18(weights=None)  # You can load your trained weights here
# model.load_state_dict(torch.load('path/to/your/cifar10_weights.pth'))
model.eval()
print("   ✓ Model loaded")

# 2. Load CIFAR-10 test dataset
print("\n2. Loading CIFAR-10 dataset...")
transform = T.Compose([
    T.Resize(224),  # ResNet expects 224x224
    T.ToTensor(),
])

try:
    dataset = CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    # Take a subset for faster testing
    num_samples = 100
    test_data = [dataset[i] for i in range(num_samples)]
    print(f"   ✓ Loaded {num_samples} test samples")
except Exception as e:
    print(f"   ⚠️  Could not download CIFAR-10: {e}")
    print("   Creating dummy data instead...")
    test_images = torch.randn(100, 3, 224, 224)
    test_labels = torch.randint(0, 10, (100,))
    test_data = [(img, int(label.item())) for img, label in zip(test_images, test_labels)]

# 3. Test with "standard" preset
print("\n3. Running robustness test with 'standard' preset...")
print("   (This tests brightness, blur, noise, compression + compositional)")

report = search(
    model=model,
    data=test_data,
    preset="standard",
    budget=1000,
    device="auto",
    mean=CIFAR_MEAN,  # Important: Use CIFAR-10 normalization
    std=CIFAR_STD
)

# 4. Display results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
report.show()

# 5. Analyze results
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print(f"\nOverall Robustness Score: {report.score:.1%}")
print(f"  → {'✅ GOOD' if report.score > 0.7 else '⚠️  NEEDS IMPROVEMENT' if report.score > 0.4 else '❌ POOR'}")

print(f"\nTotal Failures Found: {len(report.failures)}")
if report.failures:
    print(f"  → {report.summary['unique_failed_samples']} unique samples failed")

    # Show which perturbations caused most failures
    print("\nFailure breakdown:")
    if report.search and 'results' in report.search:
        for result in report.search['results']:
            strategy = result.get('strategy', 'Unknown')
            num_failures = len(result.get('failures', []))
            score = result.get('robustness_score', 0)
            print(f"  • {strategy:30s}: {num_failures:2d} failures (score: {score:.1%})")

# 6. Export worst failures for analysis
if report.failures:
    print("\n6. Exporting worst failures...")
    export_path = report.export_failures(n=10)
    print(f"   ✓ Exported to: {export_path}")
    print(f"   → Use these for retraining or analysis")

# 7. Save full report
print("\n7. Report saved automatically to:")
print(f"   {report.summary['test_name']}.json")

print("\n" + "="*70)
print("✅ Testing complete!")
print("="*70)
print("\nNext steps:")
print("  1. Review failures in the exported directory")
print("  2. Try different presets: 'lighting', 'blur', 'corruption'")
print("  3. Use failures to improve your model training")
