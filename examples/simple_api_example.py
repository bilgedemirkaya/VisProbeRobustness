#!/usr/bin/env python3
"""
Example: Simplified Perturbation API

This demonstrates the new simple API that eliminates the need for:
- Manual strategy construction
- Parameter range selection
- Domain expertise about perturbation parameters

Compare:

OLD (Complex):
    report = search(
        model, data,
        strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
        level_lo=0.0,
        level_hi=0.15  # How did we know this?
    )

NEW (Simple):
    report = search(model, data, perturbation="gaussian_noise")

The library now handles:
- Reasonable parameter ranges for each perturbation type
- Dataset-specific ranges (ImageNet vs CIFAR-10)
- Strategy construction
"""

import torch
import torchvision
import torchvision.transforms as transforms
from visprobe import search, list_perturbations, Perturbation


def main():
    print("=" * 70)
    print("VisProbe - Simplified Perturbation API Demo")
    print("=" * 70)

    # 1. Setup
    print("\n1. Loading model and data...")
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_data = [(testset[i][0], testset[i][1]) for i in range(50)]

    print(f"   Model: {model.__class__.__name__}")
    print(f"   Data: {len(test_data)} CIFAR-10 samples")

    # 2. List available perturbations
    print("\n2. Available perturbations:")
    perturbations = list_perturbations()
    for name, desc in list(perturbations.items())[:5]:  # Show first 5
        print(f"   - {name}: {desc[:60]}...")
    print(f"   ... and {len(perturbations) - 5} more")

    # 3. Test with simple API (using named constants)
    print("\n3. Testing with simple API...")
    print("=" * 70)

    print("\nNote: You can use either strings or named constants:")
    print("  - Strings: search(model, data, perturbation='gaussian_noise')")
    print("  - Named constants: search(model, data, perturbation=Perturbation.GAUSSIAN_NOISE)")
    print("  Named constants provide IDE autocomplete and type safety!\n")

    # Using named constants for IDE autocomplete
    test_perturbations = [
        Perturbation.GAUSSIAN_NOISE,
        Perturbation.GAUSSIAN_BLUR,
        Perturbation.BRIGHTNESS_DECREASE,
    ]

    for pert in test_perturbations:
        print(f"\n{pert}:")
        print("   Command: search(model, data, perturbation='{}')")
        print("   Library automatically:")
        print("   - Selects appropriate parameter ranges")
        print("   - Constructs strategy factory")
        print("   - Handles normalization workflow")

        report = search(
            model=model,
            data=test_data,
            perturbation=pert,
            max_queries=10,
            device="cpu",
            verbose=False,
        )

        print(f"\n   Results:")
        print(f"   - Failure threshold: {report.metrics['failure_threshold']:.4f}")
        print(f"   - Robustness score: {report.metrics['overall_robustness_score']:.1%}")
        print(f"   - Queries used: {report.metrics.get('queries', 'N/A')}")

    # 4. Advanced: Override ranges
    print("\n\n4. Advanced: Override automatic ranges")
    print("=" * 70)
    print("\nIf you need custom ranges, you can still override:")

    report = search(
        model=model,
        data=test_data,
        perturbation="gaussian_noise",
        level_lo=0.0,
        level_hi=0.05,  # Custom range
        max_queries=10,
        device="cpu",
        verbose=False,
    )

    print(f"   Custom range [0.0, 0.05]: threshold = {report.metrics['failure_threshold']:.4f}")

    # 5. Dataset-specific ranges
    print("\n5. Dataset-specific ranges")
    print("=" * 70)
    print("\nDifferent datasets need different ranges:")
    print("   - ImageNet: normalization='imagenet' → automatically uses wider ranges")
    print("   - CIFAR: normalization='cifar10' → automatically uses narrower ranges")
    print("\nExample:")
    print("   search(model, imagenet_data, perturbation='gaussian_noise',")
    print("          normalization='imagenet')  # Ranges auto-selected!")
    print("\nThe 'normalization' parameter serves dual purpose:")
    print("   1. Handles denorm→perturb→renorm workflow")
    print("   2. Selects appropriate parameter ranges for the perturbation")

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("\nKey Benefits:")
    print("  ✓ No domain expertise required")
    print("  ✓ Sensible defaults that work out of the box")
    print("  ✓ Still allows advanced customization when needed")
    print("  ✓ Consistent API across all perturbation types")
    print("=" * 70)


if __name__ == "__main__":
    main()
