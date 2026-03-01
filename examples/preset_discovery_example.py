#!/usr/bin/env python3
"""
Example: Discovering and Using Presets

This demonstrates how to discover available presets and inspect their details
before running robustness tests.
"""

from visprobe import list_presets, get_preset_info, search
import torch
import torchvision
import torchvision.transforms as transforms


def main():
    print("=" * 70)
    print("VisProbe - Preset Discovery Demo")
    print("=" * 70)

    # 1. List all available presets (simple)
    print("\n1. Getting available presets...")
    print("-" * 70)
    presets = list_presets()
    print(f"Found {len(presets)} presets:")
    for name, description in presets.items():
        print(f"  • {name}: {description}")

    # 2. List with verbose output
    print("\n2. Detailed preset information...")
    print("-" * 70)
    list_presets(verbose=True)

    # 3. Get detailed info on a specific preset
    print("\n3. Inspecting the 'natural' preset...")
    print("-" * 70)
    info = get_preset_info('natural')
    print(info)

    # 4. Run a test with the preset
    print("\n4. Running search with 'natural' preset...")
    print("-" * 70)

    # Load a simple model and data
    print("Loading model and data...")
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    try:
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        test_data = [(testset[i][0], testset[i][1]) for i in range(20)]
    except Exception:
        # Fallback to synthetic data
        test_data = [(torch.rand(3, 224, 224), i % 10) for i in range(20)]

    print(f"Loaded {len(test_data)} samples")

    # Run search with the preset
    print("\nRunning robustness test with 'natural' preset...")
    report = search(
        model=model,
        data=test_data,
        preset='natural',
        budget=100,  # Lower budget for demo
        device='cpu',
        normalization='cifar10',
    )

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Overall Score: {report.score:.1f}%")
    print(f"Passed: {report.passed_samples}/{report.total_samples}")
    print(f"Failed: {report.failed_samples}/{report.total_samples}")

    # Save report
    save_path = report.save()
    print(f"\nReport saved: {save_path}")
    print("\nView results: visprobe visualize")

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
