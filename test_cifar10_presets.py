#!/usr/bin/env python3
"""
Test presets with CIFAR-10 data.

Simple example showing how to use visprobe.search() with different presets.

Usage:
    python test_cifar10_presets.py [--preset PRESET] [--skip-adversarial]
"""

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from visprobe import search
from visprobe.presets import (
    list_available_presets,
    requires_art,
    get_preset_info,
    CIFAR_MEAN,
    CIFAR_STD,
)


def get_imagenet_classes():
    """Get ImageNet class names."""
    try:
        from torchvision.models import ResNet18_Weights
        return ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    except Exception:
        return None


# Check if ART is available
try:
    from art.attacks.evasion import FastGradientMethod
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test VisProbe presets with CIFAR-10")
    parser.add_argument("--preset", type=str, default=None, help="Specific preset to test")
    parser.add_argument("--skip-adversarial", action="store_true", help="Skip adversarial presets")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--budget", type=int, default=100, help="Search budget per preset")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Testing VisProbe Presets with CIFAR-10")
    print("=" * 70)

    if ART_AVAILABLE:
        print("✓ Adversarial Robustness Toolbox (ART) is available")
    else:
        print("⚠ ART not installed - adversarial presets will be skipped")

    # 1. Load model
    print("\n1. Loading model...")
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.eval()
    print(f"   ✓ Model: {model.__class__.__name__}")

    # 2. Load CIFAR-10 test data
    print("\n2. Loading CIFAR-10 test data...")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    try:
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        test_data = [(testset[i][0], testset[i][1]) for i in range(args.samples)]
        print(f"   ✓ Loaded {args.samples} CIFAR-10 test samples")
    except Exception as e:
        print(f"   ⚠ Could not load CIFAR-10: {e}")
        test_data = [(torch.rand(3, 224, 224), i % 10) for i in range(args.samples)]

    # 3. Determine presets to test
    available_presets = list_available_presets()

    if args.preset:
        if args.preset not in available_presets:
            print(f"   ❌ Preset '{args.preset}' not found! Available: {available_presets}")
            return
        presets_to_test = [args.preset]
    else:
        presets_to_test = available_presets

    # Filter adversarial presets if needed
    if args.skip_adversarial or not ART_AVAILABLE:
        presets_to_test = [p for p in presets_to_test if not requires_art(p)]

    print(f"\n3. Testing presets: {presets_to_test}")
    print("=" * 70)

    # 4. Run tests
    IMAGENET_CLASSES = get_imagenet_classes()
    results = {}

    for preset_name in presets_to_test:
        print(f"\n{'=' * 70}")
        print(f"Testing Preset: {preset_name.upper()}")
        print(f"{'=' * 70}")

        try:
            info = get_preset_info(preset_name)
            if info:
                print(f"   Description: {info.get('description', 'N/A')}")
        except Exception:
            pass

        if requires_art(preset_name) and not ART_AVAILABLE:
            print(f"   ⚠ Skipping - requires ART")
            continue

        try:
            report = search(
                model=model,
                data=test_data,
                preset=preset_name,
                budget=args.budget,
                device="cpu",
                normalization="cifar10",
                class_names=IMAGENET_CLASSES,
            )

            # Save report (auto-detects test file name and saves to visprobe results dir)
            save_path = report.save()
            print(f"   ✓ Report saved to: {save_path}")

            results[preset_name] = report

        except ImportError as e:
            print(f"   ⚠ Skipping - missing dependency: {e}")
        except Exception as e:
            print(f"   ❌ Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Summary
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)

    if results:
        print(f"\n✅ Successfully tested {len(results)}/{len(presets_to_test)} presets\n")

        for preset_name, report in sorted(results.items(), key=lambda x: x[1].score):
            print(f"  {preset_name:20s} Score: {report.score:5.1f}%  Failures: {report.failed_samples}/{report.total_samples}")

        print("\n" + "=" * 70)
        print("🎉 Tests completed! View results with: visprobe visualize")
        print("=" * 70)
    else:
        print("\n❌ No presets were successfully tested")


if __name__ == "__main__":
    main()
