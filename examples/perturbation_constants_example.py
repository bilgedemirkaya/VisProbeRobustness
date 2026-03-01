#!/usr/bin/env python3
"""
Example: Using Named Constants vs String Literals

This demonstrates the two ways to specify perturbations:
1. String literals (simple, but no IDE support)
2. Named constants (IDE autocomplete + type safety)
"""

from visprobe import search, Perturbation


def example_string_literals(model, data):
    """Traditional approach using string literals."""
    print("Approach 1: String Literals")
    print("-" * 70)

    # Works, but:
    # - No IDE autocomplete
    # - Typos only caught at runtime
    # - Hard to discover what's available
    report = search(
        model=model,
        data=data,
        perturbation="gaussian_noise",  # String literal
    )

    print("✓ Works, but no IDE support for discovering perturbations")
    return report


def example_named_constants(model, data):
    """Modern approach using named constants."""
    print("\nApproach 2: Named Constants")
    print("-" * 70)

    # Better:
    # - IDE autocomplete (type 'Perturbation.' and see all options)
    # - Typos caught immediately as AttributeError
    # - Self-documenting code
    report = search(
        model=model,
        data=data,
        perturbation=Perturbation.GAUSSIAN_NOISE,  # Named constant
    )

    print("✓ IDE autocomplete + type safety!")
    return report


def example_batch_testing(model, data):
    """Testing multiple perturbations with named constants."""
    print("\nApproach 3: Batch Testing with Named Constants")
    print("-" * 70)

    # Clean, discoverable list of perturbations
    perturbations_to_test = [
        Perturbation.GAUSSIAN_NOISE,
        Perturbation.GAUSSIAN_BLUR,
        Perturbation.MOTION_BLUR,
        Perturbation.BRIGHTNESS_DECREASE,
        Perturbation.ROTATION,
    ]

    # Or get all available perturbations
    # perturbations_to_test = Perturbation.all()

    reports = []
    for pert in perturbations_to_test:
        report = search(model=model, data=data, perturbation=pert)
        reports.append(report)
        print(f"  ✓ Tested: {pert}")

    return reports


def show_available_perturbations():
    """Demonstrate discovery of available perturbations."""
    print("\nDiscovering Available Perturbations")
    print("=" * 70)

    # Method 1: Get all as list
    all_perts = Perturbation.all()
    print(f"\nMethod 1: Perturbation.all()")
    print(f"Found {len(all_perts)} perturbations:")
    for p in all_perts:
        print(f"  - {p}")

    # Method 2: Use dir() for IDE-like view
    print(f"\nMethod 2: Available as constants:")
    constants = [attr for attr in dir(Perturbation) if attr.isupper()]
    for const in constants:
        value = getattr(Perturbation, const)
        print(f"  Perturbation.{const:<25} = '{value}'")


def main():
    """Compare approaches."""
    import torch
    import torchvision
    import torchvision.transforms as transforms

    print("=" * 70)
    print("VisProbe - Named Constants vs String Literals")
    print("=" * 70)

    # Show available perturbations first
    show_available_perturbations()

    # Setup simple test
    print("\n" + "=" * 70)
    print("Running Test Examples")
    print("=" * 70)

    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # Create minimal test data
    test_data = [(torch.rand(3, 224, 224), i % 10) for i in range(5)]
    print(f"\nUsing {len(test_data)} synthetic samples for demo")

    # Compare approaches
    example_string_literals(model, test_data)
    example_named_constants(model, test_data)
    example_batch_testing(model, test_data)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nBoth approaches work, but named constants provide:")
    print("  ✓ IDE autocomplete - discover available perturbations")
    print("  ✓ Type safety - catch typos at import time")
    print("  ✓ Refactoring safety - changes propagate automatically")
    print("  ✓ Self-documenting - clear what values are valid")
    print("\nRecommendation: Use Perturbation.* constants in your code!")
    print("=" * 70)


if __name__ == "__main__":
    main()
