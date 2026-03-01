#!/usr/bin/env python3
"""
Example: Understanding Accuracy Preservation vs Prediction Consistency

This example demonstrates what VisProbe actually tests and why it filters
to correctly-classified samples.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from visprobe import search, Perturbation


def create_test_data():
    """Create controlled test data to demonstrate filtering."""
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Take 50 samples
    return [(dataset[i][0], dataset[i][1]) for i in range(50)]


def demonstrate_baseline_filtering():
    """Show how VisProbe filters to correctly-classified samples."""
    print("=" * 70)
    print("Demonstrating: Accuracy Preservation (Baseline Filtering)")
    print("=" * 70)

    # Load model
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.eval()

    # Load test data
    test_data = create_test_data()

    print(f"\nOriginal test set: {len(test_data)} samples")
    print("\nWhat VisProbe does:")
    print("  1. Evaluates baseline accuracy (model correctness on original data)")
    print("  2. Filters to correctly-classified samples (where pred == label)")
    print("  3. Tests if these correct predictions stay correct under perturbation")

    # Run search - it will show baseline filtering
    print("\nRunning search with gaussian_noise...")
    print("-" * 70)

    report = search(
        model=model,
        data=test_data,
        perturbation=Perturbation.GAUSSIAN_NOISE,
        max_queries=10,
        device='cpu',
        verbose=True,  # Shows filtering details
    )

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)

    baseline_acc = report.metrics.get('baseline_accuracy')
    valid_samples = report.metrics.get('valid_samples', 0)
    excluded = report.metrics.get('skipped_samples', 0)
    total = len(test_data)

    print(f"\nBaseline Accuracy: {baseline_acc:.1%}")
    print(f"  - Correct predictions: {valid_samples}/{total}")
    print(f"  - Incorrect predictions: {excluded}/{total} (excluded from testing)")

    print(f"\nRobustness Testing:")
    print(f"  - Tested {valid_samples} correctly-classified samples")
    print(f"  - Failure threshold: {report.metrics['failure_threshold']:.4f}")
    print(f"  - Robustness score: {report.metrics['overall_robustness_score']:.1%}")

    return report


def explain_why_filtering_matters():
    """Explain why we filter to correct samples."""
    print("\n" + "=" * 70)
    print("Why Filter to Correctly-Classified Samples?")
    print("=" * 70)

    print("\n📌 Scenario: Testing a sample where model is initially WRONG")
    print("-" * 70)
    print("  Image: Dog (true label)")
    print("  Model predicts: Cat (WRONG)")
    print()
    print("  After perturbation (blur):")
    print("    Option A: Model still predicts Cat")
    print("      → Should this PASS? No! Model is still wrong.")
    print("    Option B: Model now predicts Bird")
    print("      → Should this PASS? No! Model is still wrong.")
    print("    Option C: Model now predicts Dog")
    print("      → Should this PASS? No! It became correct by accident.")
    print()
    print("  ❌ None of these scenarios make sense for robustness testing!")
    print()
    print("📌 Correct Approach: Only test samples where model is initially CORRECT")
    print("-" * 70)
    print("  Image: Dog (true label)")
    print("  Model predicts: Dog (CORRECT) ✓")
    print()
    print("  After perturbation (blur):")
    print("    Model still predicts: Dog")
    print("      → PASS ✓ Model maintained correct prediction")
    print("    Model now predicts: Cat")
    print("      → FAIL ✗ Perturbation caused incorrect prediction")
    print()
    print("  ✅ This is meaningful robustness testing!")

    print("\n" + "=" * 70)
    print("Key Insight:")
    print("=" * 70)
    print("  VisProbe tests: 'Can the model STAY correct under perturbation?'")
    print("  NOT: 'Do predictions stay consistent regardless of correctness?'")


def explain_edge_case():
    """Explain the edge case with mismatched labels."""
    print("\n" + "=" * 70)
    print("Edge Case: Mismatched Label Spaces")
    print("=" * 70)

    print("\n📌 Scenario: ImageNet model tested on CIFAR-10 data")
    print("-" * 70)
    print("  Problem: CIFAR-10 labels don't match ImageNet classes")
    print("    - CIFAR-10: 10 classes (airplane, automobile, ...)")
    print("    - ImageNet: 1000 classes (golden retriever, sports car, ...)")
    print()
    print("  Result: baseline_accuracy = 0% (no label matches)")
    print()
    print("  ViProbe's Fallback:")
    print("    - Use model's predictions as 'pseudo-labels'")
    print("    - Test prediction consistency: 'Is model consistent with itself?'")
    print("    - Sets baseline_accuracy = -1.0 (special marker)")
    print()
    print("  Example:")
    print("    Original: Model predicts 'golden retriever' (class 207)")
    print("    Perturbed: Model still predicts 'golden retriever'")
    print("      → PASS ✓ Predictions are consistent")
    print("    Perturbed: Model now predicts 'Labrador' (class 208)")
    print("      → FAIL ✗ Predictions changed")

    print("\n  ⚠️  Note: This fallback mode is useful but less meaningful than")
    print("      testing with proper labels. Use it only when necessary.")


def main():
    """Run the demonstration."""
    print("\n" + "=" * 70)
    print("VisProbe: Accuracy Preservation vs Prediction Consistency")
    print("=" * 70)

    # Part 1: Show baseline filtering in action
    report = demonstrate_baseline_filtering()

    # Part 2: Explain why filtering matters
    explain_why_filtering_matters()

    # Part 3: Explain edge case
    explain_edge_case()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nVisProbe tests ACCURACY PRESERVATION:")
    print("  ✓ Filters to correctly-classified samples")
    print("  ✓ Tests if correct predictions stay correct under perturbation")
    print("  ✓ Reports baseline accuracy and excluded samples")
    print("  ✓ Falls back to prediction consistency when labels unavailable")
    print("\nThis is the right approach and matches research best practices!")
    print("=" * 70)


if __name__ == "__main__":
    main()
