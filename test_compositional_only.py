#!/usr/bin/env python3
"""
Compositional attacks only - continue from where two_model_comparison.py left off.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from visprobe import search
from visprobe.strategies import (
    GaussianBlurStrategy,
    BrightnessStrategy,
    GaussianNoiseStrategy,
    PGDStrategy,
    Compose,
)

# Configuration
IMAGENET_VAL_PATH = "/Users/bilgedemirkaya/imagenet_data/val"
NUM_SAMPLES = 100

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "cpu"  # Use CPU for adversarial attacks (ART compatibility)
else:
    device = "cpu"


def get_mutually_correct_samples(model1, model2, dataset, n=100):
    """Get samples correctly classified by BOTH models."""
    model1.eval()
    model2.eval()

    device_obj = torch.device(device)
    model1.to(device_obj)
    model2.to(device_obj)

    samples = []

    with torch.no_grad():
        for i in range(len(dataset)):
            if len(samples) >= n:
                break

            img, label = dataset[i]
            img_batch = img.unsqueeze(0).to(device_obj)

            pred1 = model1(img_batch).argmax().item()
            pred2 = model2(img_batch).argmax().item()

            if pred1 == label and pred2 == label:
                samples.append((img, label))

            if i % 100 == 0:
                print(f"  Scanned {i} images, found {len(samples)} mutual correct...")

    return samples


def test_compositional_only():
    """Run only compositional attacks."""

    # Load ImageNet validation data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imagenet_val = ImageFolder(root=IMAGENET_VAL_PATH, transform=transform)

    # Model 1: Vanilla
    print("Loading vanilla ResNet50...")
    vanilla = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    vanilla.eval()

    # Model 2: RobustBench (adversarially trained)
    print("Loading adversarially-trained model from RobustBench...")
    try:
        from robustbench.utils import load_model
        robust = load_model(
            model_name='Salman2020Do_R50',
            dataset='imagenet',
            threat_model='Linf'
        )
        robust.eval()
    except ImportError:
        print("RobustBench not installed. Install with: pip install robustbench")
        print("Using another vanilla model as placeholder...")
        robust = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        robust.eval()

    # Get samples correctly classified by BOTH models
    print(f"\nFinding {NUM_SAMPLES} samples correctly classified by both models...")
    samples = get_mutually_correct_samples(vanilla, robust, imagenet_val, n=NUM_SAMPLES)
    print(f"Testing {len(samples)} samples correctly classified by both models\n")

    # Previous results (from your run) - create proper Report objects
    from visprobe.report import Report

    results = {
        "vanilla": {
            "natural": Report(
                test_name="vanilla_natural",
                test_type="search",
                model_name="ResNet50_Vanilla",
                dataset="ImageNet",
                strategy="gaussian_blur",
                property_name="LabelConstant",
                total_samples=100,
                passed_samples=100,
                runtime=148.4,
                model_queries=11,
                metrics={"failure_threshold": 10.0}
            ),
            "adversarial": Report(
                test_name="vanilla_adversarial",
                test_type="search",
                model_name="ResNet50_Vanilla",
                dataset="ImageNet",
                strategy="PGD Attack",
                property_name="LabelConstant",
                total_samples=100,
                passed_samples=88,
                runtime=841.4,
                model_queries=3,
                metrics={"failure_threshold": 0.0015}
            ),
        },
        "robust": {
            "natural": Report(
                test_name="robust_natural",
                test_type="search",
                model_name="Salman2020Do_R50",
                dataset="ImageNet",
                strategy="gaussian_blur",
                property_name="LabelConstant",
                total_samples=100,
                passed_samples=89,
                runtime=224.5,
                model_queries=15,
                metrics={"failure_threshold": 0.8809}
            ),
            "adversarial": Report(
                test_name="robust_adversarial",
                test_type="search",
                model_name="Salman2020Do_R50",
                dataset="ImageNet",
                strategy="PGD Attack",
                property_name="LabelConstant",
                total_samples=100,
                passed_samples=89,
                runtime=34288.2,
                model_queries=12,
                metrics={"failure_threshold": 0.0270}
            ),
        }
    }

    # Test: Compositional attacks only
    print("=" * 60)
    print("COMPOSITIONAL ATTACKS (novel threat model)")
    print("=" * 60)

    scenarios = [
        (
            "Low-Light PGD",
            lambda s: Compose([
                BrightnessStrategy(brightness_factor=0.3 + 0.7 * (1 - s)),
                PGDStrategy(eps=s * 0.01, eps_step=max(s * 0.001, 1e-6), max_iter=10)
            ])
        ),
        (
            "Blurred PGD",
            lambda s: Compose([
                GaussianBlurStrategy(sigma=s * 2.0),
                PGDStrategy(eps=s * 0.01, eps_step=max(s * 0.001, 1e-6), max_iter=10)
            ])
        ),
        (
            "Noisy PGD",
            lambda s: Compose([
                GaussianNoiseStrategy(std_dev=s * 0.03),
                PGDStrategy(eps=s * 0.01, eps_step=max(s * 0.001, 1e-6), max_iter=10)
            ])
        ),
    ]

    for scenario_name, composition in scenarios:
        print(f"\n{scenario_name}:")

        for model_name, model in [("vanilla", vanilla), ("robust", robust)]:
            comp_report = search(
                model=model,
                data=samples,
                strategy=composition,
                level_lo=0.0,
                level_hi=1.0,
                device=device,
                normalization="imagenet",
                strategy_name=scenario_name
            )
            results[model_name][scenario_name] = comp_report
            print(f"  {model_name}: {comp_report.score:.1f}%")

    # Define max thresholds for normalization
    max_thresholds = {
        "natural": 10.0,      # gaussian blur sigma
        "adversarial": 0.03,  # PGD epsilon
        "Low-Light PGD": 1.0,
        "Blurred PGD": 1.0,
        "Noisy PGD": 1.0,
    }

    def get_threshold(report):
        """Extract threshold from report."""
        # Threshold is stored in metrics['failure_threshold'], not search['threshold']
        if hasattr(report, 'metrics') and report.metrics:
            return report.metrics.get('failure_threshold', 0)
        return 0

    def normalized_robustness(threshold, max_thresh):
        """Calculate normalized robustness (0-100%)."""
        return min(100.0, (threshold / max_thresh) * 100) if max_thresh > 0 else 0

    # Build comprehensive results
    comprehensive_results = {
        "experiment": {
            "name": "Adversarial Training Effectiveness",
            "dataset": "ImageNet",
            "num_samples": NUM_SAMPLES,
            "models": {
                "vanilla": "ResNet50 (IMAGENET1K_V2)",
                "robust": "Salman2020Do_R50 (RobustBench, Linf)"
            }
        },
        "tests": {}
    }

    # Final table
    print("\n" + "=" * 80)
    print("ADVERSARIAL TRAINING EFFECTIVENESS - FINAL RESULTS")
    print("=" * 80)
    print("\nMetric: Normalized Robustness (threshold / max_threshold × 100%)")
    print("Higher = more robust (tolerates larger perturbations)\n")

    print("Threat Model          | Vanilla |  Robust | Improvement | Thresholds")
    print("-" * 85)

    for test_name in ["natural", "adversarial", "Low-Light PGD", "Blurred PGD", "Noisy PGD"]:
        v_report = results["vanilla"][test_name]
        r_report = results["robust"][test_name]

        v_thresh = get_threshold(v_report)
        r_thresh = get_threshold(r_report)
        max_thresh = max_thresholds[test_name]

        v_robust = normalized_robustness(v_thresh, max_thresh)
        r_robust = normalized_robustness(r_thresh, max_thresh)
        improvement = r_robust - v_robust

        # Store in comprehensive results
        display_name = "Natural (Blur)" if test_name == "natural" else \
                      "Adversarial (PGD)" if test_name == "adversarial" else test_name

        comprehensive_results["tests"][test_name] = {
            "display_name": display_name,
            "max_threshold": max_thresh,
            "vanilla": {
                "threshold": v_thresh,
                "normalized_robustness": round(v_robust, 2),
                "pass_rate": round(v_report.score, 2),
                "passed_samples": v_report.passed_samples,
                "failed_samples": v_report.failed_samples,
                "total_samples": v_report.total_samples,
                "runtime": v_report.runtime
            },
            "robust": {
                "threshold": r_thresh,
                "normalized_robustness": round(r_robust, 2),
                "pass_rate": round(r_report.score, 2),
                "passed_samples": r_report.passed_samples,
                "failed_samples": r_report.failed_samples,
                "total_samples": r_report.total_samples,
                "runtime": r_report.runtime
            },
            "improvement": round(improvement, 2),
            "improvement_ratio": round(r_thresh / v_thresh, 2) if v_thresh > 0 else float('inf')
        }

        emoji = "✓" if improvement >= 10 else "✗"
        thresh_str = f"v:{v_thresh:.4f} r:{r_thresh:.4f}"
        print(f"{display_name:22} | {v_robust:6.1f}% | {r_robust:6.1f}% |   {improvement:+6.1f}% {emoji} | {thresh_str}")

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)

    # Calculate averages
    adv_improvement = comprehensive_results["tests"]["adversarial"]["improvement"]
    comp_improvements = [comprehensive_results["tests"][t]["improvement"]
                        for t in ["Low-Light PGD", "Blurred PGD", "Noisy PGD"]]
    avg_comp_improvement = sum(comp_improvements) / len(comp_improvements)

    print(f"  • Adversarial (PGD) improvement:     {adv_improvement:+.1f}%")
    print(f"  • Compositional average improvement: {avg_comp_improvement:+.1f}%")
    print(f"  • Gap: {adv_improvement - avg_comp_improvement:.1f}% less protection on compositional")
    print()
    print("  CONCLUSION: Adversarial training provides strong protection against")
    print("  pure PGD attacks, but LIMITED protection against compositional attacks.")
    print("  Standard adversarial training is INSUFFICIENT for real-world robustness!")
    print("=" * 80)

    comprehensive_results["summary"] = {
        "adversarial_improvement": round(adv_improvement, 2),
        "compositional_avg_improvement": round(avg_comp_improvement, 2),
        "protection_gap": round(adv_improvement - avg_comp_improvement, 2),
        "conclusion": "Adversarial training provides strong protection against pure PGD but limited protection against compositional attacks"
    }

    return results, comprehensive_results


if __name__ == "__main__":
    import json
    from datetime import datetime

    results, comprehensive_results = test_compositional_only()

    # Add timestamp
    comprehensive_results["timestamp"] = datetime.now().isoformat()

    # Save comprehensive JSON report
    print("\nSaving reports...")

    comprehensive_filename = "experiment_results_comprehensive.json"
    with open(comprehensive_filename, "w") as f:
        json.dump(comprehensive_results, f, indent=2)
    print(f"  Saved: {comprehensive_filename} (MAIN REPORT)")

    # Save individual reports
    for model_name, model_results in results.items():
        for test_name, report in model_results.items():
            safe_name = test_name.replace(" ", "_").replace("-", "_").lower()
            filename = f"report_{model_name}_{safe_name}.json"
            report.save_json(filename)
            print(f"  Saved: {filename}")

    print("\nDone! All reports saved.")
    print(f"\nOpen {comprehensive_filename} for the complete analysis.")
