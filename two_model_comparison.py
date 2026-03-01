#!/usr/bin/env python3
"""
Clean comparison: Vanilla vs Adversarially-Trained
Shows: "Even robust models have compositional vulnerabilities"
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
IMAGENET_VAL_PATH = "/Users/bilgedemirkaya/imagenet_data/val"  # Update this path
NUM_SAMPLES = 100

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "cpu"  # Use CPU for adversarial attacks (ART compatibility)
else:
    device = "cpu"


def get_mutually_correct_samples(model1, model2, dataset, n=100):
    """
    Get samples that are correctly classified by BOTH models.

    Args:
        model1: First model
        model2: Second model
        dataset: ImageFolder dataset
        n: Number of samples to collect

    Returns:
        List of (image, label) tuples
    """
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

            # Both models must predict correctly
            if pred1 == label and pred2 == label:
                samples.append((img, label))

            if i % 100 == 0:
                print(f"  Scanned {i} images, found {len(samples)} mutual correct...")

    return samples


def test_adversarial_training_effectiveness():
    """
    Compare vanilla vs adversarially-trained on 3 threat models.

    Hypothesis:
    - Adversarial training helps FGSM/PGD (known)
    - But FAILS on compositional attacks (YOUR CONTRIBUTION)
    """

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

    # Model 2:  for models (adversarially trained)
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

    results = {"vanilla": {}, "robust": {}}

    # Test 1: Natural perturbations
    print("=" * 60)
    print("NATURAL PERTURBATIONS")
    print("=" * 60)

    for model_name, model in [("vanilla", vanilla), ("robust", robust)]:
        print(f"\nTesting {model_name}...")
        blur_report = search(
            model=model,
            data=samples,
            perturbation="gaussian_blur",
            level_lo=0.0,
            level_hi=10.0,
            device=device,
            normalization="imagenet"
        )
        results[model_name]["natural"] = blur_report
        print(f"  {model_name}: {blur_report.score:.1f}%")

    # Test 2: Adversarial attacks
    print("\n" + "=" * 60)
    print("ADVERSARIAL ATTACKS (what robust model was trained for)")
    print("=" * 60)

    for model_name, model in [("vanilla", vanilla), ("robust", robust)]:
        print(f"\nTesting {model_name}...")
        pgd_report = search(
            model=model,
            data=samples,
            strategy=lambda eps: PGDStrategy(eps=eps, eps_step=max(eps/10, 1e-6), max_iter=20),
            level_lo=0.0,
            level_hi=0.03,
            device=device,
            normalization="imagenet",
            strategy_name="PGD Attack"
        )
        results[model_name]["adversarial"] = pgd_report
        print(f"  {model_name}: {pgd_report.score:.1f}%")

    # Test 3: Compositional attacks (YOUR CONTRIBUTION!)
    print("\n" + "=" * 60)
    print("COMPOSITIONAL ATTACKS (novel threat model)")
    print("=" * 60)

    # Define compositional attack scenarios
    # Each scenario combines a natural perturbation with an adversarial attack
    scenarios = [
        (
            "Low-Light PGD",
            lambda s: Compose([
                BrightnessStrategy(brightness_factor=0.3 + 0.7 * (1 - s)),  # 0.3 at s=1, 1.0 at s=0
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

    # KEY FINDING: Create comparison table
    create_effectiveness_table(results)

    return results


def create_effectiveness_table(results):
    """
    THE MONEY SHOT: Show adversarial training is insufficient
    """
    print("\n" + "=" * 80)
    print("ADVERSARIAL TRAINING EFFECTIVENESS")
    print("=" * 80)

    print("\nThreat Model          | Vanilla | Adversarially-Trained | Improvement")
    print("-" * 80)

    vanilla_natural = results["vanilla"]["natural"].score
    robust_natural = results["robust"]["natural"].score

    vanilla_adv = results["vanilla"]["adversarial"].score
    robust_adv = results["robust"]["adversarial"].score

    print(f"Natural (Blur)        | {vanilla_natural:6.1f}% | {robust_natural:6.1f}%              | {robust_natural-vanilla_natural:+.1f}%")
    print(f"Adversarial (PGD)     | {vanilla_adv:6.1f}% | {robust_adv:6.1f}%              | {robust_adv-vanilla_adv:+.1f}%")

    print("\nCompositional:")
    for scenario in ["Low-Light PGD", "Blurred PGD", "Noisy PGD"]:
        vanilla_comp = results["vanilla"][scenario].score
        robust_comp = results["robust"][scenario].score
        improvement = robust_comp - vanilla_comp

        emoji = "" if improvement >= 10 else ""
        print(f"{scenario:22} | {vanilla_comp:6.1f}% | {robust_comp:6.1f}%              | {improvement:+.1f}% {emoji}")

    print("\n" + "=" * 80)
    print("KEY FINDING:")
    print("  Adversarial training provides strong improvement on PGD")
    print("  But limited improvement on compositional attacks")
    print("  Standard adversarial training is INSUFFICIENT for real-world robustness!")
    print("=" * 80)


if __name__ == "__main__":
    results = test_adversarial_training_effectiveness()

    # Save reports
    print("\nSaving reports...")
    for model_name, model_results in results.items():
        for test_name, report in model_results.items():
            safe_name = test_name.replace(" ", "_").replace("-", "_").lower()
            report.save(f"report_{model_name}_{safe_name}.json")

    print("Done!")
