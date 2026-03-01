#!/usr/bin/env python3
"""
Test 1: Natural perturbations only (fast, ~5-10 min on CPU)
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import json
from pathlib import Path

from visprobe import search

# Configuration
IMAGENET_VAL_PATH = "/Users/bilgedemirkaya/imagenet_data/val"
NUM_SAMPLES = 500
RESULTS_DIR = Path("experiment_results")
RESULTS_DIR.mkdir(exist_ok=True)

device = "cpu"  # Natural perturbations are fast on CPU


def load_models():
    """Load both models"""
    print("Loading vanilla ResNet50...")
    vanilla = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    vanilla.eval()

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
        print("RobustBench not installed, using placeholder...")
        robust = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        robust.eval()

    return vanilla, robust


def get_mutually_correct_samples(model1, model2, dataset, n=500):
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


def main():
    # Load data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_val = ImageFolder(root=IMAGENET_VAL_PATH, transform=transform)

    # Load models
    vanilla, robust = load_models()

    # Get common samples
    print(f"\nFinding {NUM_SAMPLES} mutually correct samples...")
    samples = get_mutually_correct_samples(vanilla, robust, imagenet_val, n=NUM_SAMPLES)
    print(f"Found {len(samples)} samples\n")

    # Save samples indices for consistency across tests
    # (In practice, save the actual indices to reload)

    # Test natural perturbations
    print("=" * 60)
    print("TEST 1: NATURAL PERTURBATIONS")
    print("=" * 60)

    results = {}

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

        results[f"{model_name}_natural"] = {
            "score": blur_report.score,
            "threshold": blur_report.threshold
        }
        print(f"  {model_name}: {blur_report.score:.1f}%")

        # Save individual report
        blur_report.save(str(RESULTS_DIR / f"report_{model_name}_natural.json"))

    # Save combined results
    with open(RESULTS_DIR / "results_natural.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Run test_2_adversarial.py next (slow, run overnight)")


if __name__ == "__main__":
    main()
