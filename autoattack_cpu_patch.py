#!/usr/bin/env python3
"""
Patch to run AutoAttack on CPU (limited functionality)
Note: Only some attacks work on CPU, and it will be VERY slow
"""

import torch
import torchvision.models as models
import numpy as np
from autoattack import AutoAttack
import warnings

# Force CPU usage
device = "cpu"
print(f"Forcing CPU usage (will be slow!)")

# Suppress CUDA warnings
warnings.filterwarnings("ignore", message=".*CUDA.*")


class NormalizedModel(torch.nn.Module):
    """Wrapper that applies ImageNet normalization before forward pass."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def run_autoattack_cpu_minimal():
    """Minimal AutoAttack test on CPU"""

    print("Loading model...")
    vanilla = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    vanilla = vanilla.to(device).eval()

    vanilla_wrapped = NormalizedModel(
        vanilla,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device).eval()

    # Create tiny test batch (CPU is very slow!)
    print("Creating test data (5 samples only for CPU)...")
    test_images = torch.rand(5, 3, 224, 224).to(device)

    # Get model's own predictions as labels
    with torch.no_grad():
        outputs = vanilla_wrapped(test_images)
        test_labels = outputs.argmax(dim=1)

    print(f"Test labels: {test_labels.numpy()}")

    # Try AutoAttack with CPU-compatible settings
    print("\nAttempting AutoAttack on CPU (this will be SLOW)...")
    print("Using only FGSM attack (fastest, CPU-compatible)...")

    try:
        # Create adversary with minimal settings
        adversary = AutoAttack(
            vanilla_wrapped,
            norm='Linf',
            eps=0.01,
            version='custom',
            attacks_to_run=['fgsm'],  # FGSM might work on CPU
            verbose=True,
            device=device
        )

        # Manually set device to CPU for all components
        adversary.device = device

        # Try to run
        x_adv = adversary.run_standard_evaluation(
            test_images, test_labels, bs=5
        )

        # Check results
        with torch.no_grad():
            adv_outputs = vanilla_wrapped(x_adv)
            adv_preds = adv_outputs.argmax(dim=1)
            robust_acc = (adv_preds == test_labels).float().mean().item()

        print(f"\n✓ Success! Robust accuracy: {robust_acc*100:.1f}%")

    except Exception as e:
        print(f"\n✗ AutoAttack failed on CPU: {e}")
        print("\nAutoAttack is not fully CPU-compatible.")
        print("Please use one of these alternatives:")
        print("1. Google Colab with GPU (best)")
        print("2. cpu_compatible_attack.py using Foolbox")
        print("3. Local GPU if available")


if __name__ == "__main__":
    print("="*80)
    print("AUTOATTACK CPU PATCH TEST")
    print("="*80)
    print("\n⚠️ WARNING: AutoAttack on CPU is experimental and VERY slow!")
    print("           Recommended to use Google Colab with GPU instead.\n")

    run_autoattack_cpu_minimal()