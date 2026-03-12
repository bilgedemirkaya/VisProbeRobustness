#!/usr/bin/env python3
"""
Diagnostic script to debug AutoAttack issues with Vanilla ResNet50
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from autoattack import AutoAttack
import time

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class NormalizedModel(torch.nn.Module):
    """Wrapper that applies ImageNet normalization before forward pass."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def test_model_basic(model, name):
    """Test model with random input to ensure it's working"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print('='*60)

    # Create a batch of random images
    test_images = torch.rand(5, 3, 224, 224).to(device)

    # Test forward pass
    with torch.no_grad():
        outputs = model(test_images)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

    print(f"Output shape: {outputs.shape}")
    print(f"Predictions: {preds.cpu().numpy()}")
    print(f"Max prob: {probs.max(dim=1)[0].mean().item():.4f}")
    print(f"Min/Max output values: {outputs.min().item():.2f} / {outputs.max().item():.2f}")

    return True


def test_clean_accuracy(model, name, n_samples=10):
    """Test model on synthetic correctly classified samples"""
    print(f"\n{'='*60}")
    print(f"Testing Clean Accuracy for {name}")
    print('='*60)

    # Create synthetic "perfect" samples that should be correctly classified
    # We'll create samples that strongly activate specific classes
    correct = 0

    for i in range(n_samples):
        # Create a random image
        img = torch.rand(1, 3, 224, 224).to(device)

        # Get the model's prediction for this random image
        with torch.no_grad():
            output = model(img)
            pred = output.argmax().item()

        # Use the model's own prediction as the "true" label
        # This ensures we have correctly classified samples
        label = torch.tensor([pred]).to(device)

        # Verify the model classifies it correctly
        with torch.no_grad():
            output2 = model(img)
            pred2 = output2.argmax().item()

        if pred2 == label.item():
            correct += 1

    acc = correct / n_samples
    print(f"Clean accuracy on synthetic samples: {acc*100:.1f}%")

    if acc < 0.5:
        print("⚠️ WARNING: Model seems to be inconsistent or broken!")

    return acc


def test_autoattack_minimal(model, name, eps=0.01):
    """Test AutoAttack with minimal settings"""
    print(f"\n{'='*60}")
    print(f"Testing AutoAttack (eps={eps}) for {name}")
    print('='*60)

    # Create a small batch of test images and labels
    n_test = 5
    test_images = torch.rand(n_test, 3, 224, 224).to(device)

    # Get model predictions as "ground truth" labels
    with torch.no_grad():
        outputs = model(test_images)
        test_labels = outputs.argmax(dim=1)

    print(f"Test batch shape: {test_images.shape}")
    print(f"Original predictions: {test_labels.cpu().numpy()}")

    # Run AutoAttack
    print(f"\nRunning AutoAttack with eps={eps}...")
    start_time = time.time()

    try:
        adversary = AutoAttack(model, norm='Linf', eps=eps,
                              version='standard', verbose=True)

        # Use only APGD-CE for faster testing
        adversary.attacks_to_run = ['apgd-ce']

        x_adv = adversary.run_standard_evaluation(
            test_images, test_labels, bs=n_test
        )

        elapsed = time.time() - start_time
        print(f"AutoAttack completed in {elapsed:.1f}s")

        # Check robust accuracy
        with torch.no_grad():
            adv_outputs = model(x_adv)
            adv_preds = adv_outputs.argmax(dim=1)
            robust_acc = (adv_preds == test_labels).float().mean().item()

        print(f"Adversarial predictions: {adv_preds.cpu().numpy()}")
        print(f"Robust accuracy: {robust_acc*100:.1f}%")

        # Check perturbation magnitude
        perturbation = (x_adv - test_images).abs()
        print(f"Max perturbation: {perturbation.max().item():.4f} (should be ≤ {eps})")
        print(f"Mean perturbation: {perturbation.mean().item():.4f}")

        if robust_acc == 0.0:
            print("\n⚠️ WARNING: 0% robust accuracy is suspicious!")
            print("Possible issues:")
            print("  1. Model normalization is incorrect")
            print("  2. Epsilon is too large")
            print("  3. Model weights are not loaded correctly")

            # Additional diagnostic: check if model outputs change at all
            with torch.no_grad():
                clean_out = model(test_images)
                adv_out = model(x_adv)
                out_diff = (clean_out - adv_out).abs().mean().item()
                print(f"\nOutput difference (clean vs adv): {out_diff:.4f}")
                if out_diff < 0.01:
                    print("  → Model outputs barely changed despite perturbation!")
                    print("  → This suggests a normalization/preprocessing issue")

    except Exception as e:
        print(f"❌ AutoAttack failed with error: {e}")
        elapsed = time.time() - start_time
        print(f"Failed after {elapsed:.1f}s")

    return robust_acc if 'robust_acc' in locals() else None


def main():
    print("="*80)
    print("AUTOATTACK DIAGNOSTIC TOOL")
    print("="*80)

    # Load models
    print("\n1. Loading models...")

    # Vanilla ResNet50 with normalization wrapper
    vanilla = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    vanilla = vanilla.to(device).eval()

    vanilla_wrapped = NormalizedModel(
        vanilla,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device).eval()

    print("✓ Vanilla ResNet50 loaded with NormalizedModel wrapper")

    # Try loading RobustBench model
    try:
        from robustbench.utils import load_model
        robust_salman = load_model(
            model_name='Salman2020Do_R50',
            dataset='imagenet',
            threat_model='Linf'
        ).to(device).eval()
        print("✓ Salman2020Do_R50 loaded (with internal normalization)")
        has_robust = True
    except Exception as e:
        print(f"✗ Could not load RobustBench model: {e}")
        has_robust = False

    # Test vanilla model
    print("\n2. Testing Vanilla ResNet50...")
    test_model_basic(vanilla_wrapped, "Vanilla ResNet50 (wrapped)")
    clean_acc = test_clean_accuracy(vanilla_wrapped, "Vanilla ResNet50 (wrapped)")

    if clean_acc < 0.8:
        print("\n⚠️ Clean accuracy is suspiciously low!")
        print("Testing unwrapped model for comparison...")
        test_model_basic(vanilla, "Vanilla ResNet50 (unwrapped)")
        test_clean_accuracy(vanilla, "Vanilla ResNet50 (unwrapped)")

    # Test with different epsilon values
    print("\n3. Testing AutoAttack with different epsilon values...")
    for eps in [0.001, 0.01, 0.03]:
        robust_acc = test_autoattack_minimal(vanilla_wrapped,
                                            f"Vanilla ResNet50 (eps={eps})",
                                            eps=eps)
        if robust_acc is not None and robust_acc > 0:
            print(f"✓ Model shows non-zero robustness at eps={eps}")
            break

    # Test robust model if available
    if has_robust:
        print("\n4. Testing Robust Model for comparison...")
        test_model_basic(robust_salman, "Salman2020Do_R50")
        test_clean_accuracy(robust_salman, "Salman2020Do_R50")
        test_autoattack_minimal(robust_salman, "Salman2020Do_R50", eps=0.01)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nRecommendations:")
    print("1. If Vanilla shows 0% with small epsilon, check normalization")
    print("2. If execution is slow but working, consider:")
    print("   - Reducing batch size")
    print("   - Using fewer attack iterations")
    print("   - Using 'rand' version instead of 'standard'")
    print("3. Monitor GPU memory usage during execution")
    print("4. Consider testing with fewer samples initially")


if __name__ == "__main__":
    main()