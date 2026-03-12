#!/usr/bin/env python3
"""
Diagnostic script to find why adversarial accuracy is 0%
Tests multiple hypotheses and epsilon values
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Set seed
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
        # x is expected to be in [0, 1]
        normalized = (x - self.mean) / self.std
        return self.model(normalized)


def create_high_confidence_samples(model, n_samples=10):
    """Create samples that the model classifies with high confidence"""
    model.eval()
    samples = []
    labels = []
    confidences = []

    print(f"Creating {n_samples} high-confidence samples...")

    with torch.no_grad():
        attempts = 0
        while len(samples) < n_samples and attempts < n_samples * 10:
            attempts += 1

            # Create a random image
            img = torch.rand(1, 3, 224, 224).to(device)

            # Get prediction and confidence
            output = model(img)
            probs = F.softmax(output, dim=1)
            confidence, pred = probs.max(dim=1)

            # Only keep if confidence is high
            if confidence.item() > 0.8:  # High confidence threshold
                samples.append(img.cpu())
                labels.append(pred.item())
                confidences.append(confidence.item())
                print(f"  Sample {len(samples)}: class {pred.item()}, conf={confidence.item():.3f}")

    if len(samples) < n_samples:
        print(f"⚠️ Warning: Only found {len(samples)} high-confidence samples")
        # Fill with lower confidence samples
        while len(samples) < n_samples:
            img = torch.rand(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(img)
                pred = output.argmax(dim=1)
                samples.append(img.cpu())
                labels.append(pred.item())

    images = torch.cat(samples, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels, confidences


def simple_fgsm_attack(model, images, labels, epsilon):
    """
    Simple FGSM implementation for debugging
    """
    model.eval()
    images = images.to(device).requires_grad_(True)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Create perturbation
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_images = images + epsilon * sign_data_grad

    # Clamp to [0,1]
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images.detach()


def test_attack_implementation(model, name="Model"):
    """Test if attacks are working correctly"""
    print(f"\n{'='*70}")
    print(f"TESTING ATTACK IMPLEMENTATION ON {name}")
    print('='*70)

    # Create high-confidence test samples
    test_images, test_labels, confidences = create_high_confidence_samples(model, n_samples=5)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    if confidences:
        print(f"Average confidence: {np.mean(confidences):.3f}")

    # Test with progressively larger epsilons
    epsilons = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = []

    print("\nTesting FGSM with different epsilon values:")
    print("-" * 50)
    print(f"{'Epsilon':<10} {'Accuracy':<10} {'Avg Perturbation':<20} {'Changed Predictions'}")
    print("-" * 50)

    for eps in epsilons:
        if eps == 0:
            # Clean accuracy
            perturbed = test_images
        else:
            # Attack
            perturbed = simple_fgsm_attack(model, test_images, test_labels, eps)

        # Evaluate
        with torch.no_grad():
            outputs = model(perturbed)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == test_labels).float().mean().item()

            # Check perturbation magnitude
            perturbation = (perturbed - test_images).abs()
            avg_pert = perturbation.mean().item()
            max_pert = perturbation.max().item()

            # Count changed predictions
            changed = (predictions != test_labels).sum().item()

        results.append({
            'epsilon': eps,
            'accuracy': accuracy,
            'avg_pert': avg_pert,
            'max_pert': max_pert,
            'changed': changed
        })

        print(f"{eps:<10.3f} {accuracy*100:<10.1f}% {avg_pert:<20.6f} {changed}/{len(test_labels)}")

    # Analyze results
    print("\n" + "="*50)
    print("ANALYSIS:")

    # Find where accuracy drops
    for i, r in enumerate(results[1:], 1):
        if r['accuracy'] < results[0]['accuracy']:
            print(f"✓ Accuracy drops at epsilon={r['epsilon']:.3f}")
            print(f"  Clean: {results[0]['accuracy']*100:.1f}% → Adversarial: {r['accuracy']*100:.1f}%")
            break
    else:
        print("❌ WARNING: Accuracy never drops! Attack might not be working.")

    # Check if perturbations are being applied
    if results[-1]['avg_pert'] < 0.001:
        print("❌ WARNING: Perturbations are too small or not being applied!")
    else:
        print(f"✓ Perturbations are being applied (max avg: {results[-1]['avg_pert']:.4f})")

    return results


def test_pgd_attack(model, images, labels, epsilon, steps=20):
    """
    PGD attack implementation for testing
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    # Random initialization
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta.requires_grad = True

    for _ in range(steps):
        # Forward pass
        outputs = model(torch.clamp(images + delta, 0, 1))
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # Update perturbation
        delta.data = delta + epsilon/steps * delta.grad.sign()
        delta.data = torch.clamp(delta, -epsilon, epsilon)
        delta.grad.zero_()

    perturbed = torch.clamp(images + delta.detach(), 0, 1)
    return perturbed


def test_with_real_images(model):
    """Test with real ImageNet images if available"""
    print("\n" + "="*70)
    print("TESTING WITH REAL IMAGENET SAMPLES (if available)")
    print("="*70)

    # Try to load real ImageNet samples
    imagenet_paths = [
        '/content/drive/MyDrive/val',  # Colab path
        './imagenet_val',  # Local path
        '../imagenet_val'  # Parent directory
    ]

    dataset = None
    for path in imagenet_paths:
        if Path(path).exists():
            print(f"Found ImageNet data at: {path}")
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            try:
                dataset = ImageFolder(root=path, transform=transform)
                break
            except:
                continue

    if dataset is None:
        print("No ImageNet data found. Using synthetic samples.")
        return test_attack_implementation(model, "Model (synthetic)")

    # Get 10 correctly classified samples
    correct_samples = []
    correct_labels = []

    model.eval()
    with torch.no_grad():
        for i in range(min(100, len(dataset))):
            img, label = dataset[i]
            img_batch = img.unsqueeze(0).to(device)

            output = model(img_batch)
            pred = output.argmax().item()

            if pred == label:
                correct_samples.append(img)
                correct_labels.append(label)

            if len(correct_samples) >= 10:
                break

    if len(correct_samples) == 0:
        print("No correctly classified samples found!")
        return None

    print(f"Found {len(correct_samples)} correctly classified samples")

    # Stack samples
    images = torch.stack(correct_samples).to(device)
    labels = torch.tensor(correct_labels).to(device)

    # Test attacks
    print("\nTesting attacks on REAL ImageNet samples:")
    print("-" * 50)

    epsilons = [0.0, 0.001, 0.01, 0.03, 0.05]
    for eps in epsilons:
        if eps == 0:
            acc = 1.0  # Clean accuracy is 100% by construction
        else:
            # Test FGSM
            adv_images = simple_fgsm_attack(model, images, labels, eps)
            with torch.no_grad():
                outputs = model(adv_images)
                preds = outputs.argmax(dim=1)
                acc = (preds == labels).float().mean().item()

        print(f"Epsilon={eps:.3f}: Accuracy={acc*100:.1f}%")

    # Test PGD
    print("\nTesting PGD attack:")
    pgd_adv = test_pgd_attack(model, images, labels, epsilon=0.01, steps=20)
    with torch.no_grad():
        outputs = model(pgd_adv)
        preds = outputs.argmax(dim=1)
        pgd_acc = (preds == labels).float().mean().item()
    print(f"PGD (eps=0.01, 20 steps): Accuracy={pgd_acc*100:.1f}%")

    return True


def visualize_attack_effect(model):
    """Visualize the effect of adversarial perturbations"""
    print("\n" + "="*70)
    print("VISUALIZING ATTACK EFFECTS")
    print("="*70)

    # Create a test image
    test_img = torch.rand(1, 3, 224, 224).to(device)

    # Get original prediction
    with torch.no_grad():
        orig_output = model(test_img)
        orig_probs = F.softmax(orig_output, dim=1)
        orig_conf, orig_pred = orig_probs.max(dim=1)

    print(f"Original: class {orig_pred.item()}, confidence {orig_conf.item():.3f}")

    # Create adversarial example
    adv_img = simple_fgsm_attack(model, test_img, orig_pred, epsilon=0.01)

    # Get adversarial prediction
    with torch.no_grad():
        adv_output = model(adv_img)
        adv_probs = F.softmax(adv_output, dim=1)
        adv_conf, adv_pred = adv_probs.max(dim=1)

    print(f"Adversarial: class {adv_pred.item()}, confidence {adv_conf.item():.3f}")

    # Compute perturbation
    perturbation = (adv_img - test_img).abs()
    print(f"Perturbation stats:")
    print(f"  Mean: {perturbation.mean().item():.6f}")
    print(f"  Max: {perturbation.max().item():.6f}")
    print(f"  Min: {perturbation.min().item():.6f}")

    # Check if prediction changed
    if orig_pred.item() != adv_pred.item():
        print("✓ Attack successful: prediction changed!")
    else:
        print("✗ Attack failed: prediction unchanged")
        print("  This could indicate:")
        print("  - Epsilon too small")
        print("  - Model is robust")
        print("  - Attack implementation issue")


def main():
    print("="*80)
    print("ADVERSARIAL ACCURACY DIAGNOSTIC")
    print("="*80)

    # Load model
    print("\n1. Loading Vanilla ResNet50...")
    vanilla = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    vanilla = vanilla.to(device).eval()

    vanilla_wrapped = NormalizedModel(
        vanilla,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device).eval()

    # Run tests
    print("\n2. Testing attack implementation...")
    results = test_attack_implementation(vanilla_wrapped, "Vanilla ResNet50")

    print("\n3. Testing with real images (if available)...")
    test_with_real_images(vanilla_wrapped)

    print("\n4. Visualizing attack effects...")
    visualize_attack_effect(vanilla_wrapped)

    # Final diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    # Check if attacks are working
    clean_acc = results[0]['accuracy']
    final_acc = results[-1]['accuracy']

    if final_acc == 0.0 and clean_acc > 0.5:
        print("✓ Attacks ARE working - model is just very vulnerable")
        print("  Vanilla ResNet50 can indeed drop to 0% with small perturbations")
        print("  Try smaller epsilon values (0.001-0.005) for non-zero accuracy")
    elif final_acc == clean_acc:
        print("❌ Attacks NOT working properly")
        print("  Possible issues:")
        print("  - Gradient computation problems")
        print("  - Model not in eval mode")
        print("  - Normalization issues")
    else:
        print("✓ Attacks are working normally")
        print(f"  Accuracy drops from {clean_acc*100:.1f}% to {final_acc*100:.1f}%")

    print("\nRECOMMENDATIONS:")
    print("1. For Vanilla ResNet50, use epsilon ≤ 0.005 for non-zero accuracy")
    print("2. Test with robust models for comparison")
    print("3. Use real ImageNet data when possible")
    print("4. Monitor confidence scores, not just accuracy")


if __name__ == "__main__":
    main()