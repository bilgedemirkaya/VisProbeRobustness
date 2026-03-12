#!/usr/bin/env python3
"""
CPU-compatible adversarial attack evaluation using Foolbox
(Alternative to AutoAttack which requires CUDA)
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import json
import time

# Install foolbox if not available
try:
    import foolbox as fb
except ImportError:
    print("Installing foolbox...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'foolbox'])
    import foolbox as fb

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("⚠️ Running on CPU - this will be slower but will work!")

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class NormalizedModel(torch.nn.Module):
    """Wrapper that applies ImageNet normalization before forward pass."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def create_synthetic_data(n_samples=50):
    """Create synthetic test data"""
    print(f"Creating {n_samples} synthetic test samples...")

    # Create random images
    images = torch.rand(n_samples, 3, 224, 224)

    # Create random labels (ImageNet has 1000 classes)
    labels = torch.randint(0, 1000, (n_samples,))

    return images, labels


def evaluate_clean(model, images, labels):
    """Evaluate clean accuracy"""
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

    return accuracy


def evaluate_with_foolbox(model, images, labels, eps=0.01, attack_type='pgd'):
    """
    Evaluate robustness using Foolbox attacks (CPU-compatible)

    Args:
        model: PyTorch model
        images: Input images [0,1]
        labels: True labels
        eps: Epsilon for L-inf perturbation
        attack_type: 'pgd', 'fgsm', or 'deepfool'
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    # Create Foolbox model
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Select attack
    if attack_type == 'pgd':
        attack = fb.attacks.LinfPGD(steps=40, abs_stepsize=eps/10)
    elif attack_type == 'fgsm':
        attack = fb.attacks.FGSM()
    elif attack_type == 'deepfool':
        attack = fb.attacks.LinfDeepFool()
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    print(f"Running {attack_type.upper()} attack with eps={eps}...")

    # Run attack
    _, adv_images, success = attack(fmodel, images, labels, epsilons=eps)

    # Calculate robust accuracy
    robust_accuracy = 1.0 - success.float().mean().item()

    return robust_accuracy, adv_images


def apply_environmental_perturbation(images, perturbation_type, severity):
    """Apply environmental perturbations"""

    if perturbation_type == 'brightness':
        # Reduce brightness
        factor = 1.0 - 0.7 * severity
        return images * factor

    elif perturbation_type == 'noise':
        # Add Gaussian noise
        noise = torch.randn_like(images) * (severity * 0.1)
        return (images + noise).clamp(0, 1)

    elif perturbation_type == 'blur':
        # Simple box blur (CPU-friendly)
        from torch.nn.functional import avg_pool2d
        if severity > 0:
            kernel_size = int(severity * 5) * 2 + 1  # Ensure odd
            padding = kernel_size // 2
            blurred = avg_pool2d(images, kernel_size=kernel_size,
                                stride=1, padding=padding)
            return blurred
        return images

    else:
        return images


def run_compositional_evaluation(model, images, labels, env_severity=0.5, eps=0.01):
    """
    Test model under compositional threats (environmental + adversarial)
    """
    results = {}

    # Clean evaluation
    clean_acc = evaluate_clean(model, images, labels)
    results['clean'] = clean_acc
    print(f"Clean accuracy: {clean_acc*100:.1f}%")

    # Pure adversarial
    adv_acc, _ = evaluate_with_foolbox(model, images, labels, eps=eps, attack_type='pgd')
    results['adversarial'] = adv_acc
    print(f"Adversarial accuracy (eps={eps}): {adv_acc*100:.1f}%")

    # Environmental perturbations
    for env_type in ['brightness', 'noise', 'blur']:
        env_images = apply_environmental_perturbation(images, env_type, env_severity)

        # Pure environmental
        env_acc = evaluate_clean(model, env_images, labels)
        results[f'{env_type}'] = env_acc
        print(f"{env_type.capitalize()} accuracy (severity={env_severity}): {env_acc*100:.1f}%")

        # Compositional (environmental + adversarial)
        comp_acc, _ = evaluate_with_foolbox(model, env_images, labels, eps=eps, attack_type='pgd')
        results[f'{env_type}_adversarial'] = comp_acc
        print(f"{env_type.capitalize()} + Adversarial: {comp_acc*100:.1f}%")

    return results


def main():
    print("="*80)
    print("CPU-COMPATIBLE ADVERSARIAL ROBUSTNESS EVALUATION")
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

    print("✓ Vanilla ResNet50 loaded")

    # Try loading robust model
    try:
        from robustbench.utils import load_model
        robust = load_model(
            model_name='Salman2020Do_R50',
            dataset='imagenet',
            threat_model='Linf'
        ).to(device).eval()
        print("✓ Robust model (Salman2020Do_R50) loaded")
        has_robust = True
    except Exception as e:
        print(f"✗ Could not load robust model: {e}")
        has_robust = False
        robust = None

    # Create test data
    print("\n2. Creating test data...")
    test_images, test_labels = create_synthetic_data(n_samples=20)

    # Get model predictions as "ground truth" for synthetic data
    with torch.no_grad():
        vanilla_preds = vanilla_wrapped(test_images.to(device)).argmax(dim=1)
        test_labels = vanilla_preds.cpu()  # Use model's own predictions

    print(f"Test data shape: {test_images.shape}")

    # Evaluate vanilla model
    print("\n3. Evaluating Vanilla ResNet50...")
    print("-"*60)
    vanilla_results = run_compositional_evaluation(
        vanilla_wrapped, test_images, test_labels,
        env_severity=0.5, eps=0.01
    )

    # Evaluate robust model if available
    if has_robust and robust is not None:
        # Update labels for robust model
        with torch.no_grad():
            robust_preds = robust(test_images.to(device)).argmax(dim=1)
            robust_labels = robust_preds.cpu()

        print("\n4. Evaluating Robust Model...")
        print("-"*60)
        robust_results = run_compositional_evaluation(
            robust, test_images, robust_labels,
            env_severity=0.5, eps=0.01
        )
    else:
        robust_results = None

    # Save results
    print("\n5. Saving results...")
    results = {
        'vanilla': vanilla_results,
        'robust': robust_results if robust_results else {},
        'metadata': {
            'device': device,
            'n_samples': len(test_labels),
            'seed': SEED,
            'attack_library': 'foolbox',
            'note': 'CPU-compatible evaluation'
        }
    }

    with open('cpu_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved to cpu_evaluation_results.json")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nVanilla ResNet50:")
    for key, value in vanilla_results.items():
        print(f"  {key:25s}: {value*100:5.1f}%")

    if robust_results:
        print("\nRobust Model (Salman2020Do_R50):")
        for key, value in robust_results.items():
            print(f"  {key:25s}: {value*100:5.1f}%")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    if device == "cpu":
        print("\n⚠️ Note: Running on CPU. For faster evaluation and access to")
        print("   AutoAttack, please use a GPU-enabled environment like")
        print("   Google Colab with GPU runtime.")


if __name__ == "__main__":
    main()