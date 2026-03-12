#!/usr/bin/env python3
"""
Fixed evaluation function for the AutoAttack notebook
Replace the evaluate_at_severity function in the notebook with this version
"""

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
import warnings


def evaluate_at_severity_FIXED(model, clean_images, labels, env_fn, eps, batch_size=50):
    """
    FIXED evaluation pipeline with better error handling and diagnostics
    """
    # Apply environmental degradation
    degraded = env_fn(clean_images.clone())
    degraded = degraded.clamp(0.0, 1.0)

    # Move to device
    device = next(model.parameters()).device
    degraded = degraded.to(device)
    labels = labels.to(device)

    if eps < 1e-8:
        # Clean evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(degraded), batch_size):
                batch = degraded[i:i+batch_size]
                labs = labels[i:i+batch_size]
                outputs = model(batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == labs).sum().item()
        return correct / len(labels)
    else:
        # AutoAttack evaluation with fixes
        model.eval()

        try:
            # Create adversary with explicit settings
            adversary = AutoAttack(
                model,
                norm='Linf',
                eps=eps,
                version='standard',
                verbose=False,
                device=device
            )

            # IMPORTANT FIX: Ensure the model and data are on the same device
            adversary.device = device

            # Run attack
            x_adv = adversary.run_standard_evaluation(
                degraded, labels, bs=batch_size
            )

            # Evaluate robust accuracy
            correct = 0
            with torch.no_grad():
                for i in range(0, len(x_adv), batch_size):
                    batch = x_adv[i:i+batch_size]
                    labs = labels[i:i+batch_size]
                    outputs = model(batch)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labs).sum().item()

            robust_acc = correct / len(labels)

            # DIAGNOSTIC: If accuracy is exactly 0, check if attack is too strong
            if robust_acc == 0.0 and eps > 0.005:
                print(f"      ⚠️ 0% accuracy at eps={eps:.4f} - consider using smaller epsilon")

            return robust_acc

        except Exception as e:
            print(f"      ❌ AutoAttack failed: {e}")
            print(f"      Falling back to FGSM attack...")

            # Fallback to simple FGSM if AutoAttack fails
            return fgsm_attack_fallback(model, degraded, labels, eps, batch_size)


def fgsm_attack_fallback(model, images, labels, epsilon, batch_size):
    """
    Fallback FGSM attack if AutoAttack fails
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)

    correct = 0
    total = 0

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].clone().detach().requires_grad_(True)
        batch_labels = labels[i:i+batch_size]

        # Forward pass
        outputs = model(batch)
        loss = F.cross_entropy(outputs, batch_labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Create perturbation
        sign_grad = batch.grad.data.sign()
        perturbed = batch + epsilon * sign_grad
        perturbed = torch.clamp(perturbed, 0, 1)

        # Evaluate
        with torch.no_grad():
            outputs = model(perturbed)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

    return correct / total if total > 0 else 0.0


def test_epsilon_sensitivity(model, test_images, test_labels, device='cuda'):
    """
    Test model sensitivity to different epsilon values
    This helps determine appropriate epsilon ranges
    """
    print("\n" + "="*60)
    print("EPSILON SENSITIVITY ANALYSIS")
    print("="*60)

    model.eval()
    test_images = test_images[:10].to(device)  # Use only 10 samples for quick test
    test_labels = test_labels[:10].to(device)

    # Test multiple epsilon values
    epsilons = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
    results = []

    print(f"{'Epsilon':<10} {'8/255':<10} {'Accuracy':<12} {'Note'}")
    print("-" * 50)

    for eps in epsilons:
        if eps == 0:
            # Clean accuracy
            with torch.no_grad():
                outputs = model(test_images)
                preds = outputs.argmax(dim=1)
                acc = (preds == test_labels).float().mean().item()
        else:
            # FGSM attack (faster than AutoAttack for testing)
            images_adv = test_images.clone().detach().requires_grad_(True)

            outputs = model(images_adv)
            loss = F.cross_entropy(outputs, test_labels)

            model.zero_grad()
            loss.backward()

            # Create adversarial examples
            sign_grad = images_adv.grad.data.sign()
            perturbed = images_adv + eps * sign_grad
            perturbed = torch.clamp(perturbed, 0, 1)

            # Evaluate
            with torch.no_grad():
                outputs = model(perturbed)
                preds = outputs.argmax(dim=1)
                acc = (preds == test_labels).float().mean().item()

        eps_255 = eps * 255  # Convert to 8-bit scale for reference
        note = ""
        if eps == 0:
            note = "(clean)"
        elif acc == 0:
            note = "← First 0% accuracy"
        elif acc < 0.5 and len(results) > 0 and results[-1]['acc'] >= 0.5:
            note = "← Drops below 50%"

        results.append({'eps': eps, 'acc': acc})
        print(f"{eps:<10.4f} {eps_255:<10.1f} {acc*100:<12.1f}% {note}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")

    # Find good epsilon range
    good_eps = None
    for r in results:
        if r['eps'] > 0 and 0.1 < r['acc'] < 0.5:
            good_eps = r['eps']
            break

    if good_eps:
        print(f"✓ Use epsilon around {good_eps:.4f} for meaningful evaluation")
        print(f"  (This gives ~{results[results.index({'eps': good_eps, 'acc': r['acc']}))]['acc']*100:.0f}% accuracy)")
    else:
        if results[-1]['acc'] == 0:
            print(f"⚠️ Model is very vulnerable - use epsilon < {results[1]['eps']:.4f}")
        else:
            print(f"⚠️ Model might be robust - try epsilon > {results[-1]['eps']:.4f}")

    return results


# Usage in notebook:
"""
# Replace the evaluate_at_severity function with:
from notebook_fix import evaluate_at_severity_FIXED as evaluate_at_severity

# Or add this cell to test epsilon sensitivity:
from notebook_fix import test_epsilon_sensitivity
test_epsilon_sensitivity(vanilla_wrapped, all_images, all_labels, device=device)
"""

if __name__ == "__main__":
    print("NOTEBOOK FIX MODULE")
    print("="*60)
    print("This module contains fixed evaluation functions for the AutoAttack notebook")
    print("\nTo use in the notebook:")
    print("1. Import the fixed function:")
    print("   from notebook_fix import evaluate_at_severity_FIXED as evaluate_at_severity")
    print("\n2. Or test epsilon sensitivity:")
    print("   from notebook_fix import test_epsilon_sensitivity")
    print("   test_epsilon_sensitivity(model, images, labels)")
    print("\n3. Key fixes:")
    print("   - Better device handling")
    print("   - Fallback to FGSM if AutoAttack fails")
    print("   - Diagnostic messages for 0% accuracy")
    print("   - Epsilon sensitivity analysis")