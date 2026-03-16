# VisProbe Workflows

High-level APIs for common robustness testing patterns.

## Overview

The `workflows` module provides pre-built experimental workflows that handle common robustness testing scenarios:

- **Severity sweeps**: Evaluate models across perturbation intensity levels
- **Compositional testing**: Environmental perturbations + adversarial attacks
- **Multi-model comparisons**: Compare robustness across model architectures
- **Automated metrics**: AUC computation, robustness curves, etc.

## Quick Start

### Basic Severity Sweep

Test a model across different perturbation levels:

```python
from visprobe.workflows import run_severity_sweep
from visprobe.strategies import gaussian_blur_severity

# Run severity sweep with Gaussian blur
results = run_severity_sweep(
    model=my_model,
    images=test_images,
    labels=test_labels,
    strategy=gaussian_blur_severity(sigma_max=3.0),
    severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)

# Compute AUC
from visprobe.workflows import compute_auc
accuracies = [r.accuracy for r in results]
auc = compute_auc(severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], accuracies=accuracies)
print(f"Robustness AUC: {auc:.3f}")
```

### Multi-Model Comparison

Compare multiple models on the same perturbation:

```python
from visprobe.workflows import SeveritySweep
from visprobe.strategies import gaussian_noise_severity

sweep = SeveritySweep(
    strategy=gaussian_noise_severity(std_max=0.1, seed=42),
    severities=[0.0, 0.5, 1.0],
)

models = {
    "ResNet50": resnet_model,
    "ViT-B": vit_model,
    "Swin-B": swin_model,
}

results = sweep.run_multi_model(models, test_images, test_labels)

# Output:
#   ResNet50                      95.0% -> 68.0%  AUC=0.833
#   ViT-B                         93.0% -> 72.0%  AUC=0.845
#   Swin-B                        94.0% -> 75.0%  AUC=0.862
```

### Compositional Testing

Test robustness under combined environmental + adversarial perturbations:

```python
from visprobe.workflows import CompositionalTest
from visprobe.strategies import lowlight_severity
from autoattack import AutoAttack

# Define attack function
def run_autoattack(model, images, labels, eps):
    if eps < 1e-8:
        return images
    aa = AutoAttack(model, norm="Linf", eps=eps, version="standard")
    return aa.run_standard_evaluation(images, labels, bs=50)

# Create compositional test
test = CompositionalTest(
    env_strategy=lowlight_severity(max_reduction=0.7),
    attack_fn=run_autoattack,
    eps_fn=lambda s: 0.01 * s,  # Scale attack strength with severity
)

# Run test
results = test.run(model, images, labels, model_name="RobustModel")
```

## API Reference

### Severity Sweep

**`SeveritySweep`**

High-level API for running severity sweeps.

```python
from visprobe.workflows import SeveritySweep

sweep = SeveritySweep(
    strategy=perturbation_strategy,
    severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    batch_size=50,
    device="cuda",
    show_progress=True,
)

# Single model
results = sweep.run(model, images, labels)

# Multiple models
all_results = sweep.run_multi_model(models_dict, images, labels)

# Compute AUC
auc = sweep.compute_auc(results)
```

**`run_severity_sweep()`**

Convenience function for one-off sweeps:

```python
from visprobe.workflows import run_severity_sweep

results = run_severity_sweep(
    model=my_model,
    images=test_images,
    labels=test_labels,
    strategy=gaussian_blur_severity(sigma_max=3.0),
    severities=[0.0, 0.5, 1.0],
)
```

### Compositional Testing

**`CompositionalTest`**

Combines environmental perturbations with adversarial attacks:

```python
from visprobe.workflows import CompositionalTest

test = CompositionalTest(
    env_strategy=environment_perturbation,
    attack_fn=adversarial_attack_function,
    eps_fn=severity_to_eps_mapping,
    severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)

results = test.run(model, images, labels)
```

**Attack Function Signature:**
```python
def attack_fn(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Apply adversarial attack.

    Args:
        model: Model under attack
        images: Input images (already env-perturbed)
        labels: Ground truth labels
        eps: Attack strength

    Returns:
        Adversarially perturbed images
    """
    ...
```

**`run_compositional_sweep()`**

Convenience function:

```python
results = run_compositional_sweep(
    model=my_model,
    images=test_images,
    labels=test_labels,
    env_strategy=lowlight_severity(max_reduction=0.7),
    attack_fn=autoattack_wrapper,
    eps_fn=lambda s: 0.01 * s,
)
```

### Metrics

**`compute_auc(severities, accuracies)`**

Compute Area Under Curve for robustness:

```python
from visprobe.workflows import compute_auc

severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
accuracies = [0.95, 0.92, 0.88, 0.82, 0.75, 0.68]

auc = compute_auc(severities, accuracies)
print(f"AUC: {auc:.3f}")  # Higher is better
```

**`compute_robustness_curve(results, metric)`**

Extract metric curve from results:

```python
from visprobe.workflows import compute_robustness_curve

acc_curve = compute_robustness_curve(results, metric="accuracy")
conf_curve = compute_robustness_curve(results, metric="mean_confidence")
loss_curve = compute_robustness_curve(results, metric="mean_loss")
```

## Common Patterns

### Pattern 1: Baseline vs Robust Model Comparison

```python
from visprobe.workflows import SeveritySweep
from visprobe.strategies import gaussian_noise_severity

sweep = SeveritySweep(
    strategy=gaussian_noise_severity(std_max=0.15, seed=42),
    severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)

models = {
    "Vanilla_ResNet50": vanilla_model,
    "Robust_ResNet50": robust_model,
}

results = sweep.run_multi_model(models, images, labels, scenario="noise_robustness")

# Compare AUCs
for name, res in results.items():
    auc = sweep.compute_auc(res)
    print(f"{name}: AUC={auc:.3f}")
```

### Pattern 2: Multi-Scenario Testing

```python
from visprobe.strategies import (
    gaussian_blur_severity,
    gaussian_noise_severity,
    lowlight_severity,
)

scenarios = {
    "blur": gaussian_blur_severity(sigma_max=3.0),
    "noise": gaussian_noise_severity(std_max=0.1, seed=42),
    "lowlight": lowlight_severity(max_reduction=0.7),
}

all_results = {}
for scenario_name, strategy in scenarios.items():
    sweep = SeveritySweep(strategy=strategy)
    results = sweep.run(model, images, labels, scenario=scenario_name)
    all_results[scenario_name] = results
    print(f"{scenario_name}: AUC={sweep.compute_auc(results):.3f}")
```

### Pattern 3: Protection Gap Analysis

Measure how much adversarial robustness degrades under environmental perturbations:

```python
from visprobe.workflows import CompositionalTest, SeveritySweep

# Pure adversarial robustness
pure_adv = evaluate_detailed(model, attacked_images, labels)

# Compositional robustness (environmental + adversarial)
comp_test = CompositionalTest(
    env_strategy=lowlight_severity(max_reduction=0.7),
    attack_fn=autoattack,
    eps_fn=lambda s: 0.01,  # Fixed attack strength
)
comp_results = comp_test.run(model, images, labels)

# Compute protection gap
adv_advantage = robust_model_acc - vanilla_model_acc  # On pure attacks
comp_advantage = compute_auc([...])  # On compositional
protection_gap = (adv_advantage - comp_advantage) / adv_advantage
print(f"Protection Gap: {protection_gap * 100:.1f}%")
```

### Pattern 4: Ablation Studies

```python
# Fixed environmental perturbation, sweep attack strength
fixed_env = lowlight_severity(max_reduction=0.7)
env_images = fixed_env.generate(images, level=1.0)  # Max severity

attack_strengths = [0.0, 0.005, 0.01, 0.015, 0.02]
results = []

for eps in attack_strengths:
    attacked = run_autoattack(model, env_images, labels, eps)
    result = evaluate_detailed(model, attacked, labels, eps=eps)
    results.append(result)

# Plot: attack strength vs accuracy
```

## Integration with Analysis Module

Workflows module works seamlessly with the analysis module:

```python
from visprobe.workflows import run_severity_sweep
from visprobe.analysis import (
    bootstrap_accuracy,
    confidence_profile,
    expected_calibration_error,
)

# Run sweep
results = run_severity_sweep(model, images, labels, strategy=...)

# Analyze each severity level
for result in results:
    # Bootstrap confidence intervals
    acc, lo, hi = bootstrap_accuracy(result.correct_mask)
    print(f"Accuracy: {acc:.1%} [{lo:.1%}, {hi:.1%}]")

    # Confidence calibration
    prof = confidence_profile(result.samples)
    ece = expected_calibration_error(result.samples)
    print(f"ECE: {ece:.3f}")
```

## Best Practices

### 1. Use Consistent Seeds

For reproducibility, always set seeds:

```python
sweep = SeveritySweep(
    strategy=gaussian_noise_severity(std_max=0.1, seed=42),
    ...
)
```

### 2. Choose Appropriate Severity Levels

```python
# Coarse sweep for exploration
coarse = [0.0, 0.5, 1.0]

# Fine sweep for publication
fine = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Adaptive (fewer points where curve is flat)
adaptive = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
```

### 3. Batch Size Selection

```python
# GPU memory limited
sweep = SeveritySweep(..., batch_size=32)

# Fast evaluation
sweep = SeveritySweep(..., batch_size=128)

# Default (balanced)
sweep = SeveritySweep(..., batch_size=50)
```

### 4. Progress Monitoring

```python
# Show progress (default)
sweep = SeveritySweep(..., show_progress=True)

# Silent (for automated scripts)
sweep = SeveritySweep(..., show_progress=False)
```

## Examples

See the `examples/` directory for complete notebooks:

- `examples/severity_sweep_tutorial.ipynb` - Basic severity sweep walkthrough
- `examples/compositional_testing.ipynb` - Environmental + adversarial testing
- `examples/multi_model_comparison.ipynb` - Comparing model robustness
- `examples/protection_gap_analysis.ipynb` - Measuring protection gaps

## See Also

- **Severity Utilities**: `visprobe.strategies.severity` - Pre-configured severity-mapped strategies
- **Analysis Module**: `visprobe.analysis` - Statistical analysis tools
- **Strategies Module**: `visprobe.strategies` - Perturbation implementations
