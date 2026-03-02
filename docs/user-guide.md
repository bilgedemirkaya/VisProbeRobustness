# User Guide

Comprehensive guide to using VisProbe for robustness testing of vision models.

## 📖 Table of Contents

1. [Core Concepts](#core-concepts)
2. [Basic Usage](#basic-usage)
3. [Working with Perturbations](#working-with-perturbations)
4. [Using Presets](#using-presets)
5. [Search Methods](#search-methods)
6. [Normalization](#normalization)
7. [Interpreting Results](#interpreting-results)
8. [Advanced Features](#advanced-features)
9. [Best Practices](#best-practices)
10. [Common Workflows](#common-workflows)

## Core Concepts

### What is Robustness Testing?

Robustness testing evaluates how well your model maintains correct predictions when inputs are slightly modified. VisProbe tests **accuracy preservation**: can your model still correctly classify images after applying realistic perturbations?

### Key Terms

- **Perturbation**: A transformation applied to an input (blur, noise, etc.)
- **Threshold**: The perturbation strength where the model starts failing
- **Robustness Score**: Percentage of perturbation range where model succeeds
- **Property**: A condition that should hold (e.g., "same prediction")
- **Preset**: A curated collection of perturbations

## Basic Usage

### Simple Test

```python
from visprobe import search
import torchvision.models as models

# Load model
model = models.resnet50(weights='IMAGENET1K_V2')
model.eval()

# Prepare data (list of (image, label) tuples)
test_data = load_your_test_data()  # Your function

# Test robustness
report = search(
    model=model,
    data=test_data,
    perturbation="gaussian_blur",
    normalization="imagenet"
)

# View results
print(f"Robustness Score: {report.score:.1f}%")
```

### Data Preparation

VisProbe expects data as a list of (image, label) tuples:

```python
# From PyTorch Dataset
dataset = YourDataset()
test_data = [(img, label) for img, label in dataset]

# From DataLoader
test_data = []
for batch_img, batch_label in dataloader:
    for img, label in zip(batch_img, batch_label):
        test_data.append((img, label.item()))

# Random example data
import torch
test_data = [(torch.randn(3, 224, 224), i) for i in range(100)]
```

## Working with Perturbations

### Available Perturbations

```python
from visprobe import list_perturbations

# See all available perturbations
perturbations = list_perturbations()
print(perturbations)
```

Categories:
- **Noise**: gaussian_noise, salt_pepper, uniform_noise
- **Blur**: gaussian_blur, motion_blur, defocus_blur
- **Lighting**: brightness, contrast, gamma, saturation
- **Geometric**: rotation, scale, translation, shear
- **Compression**: jpeg_compression

### Single Perturbation Testing

```python
# Test specific perturbation
report = search(
    model, data,
    perturbation="gaussian_noise",
    normalization="imagenet"
)
```

### Using Named Constants

```python
from visprobe import Perturbation

# Use constants for IDE autocomplete
report = search(
    model, data,
    perturbation=Perturbation.GAUSSIAN_BLUR
)
```

### Custom Parameters

```python
# Override default parameters
report = search(
    model, data,
    perturbation="gaussian_noise",
    level_lo=0.0,   # Start value
    level_hi=0.2,   # End value
    num_steps=20    # Search granularity
)
```

## Using Presets

### What are Presets?

Presets are curated collections of perturbations that test related robustness properties.

```python
from visprobe import list_presets

# See available presets
presets = list_presets()
print(presets)
# Output: {'natural': 'Natural perturbations...', 'lighting': '...', ...}
```

### Running Preset Tests

```python
# Test with multiple perturbations
report = search(
    model, data,
    preset="natural",  # Tests blur, noise, brightness, contrast
    normalization="imagenet"
)

# Results include all perturbations
report.show()
```

### Available Presets

| Preset | Description | Perturbations |
|--------|-------------|---------------|
| `natural` | Natural image corruptions | blur, noise, brightness, contrast |
| `lighting` | Lighting variations | brightness, contrast, gamma, saturation |
| `corruption` | Common corruptions | compression, noise, blur |
| `geometric` | Spatial transformations | rotation, scale, translation |
| `weather` | Weather conditions | fog, snow, rain |
| `comprehensive` | All perturbations | Everything |
| `adversarial` | Attack methods | FGSM, PGD (requires ART) |

## Search Methods

### Adaptive Search (Default)

Best for unknown parameter ranges. Starts coarse, refines around failures.

```python
report = search(
    model, data,
    perturbation="blur",
    search_method="adaptive"
)
```

### Binary Search

Efficient when you know the approximate range.

```python
report = search(
    model, data,
    perturbation="blur",
    search_method="binary",
    level_lo=0.0,
    level_hi=5.0,
    tolerance=0.01  # Precision
)
```

### Bayesian Optimization

Provides uncertainty estimates, good for expensive evaluations.

```python
report = search(
    model, data,
    perturbation="blur",
    search_method="bayesian",
    num_steps=50  # Number of evaluations
)
```

## Normalization

### Why Normalization Matters

Neural networks expect normalized inputs. VisProbe handles the denormalize → perturb → renormalize workflow automatically.

### Using Preset Normalization

```python
# ImageNet normalization
report = search(model, data, perturbation="blur", normalization="imagenet")

# CIFAR-10 normalization
report = search(model, data, perturbation="blur", normalization="cifar10")

# MNIST normalization
report = search(model, data, perturbation="blur", normalization="mnist")
```

### Custom Normalization

```python
# Specify your own normalization
report = search(
    model, data,
    perturbation="blur",
    normalization={
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    }
)
```

### No Normalization

```python
# If your data is already in [0, 1] range
report = search(
    model, data,
    perturbation="blur",
    normalization=None
)
```

## Interpreting Results

### Report Object

```python
# Access overall score
print(f"Robustness Score: {report.score}%")

# Get failure threshold
threshold = report.metrics['failure_threshold']
print(f"Model fails at: {threshold}")

# Sample-level results
print(f"Passed: {len(report.passed_samples)}")
print(f"Failed: {len(report.failed_samples)}")

# Detailed metrics
print(report.metrics)
```

### Understanding Scores

- **90-100%**: Excellent robustness
- **70-89%**: Good robustness
- **50-69%**: Moderate robustness
- **30-49%**: Poor robustness
- **0-29%**: Very poor robustness

### Failure Thresholds

What the threshold values mean:

| Perturbation | Low (Good) | Medium | High (Poor) |
|-------------|------------|---------|-------------|
| Gaussian Blur | σ > 3.0 | σ ≈ 1.5-3.0 | σ < 1.5 |
| Gaussian Noise | std > 0.15 | std ≈ 0.08-0.15 | std < 0.08 |
| Brightness | ±50% | ±25% | ±10% |
| Rotation | ±30° | ±15° | ±5° |

### Visualizing Results

```python
# Built-in visualization
report.show(detailed=True)

# Custom visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(report.metrics.keys(), report.metrics.values())
plt.title(f"Model Robustness: {report.score}%")
plt.xlabel("Metric")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Advanced Features

### Custom Strategies

```python
from visprobe.strategies import Strategy
import torch

class CustomPerturbation(Strategy):
    def __init__(self, strength: float):
        self.strength = strength

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        # Your perturbation logic
        return perturbed_images

# Use custom strategy
report = search(
    model, data,
    strategy=lambda s: CustomPerturbation(s),
    level_lo=0.0,
    level_hi=1.0
)
```

### Custom Properties

```python
def custom_property(original_output, perturbed_output, threshold=0.8):
    """Check if confidence stays above threshold"""
    original_conf = original_output.softmax(dim=1).max(dim=1).values
    perturbed_conf = perturbed_output.softmax(dim=1).max(dim=1).values
    return (perturbed_conf > threshold).float().mean() > 0.9

report = search(
    model, data,
    perturbation="blur",
    property_fn=custom_property
)
```

### Batch Processing

```python
# Control batch size for memory management
report = search(
    model, data,
    perturbation="blur",
    batch_size=16  # Smaller batches for large models
)
```

### Device Selection

```python
# Explicit device selection
report = search(
    model, data,
    perturbation="blur",
    device='cuda:0'  # Specific GPU
)

# CPU testing
report = search(
    model, data,
    perturbation="blur",
    device='cpu'
)
```

## Best Practices

### 1. Sample Selection

```python
# Use representative samples
# Bad: Random samples
test_data = random.sample(all_data, 100)

# Good: Stratified sampling
from sklearn.model_selection import train_test_split
_, test_data = train_test_split(
    all_data,
    test_size=100,
    stratify=labels,
    random_state=42
)
```

### 2. Start Small

```python
# Quick iteration with few samples
report = search(model, data[:20], perturbation="blur", num_steps=5)

# Then scale up
if report.score > 80:
    full_report = search(model, data, perturbation="blur", num_steps=20)
```

### 3. Save Results

```python
# Save for comparison
report.save("model_v1_results.json")

# Load and compare
from visprobe import Report
old_report = Report.load("model_v1_results.json")
print(f"Improvement: {new_report.score - old_report.score}%")
```

### 4. Test Incrementally

```python
# Test individual perturbations first
perturbations = ["gaussian_noise", "gaussian_blur", "brightness"]
results = {}

for pert in perturbations:
    report = search(model, data, perturbation=pert)
    results[pert] = report.score
    print(f"{pert}: {report.score}%")

# Then test comprehensive
full_report = search(model, data, preset="natural")
```

## Common Workflows

### Model Comparison

```python
models = {
    'ResNet18': models.resnet18(pretrained=True),
    'ResNet50': models.resnet50(pretrained=True),
    'EfficientNet': models.efficientnet_b0(pretrained=True)
}

results = {}
for name, model in models.items():
    model.eval()
    report = search(model, test_data, preset="natural")
    results[name] = report.score
    print(f"{name}: {report.score:.1f}%")

# Find best model
best_model = max(results, key=results.get)
print(f"Most robust: {best_model}")
```

### Progressive Training Validation

```python
# Test model at different training checkpoints
checkpoints = ['epoch_10.pth', 'epoch_20.pth', 'epoch_30.pth']

for checkpoint in checkpoints:
    model.load_state_dict(torch.load(checkpoint))
    report = search(model, val_data, preset="natural")
    print(f"{checkpoint}: {report.score}%")
```

### Deployment Readiness

```python
def check_deployment_readiness(model, data, min_score=75):
    """Check if model is ready for deployment"""

    # Test critical perturbations
    critical_tests = {
        'lighting': 'natural',  # For varying conditions
        'compression': 'jpeg_compression',  # For transmission
        'noise': 'gaussian_noise'  # For sensor noise
    }

    ready = True
    for test_name, perturbation in critical_tests.items():
        report = search(model, data, perturbation=perturbation)
        print(f"{test_name}: {report.score}%")

        if report.score < min_score:
            ready = False
            print(f"  ⚠️ Below threshold ({min_score}%)")

    return ready

is_ready = check_deployment_readiness(model, test_data)
```

### Robustness Report Generation

```python
import json
from datetime import datetime

def generate_robustness_report(model, data, output_file="report.json"):
    """Generate comprehensive robustness report"""

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model.__class__.__name__,
        'samples': len(data),
        'tests': {}
    }

    # Test multiple presets
    presets = ['natural', 'lighting', 'corruption']

    for preset in presets:
        report = search(model, data, preset=preset)
        results['tests'][preset] = {
            'score': report.score,
            'metrics': report.metrics
        }

    # Calculate overall score
    scores = [r['score'] for r in results['tests'].values()]
    results['overall_score'] = sum(scores) / len(scores)

    # Save report
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Report saved to {output_file}")
    print(f"Overall robustness: {results['overall_score']:.1f}%")

    return results

report = generate_robustness_report(model, test_data)
```

## Tips and Tricks

### Memory Optimization

```python
# Clear GPU cache between tests
import torch
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
with torch.no_grad():
    report = search(model, data, perturbation="blur")
```

### Speed Optimization

```python
# Use binary search for known ranges
report = search(
    model, data,
    perturbation="blur",
    search_method="binary",
    level_lo=0, level_hi=5
)

# Reduce search steps for quick checks
report = search(
    model, data,
    perturbation="blur",
    num_steps=5  # Fewer evaluations
)
```

### Debugging

```python
# Enable verbose mode
report = search(
    model, data,
    perturbation="blur",
    verbose=True
)

# Check search history
for step in report.search_history:
    print(f"Level: {step['level']}, Passed: {step['passed']}")
```

## Next Steps

- Explore [examples/](../examples/) for real-world usage
- Read [API Reference](api-reference.md) for detailed documentation
- Learn about [extending VisProbe](extending.md) with custom components
- Check [architecture guide](architecture.md) for deep understanding