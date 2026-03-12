# VisProbe

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Find robustness failures in your vision models in 5 minutes.**

VisProbe is a property-based testing framework for vision models. Instead of manually crafting test cases, you tell VisProbe what robustness properties matter (e.g., "predictions should be stable under blur") and it automatically finds where your model breaks.

📚 [Documentation](docs/) | 🚀 [Quickstart](docs/quickstart.md) | 📖 [User Guide](docs/user-guide.md) | 🔧 [API Reference](docs/api-reference.md) | 💡 [Examples](examples/)

## Installation

```bash
pip install visprobe

# For adversarial attacks (optional)
pip install adversarial-robustness-toolbox
```

## Quick Start

### Test with a preset (multiple perturbations)

```python
from visprobe import search

report = search(
    model=my_model,
    data=test_data,  # list of (image, label) tuples
    preset="natural",  # blur, noise, brightness, contrast, etc.
    normalization="imagenet",
)

print(f"Robustness Score: {report.score:.1f}%")
report.save()
```

### Test a specific perturbation

```python
from visprobe import search, Perturbation

report = search(
    model=my_model,
    data=test_data,
    perturbation=Perturbation.GAUSSIAN_BLUR,
    normalization="imagenet",
)

print(f"Model fails at blur σ={report.metrics['failure_threshold']:.2f}")
```

### Full example with ImageNet

```python
import torch
import torchvision.models as models
from visprobe import search

# Load model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Load your data (list of (image_tensor, label) tuples)
test_data = [...]  # 100 samples works well

# Find where model breaks under Gaussian blur
report = search(
    model=model,
    data=test_data,
    perturbation="gaussian_blur",
    normalization="imagenet",
)

print(f"Failure threshold: σ={report.metrics['failure_threshold']:.2f}")
print(f"Robustness score: {report.metrics['overall_robustness_score']:.1%}")
```

## Main Features

### 1. Presets for common scenarios

```python
from visprobe import list_presets, search

# See what's available
list_presets(verbose=True)

# Available presets:
# - natural: blur, noise, brightness, contrast
# - adversarial: FGSM, PGD attacks
# - comprehensive: everything (natural + adversarial)
# - lighting: brightness, contrast, gamma
# - corruption: JPEG compression, noise
```

### 2. Adaptive threshold search

VisProbe automatically finds the exact threshold where your model fails:

```python
# Finds failure point efficiently (typically 5-10 evaluations)
report = search(model, data, perturbation="gaussian_noise")

# Uses adaptive search by default, but you can choose:
report = search(model, data, perturbation="gaussian_blur",
                search_method="binary")  # or "adaptive", "bayesian"
```

### 3. Property-based testing

Define custom properties your model should satisfy:

```python
from visprobe.properties import ClassificationProperty

# Test if model maintains top-1 prediction
property = ClassificationProperty(
    name="top1_maintained",
    test_fn=lambda original, perturbed: original.argmax() == perturbed.argmax(),
    threshold=0.95  # 95% of samples should maintain prediction
)

report = search(model, data, perturbation="brightness", property=property)
```

### 4. Multiple search methods

- **Adaptive** (default): Fast, good for unknown ranges
- **Binary**: Efficient for known ranges
- **Bayesian**: Query-efficient, provides confidence intervals

```python
# Adaptive search (default)
report = search(model, data, perturbation="gaussian_noise",
                search_method="adaptive")

# Binary search for known ranges
report = search(model, data, perturbation="brightness",
                search_method="binary", level_lo=0.5, level_hi=1.5)

# Bayesian optimization for expensive evaluations
report = search(model, data, perturbation="pgd_attack",
                search_method="bayesian", n_iterations=10)
```

### 5. Detailed reporting

```python
# Rich information about failures
print(report)  # Summary statistics

# Access detailed metrics
print(f"Failure threshold: {report.metrics['failure_threshold']}")
print(f"Samples tested: {report.metrics['samples_tested']}")
print(f"Failure rate at threshold: {report.metrics['failure_rate']}")

# Save for later analysis
report.save("resnet50_blur_analysis.json")

# Interactive visualization
report.show()  # Opens browser dashboard
```

## Available Perturbations

```python
from visprobe import list_perturbations

list_perturbations()

# Vision perturbations:
# - gaussian_blur, motion_blur, defocus_blur
# - gaussian_noise, shot_noise, impulse_noise
# - brightness, contrast, saturation
# - rotation, translation, scale
# - jpeg_compression, pixelate

# Adversarial attacks:
# - fgsm, pgd, carlini_wagner
# - boundary_attack, hop_skip_jump
```

## Examples

### Testing multiple models

```python
models = {
    "ResNet50": models.resnet50(pretrained=True),
    "ViT": models.vit_b_16(pretrained=True),
    "EfficientNet": models.efficientnet_b0(pretrained=True),
}

for name, model in models.items():
    report = search(model, test_data, preset="natural")
    print(f"{name}: {report.score:.1f}%")
```

### Custom perturbation ranges

```python
from visprobe.strategies import GaussianNoiseStrategy

# Fine control over perturbation strength
report = search(
    model=model,
    data=test_data,
    strategy=lambda level: GaussianNoiseStrategy(std_dev=level),
    level_lo=0.0,
    level_hi=0.5,
    search_method="adaptive"
)
```

### CI/CD Integration

```python
# Run in test suite
def test_model_robustness():
    report = search(model, test_data, preset="natural")
    assert report.score > 70.0, f"Model too fragile: {report.score:.1f}%"
```

## Advanced Usage

### Compositional testing (multiple perturbations)

```python
from visprobe.strategies import CompositeStrategy

# Test blur + noise together
strategy = CompositeStrategy([
    GaussianBlurStrategy(kernel_size=5),
    GaussianNoiseStrategy(std_dev=0.1)
])

report = search(model, data, strategy=strategy)
```

### Custom properties

```python
def confidence_maintained(original_output, perturbed_output):
    """Check if confidence doesn't drop too much"""
    original_conf = torch.softmax(original_output, dim=-1).max()
    perturbed_conf = torch.softmax(perturbed_output, dim=-1).max()
    return perturbed_conf > original_conf * 0.8

property = ClassificationProperty(
    name="confidence_maintained",
    test_fn=confidence_maintained,
    threshold=0.9
)

report = search(model, data, perturbation="gaussian_blur", property=property)
```

## Advanced Analysis Module

VisProbe includes a comprehensive analysis module for deep insights into model behavior:

### Key Features

- **Per-sample tracking**: Know exactly which samples failed and why
- **Statistical confidence**: Bootstrap confidence intervals for rigorous comparison
- **Crossover detection**: Find where models exchange performance rankings
- **Disagreement analysis**: Identify complementary models for ensembles
- **Confidence profiling**: Detect overconfidence and calibration issues
- **Vulnerability analysis**: Find systematically vulnerable classes

### Quick Example

```python
from visprobe.analysis import (
    evaluate_detailed,
    bootstrap_accuracy,
    confidence_profile,
    class_vulnerability,
)

# Detailed evaluation with per-sample tracking
results = evaluate_detailed(model, images, labels, scenario="gaussian_noise")

# Get statistical confidence
acc, lower, upper = bootstrap_accuracy(results.correct_mask)
print(f"Accuracy: {acc:.1%} (95% CI: [{lower:.1%}, {upper:.1%}])")

# Check if model is overconfident
profile = confidence_profile(results.samples)
if profile.pct_high_confidence_errors > 30:
    print("⚠️ Model is overconfident on errors!")

# Find vulnerable classes
vulnerable = class_vulnerability(clean_results, noisy_results, top_k=5)
for vuln in vulnerable:
    print(f"{vuln.class_name}: {vuln.accuracy_drop:.0%} drop")
```

See [examples/advanced_analysis_example.py](examples/advanced_analysis_example.py) for comprehensive usage.

## Architecture

VisProbe follows a modular design:

- **Core**: Adaptive search engine that efficiently finds failure thresholds
- **Strategies**: Perturbation implementations (blur, noise, attacks, etc.)
- **Properties**: Invariants that models should maintain
- **Presets**: Common test scenarios
- **Report**: Rich analysis and visualization

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{visprobe2024,
  title={VisProbe: Property-Based Testing for Vision Models},
  author={...},
  year={2024},
  url={https://github.com/visprobe/visprobe}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.