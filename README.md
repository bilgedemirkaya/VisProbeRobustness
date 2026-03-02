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

VisProbe doesn't just test fixed perturbation levels—it finds the exact threshold where your model starts failing:

```python
report = search(model, data, perturbation="gaussian_noise")

# Output:
# Failure threshold: 0.087
# This means: model is robust up to σ=0.087, fails beyond that
```

### 3. Adversarial attacks

Test against FGSM, PGD, and other attacks (requires `adversarial-robustness-toolbox`):

```python
from visprobe.strategies.adversarial import FGSMStrategy

report = search(
    model=model,
    data=test_data,
    strategy=lambda eps: FGSMStrategy(eps=eps),
    level_lo=0.0,
    level_hi=0.03,  # 8/255 is standard
    normalization="imagenet",
)
```

### 4. Automatic normalization handling

VisProbe handles the tricky denormalize→perturb→renormalize workflow:

```python
# Just tell it your normalization scheme
report = search(model, data, perturbation="blur", normalization="imagenet")
report = search(model, data, perturbation="blur", normalization="cifar10")

# Or use custom normalization
report = search(model, data, perturbation="blur",
                normalization={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]})
```

### 5. Multiple search strategies

```python
# Adaptive (default) - fast, good for unknown ranges
report = search(model, data, perturbation="blur", search_method="adaptive")

# Binary - efficient when you know the range
report = search(model, data, perturbation="blur", search_method="binary")

# Bayesian - query-efficient, gives confidence intervals
report = search(model, data, perturbation="blur", search_method="bayesian")
```

## CLI Usage

```bash
# Run tests and launch interactive dashboard
visprobe run test_my_model.py

# Visualize existing results
visprobe visualize test_my_model.py
```

## Available Perturbations

```python
from visprobe import list_perturbations
list_perturbations()
```

| Perturbation | Description |
|-------------|-------------|
| `gaussian_noise` | Additive Gaussian noise |
| `gaussian_blur` | Gaussian blur filter |
| `motion_blur` | Directional motion blur |
| `brightness_increase` | Increase brightness |
| `brightness_decrease` | Decrease brightness |
| `contrast_increase` | Increase contrast |
| `contrast_decrease` | Decrease contrast |
| `rotation` | Image rotation |
| `gamma_bright` | Gamma correction (brighter) |
| `gamma_dark` | Gamma correction (darker) |
| `jpeg_compression` | JPEG compression artifacts |

## How It Works

1. **Filter to correct samples**: Only tests samples your model classifies correctly
2. **Apply perturbations**: Gradually increases perturbation strength
3. **Find failure threshold**: Uses adaptive search to find where predictions flip
4. **Report results**: Tells you exactly how robust (or fragile) your model is

## Key Concepts

**Robustness Score**: Percentage of the perturbation range where model maintains correct predictions. Higher = more robust.

**Failure Threshold**: The perturbation level where model starts making mistakes. For blur, this might be σ=2.5. For noise, maybe 0.08.

**Pass Threshold**: By default, model "passes" a perturbation level if 90% of samples remain correct. Configurable via `pass_threshold` parameter.

## Tips

- **Start with 100 samples** - fast enough for iteration, representative enough for real insights
- **Use presets first** - they have sensible defaults for common scenarios
- **Check class distribution** - make sure your test samples cover diverse classes
- **Save reports** - call `report.save()` to persist results for the dashboard

## Requirements

- Python 3.9+
- PyTorch
- torchvision (for transforms)
- tqdm (for progress bars)
- adversarial-robustness-toolbox (optional, for adversarial attacks)

## License

MIT
