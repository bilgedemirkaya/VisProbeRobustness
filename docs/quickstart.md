# Quickstart Guide

Get started with VisProbe in 5 minutes! This guide will help you test your first vision model for robustness.

## 🚀 Installation

```bash
pip install visprobe

# For adversarial attacks (optional)
pip install visprobe[adversarial]

# For all features
pip install visprobe[all]
```

## 📦 Basic Usage

### 1. Test with a Single Perturbation

```python
from visprobe import search
import torchvision.models as models
import torch

# Load your model
model = models.resnet50(weights='IMAGENET1K_V2')
model.eval()

# Prepare your test data (list of (image, label) tuples)
test_data = [
    (torch.randn(3, 224, 224), 0),  # Example image and label
    (torch.randn(3, 224, 224), 1),
    # ... add more samples (100 is a good start)
]

# Test robustness against Gaussian blur
report = search(
    model=model,
    data=test_data,
    perturbation="gaussian_blur",
    normalization="imagenet"
)

# View results
print(f"Robustness Score: {report.score:.1f}%")
print(f"Failure Threshold: {report.metrics['failure_threshold']:.3f}")
```

### 2. Test with Multiple Perturbations (Presets)

```python
from visprobe import search, list_presets

# See available presets
print(list_presets())

# Test with the "natural" preset (blur, noise, brightness, contrast)
report = search(
    model=model,
    data=test_data,
    preset="natural",
    normalization="imagenet"
)

# View comprehensive results
report.show()
```

## 🎯 Understanding the Results

### Robustness Score
- **90-100%**: Excellent robustness
- **70-89%**: Good robustness
- **50-69%**: Moderate robustness
- **Below 50%**: Poor robustness

### Failure Threshold
The exact perturbation level where your model starts failing. For example:
- Gaussian blur: σ=2.5 means the model fails when blur sigma exceeds 2.5
- Gaussian noise: 0.08 means the model fails when noise std exceeds 0.08

## 🔍 Available Perturbations

```python
from visprobe import list_perturbations, Perturbation

# List all available perturbations
list_perturbations()

# Use with named constants (IDE autocomplete!)
report = search(model, data, perturbation=Perturbation.GAUSSIAN_NOISE)
report = search(model, data, perturbation=Perturbation.MOTION_BLUR)
report = search(model, data, perturbation=Perturbation.BRIGHTNESS_INCREASE)
```

## 🎨 Available Presets

```python
from visprobe import list_presets

# See all presets with descriptions
list_presets(verbose=True)
```

Common presets:
- `"natural"` - Natural image corruptions (blur, noise, brightness)
- `"lighting"` - Lighting variations (brightness, contrast, gamma)
- `"corruption"` - Various corruptions (compression, noise, blur)
- `"geometric"` - Spatial transformations (rotation, scale, translation)
- `"adversarial"` - Adversarial attacks (requires extra dependencies)

## ⚙️ Advanced Options

### Search Methods

```python
# Adaptive search (default) - good for unknown ranges
report = search(model, data, perturbation="blur", search_method="adaptive")

# Binary search - efficient when you know the range
report = search(model, data, perturbation="blur", search_method="binary",
                level_lo=0.0, level_hi=5.0)

# Bayesian optimization - provides confidence intervals
report = search(model, data, perturbation="blur", search_method="bayesian")
```

### Custom Normalization

```python
# Use preset normalizations
report = search(model, data, perturbation="blur", normalization="imagenet")
report = search(model, data, perturbation="blur", normalization="cifar10")

# Or provide custom normalization
report = search(
    model, data,
    perturbation="blur",
    normalization={
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    }
)
```

## 💾 Saving and Loading Results

```python
# Save report
report.save("my_model_robustness.json")

# Load later
from visprobe import Report
report = Report.load("my_model_robustness.json")
```

## 🎬 Next Steps

1. **Explore Examples**: Check the [examples/](../examples/) directory
2. **Read User Guide**: Learn about [advanced features](user-guide.md)
3. **API Reference**: Detailed [API documentation](api-reference.md)
4. **Custom Perturbations**: Learn to [create your own](extending.md)

## 💡 Tips

- Start with 100 samples for quick iteration
- Use presets for comprehensive testing
- Save reports for comparison and tracking
- Test on representative data from your domain
- Consider your deployment scenario when choosing perturbations

## 🆘 Common Issues

### Out of Memory
- Reduce batch size in your model
- Use fewer samples initially
- Test on CPU if GPU memory is limited

### Slow Performance
- Start with fewer samples (50-100)
- Use binary search if you know the approximate range
- Disable gradient computation with `torch.no_grad()`

### Import Errors
- Ensure PyTorch is installed: `pip install torch torchvision`
- For adversarial attacks: `pip install visprobe[adversarial]`