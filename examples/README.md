# VisProbe Examples

This directory contains example scripts demonstrating how to use VisProbe for testing vision model robustness.

## 📚 Available Examples

### 1. [basic_example.py](basic_example.py)
**Simplest usage example**

Shows the most basic way to use VisProbe with a pretrained model.

```python
from visprobe import search

report = search(model, data, perturbation="gaussian_blur")
print(f"Robustness: {report.score}%")
```

**What you'll learn:**
- Basic API usage
- Running single perturbation tests
- Interpreting results

---

### 2. [simple_api_example.py](simple_api_example.py)
**Demonstrates the simplified API**

Shows how the new simplified API makes testing easier compared to manual strategy construction.

```python
# Simple API (new)
report = search(model, data, perturbation="gaussian_noise")

# vs. Complex API (old)
report = search(model, data,
                strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
                level_lo=0.0, level_hi=0.15)
```

**What you'll learn:**
- Simplified perturbation API
- Using named perturbations
- Automatic parameter selection

---

### 3. [preset_comparison.py](preset_comparison.py)
**Compare multiple presets on the same model**

Tests a model against different preset collections to get comprehensive robustness assessment.

```python
presets = ["natural", "lighting", "corruption", "geometric"]
for preset in presets:
    report = search(model, data, preset=preset)
    print(f"{preset}: {report.score}%")
```

**What you'll learn:**
- Using presets for comprehensive testing
- Comparing robustness across threat models
- Batch testing workflows

---

### 4. [cifar10_example.py](cifar10_example.py)
**CIFAR-10 specific example**

Shows how to test models trained on CIFAR-10 with appropriate normalization and parameters.

```python
# CIFAR-10 specific normalization
transform = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010]
)

report = search(model, data,
                preset="natural",
                normalization="cifar10")
```

**What you'll learn:**
- Dataset-specific normalization
- Loading CIFAR-10 data
- Appropriate parameter ranges for CIFAR-10

---

### 5. [custom_model_example.py](custom_model_example.py)
**Custom model integration**

Demonstrates how to test your own custom models, including proper data preparation and normalization handling.

```python
class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture

# Test custom model
model = MyCustomModel()
model.load_state_dict(torch.load('checkpoint.pth'))

report = search(model, custom_data,
                normalization=custom_normalization)
```

**What you'll learn:**
- Testing custom models
- Preparing custom datasets
- Handling custom normalization schemes
- Integrating with existing training pipelines

---

## 🚀 Running the Examples

### Basic Setup

1. **Install VisProbe:**
```bash
pip install visprobe
```

2. **Run an example:**
```bash
python examples/basic_example.py
```

### With Custom Data

Most examples use random data for demonstration. To use real data:

```python
# Replace this:
test_data = [(torch.randn(3, 224, 224), i) for i in range(100)]

# With this:
from your_dataset import load_test_data
test_data = load_test_data()  # Returns list of (image, label) tuples
```

### Common Modifications

#### Change Model
```python
# Instead of pretrained
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')

# Use your model
model = YourModel()
model.load_state_dict(torch.load('your_model.pth'))
```

#### Change Perturbation
```python
# Single perturbation
report = search(model, data, perturbation="motion_blur")

# Or use preset
report = search(model, data, preset="adversarial")
```

#### Adjust Parameters
```python
report = search(
    model, data,
    perturbation="gaussian_noise",
    level_lo=0.0,      # Start from no noise
    level_hi=0.2,      # Max noise level
    tolerance=0.001,   # Higher precision
    batch_size=16      # Smaller batches for memory
)
```

## 📊 Understanding Results

Each example produces a `Report` object with:

```python
# Overall robustness score (0-100%)
print(f"Score: {report.score}%")

# Failure threshold
print(f"Fails at: {report.metrics['failure_threshold']}")

# Sample-level results
print(f"Passed: {len(report.passed_samples)} samples")
print(f"Failed: {len(report.failed_samples)} samples")

# Detailed view
report.show()
```

## 🎯 Example Use Cases

### Model Selection
Compare different architectures:
```python
models = [resnet18(), resnet50(), efficientnet()]
for model in models:
    report = search(model, data, preset="natural")
    print(f"{model.__class__.__name__}: {report.score}%")
```

### Hyperparameter Tuning
Test models trained with different augmentations:
```python
for checkpoint in ['model_v1.pth', 'model_v2.pth', 'model_v3.pth']:
    model.load_state_dict(torch.load(checkpoint))
    report = search(model, data, preset="natural")
    save_results(checkpoint, report)
```

### Deployment Validation
Test before deploying:
```python
# Test against expected production perturbations
production_preset = "lighting"  # If deploying to variable lighting
report = search(model, production_data, preset=production_preset)

if report.score < 80:
    print("⚠️ Model may not be robust enough for production")
```

## 💡 Tips and Tricks

### Memory Management
```python
# For large models or limited GPU memory
report = search(
    model, data,
    batch_size=8,        # Smaller batches
    device='cpu'         # Use CPU if needed
)
```

### Speed Optimization
```python
# For quick iteration
report = search(
    model,
    data[:50],           # Use fewer samples initially
    search_method='binary',  # Faster if you know the range
    num_steps=5          # Fewer search steps
)
```

### Debugging
```python
# Enable verbose output
report = search(
    model, data,
    perturbation="gaussian_blur",
    verbose=True         # See detailed progress
)

# Check intermediate results
print(report.search_history)  # See search progression
```

## 📚 Advanced Examples (Coming Soon)

- **Adversarial robustness**: Testing against attacks
- **Composed perturbations**: Multiple perturbations together
- **Custom properties**: Define your own test criteria
- **Distributed testing**: Multi-GPU evaluation
- **CI/CD integration**: Automated robustness testing

## 🤝 Contributing Examples

We welcome new examples! If you have an interesting use case:

1. Create a new example file
2. Add clear documentation
3. Include expected output
4. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## 📖 Further Resources

- [Documentation](../docs/)
- [API Reference](../docs/api-reference.md)
- [Architecture Guide](../docs/architecture.md)
- [GitHub Issues](https://github.com/bilgedemirkaya/VisProbe/issues)