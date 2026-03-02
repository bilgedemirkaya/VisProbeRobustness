# VisProbe Architecture

This document describes the internal architecture and design of VisProbe, helping developers understand the codebase structure and design decisions.

## 📁 Project Structure

```
visprobe/
├── __init__.py           # Main package exports
├── api.py                # Primary search() function and API
├── report.py             # Report class for results
├── perturbations.py      # Perturbation registry and discovery
├── presets.py            # Preset configurations
│
├── core/                 # Core functionality
│   ├── search_engine.py  # Adaptive search algorithms
│   ├── normalization.py  # Image normalization handling
│   └── utils.py          # Utility functions
│
├── strategies/           # Perturbation implementations
│   ├── base.py          # Abstract Strategy class
│   ├── image.py         # Legacy image strategies
│   ├── noise.py         # Noise perturbations
│   ├── blur.py          # Blur effects
│   ├── lighting.py      # Lighting changes
│   ├── spatial.py       # Geometric transformations
│   ├── adversarial.py   # Adversarial attacks
│   └── composition.py   # Strategy composition
│
├── properties/           # Property definitions
│   ├── base.py          # Abstract Property class
│   └── classification.py # Classification properties
│
├── config/              # Configuration
│   ├── presets/         # Preset YAML files
│   └── perturbations.yaml # Perturbation definitions
│
└── cli/                 # Command-line interface
    ├── cli.py           # CLI entry point
    └── dashboard.py     # Streamlit dashboard
```

## 🏗️ Core Components

### 1. Strategy Pattern

The framework uses the Strategy pattern for perturbations:

```python
class Strategy(ABC):
    """Abstract base class for all perturbation strategies."""

    @abstractmethod
    def apply(self, images: torch.Tensor) -> torch.Tensor:
        """Apply perturbation to images."""
        pass
```

Each perturbation type inherits from `Strategy`:

```python
class GaussianNoise(Strategy):
    def __init__(self, std_dev: float):
        self.std_dev = std_dev

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(images) * self.std_dev
        return images + noise
```

### 2. Property-Based Testing

Properties define what should hold true:

```python
class Property(ABC):
    """Abstract base class for robustness properties."""

    @abstractmethod
    def evaluate(self, original_output, perturbed_output) -> bool:
        """Check if property holds."""
        pass
```

Example property:

```python
class AccuracyPreservation(Property):
    def evaluate(self, original_output, perturbed_output):
        original_pred = original_output.argmax(dim=1)
        perturbed_pred = perturbed_output.argmax(dim=1)
        return (original_pred == perturbed_pred).float().mean() > 0.9
```

### 3. Search Engine

The `SearchEngine` implements adaptive search algorithms:

```python
class SearchEngine:
    def adaptive_search(self, evaluate_fn, lo, hi, tolerance):
        """Find failure threshold using adaptive search."""
        # Starts with coarse search
        # Refines around failure boundary
        # Returns minimal failing threshold
```

Key algorithms:
- **Adaptive Search**: Coarse-to-fine threshold discovery
- **Binary Search**: Efficient for known ranges
- **Bayesian Optimization**: Provides uncertainty estimates

### 4. Normalization Handler

Handles the critical denormalize → perturb → renormalize workflow:

```python
class NormalizationHandler:
    def denormalize(self, tensor):
        """Convert from normalized to [0,1] range"""

    def normalize(self, tensor):
        """Convert from [0,1] to normalized range"""

    def apply_perturbation(self, images, strategy):
        """Correctly apply perturbation with normalization"""
        # 1. Denormalize to [0,1]
        # 2. Apply perturbation
        # 3. Renormalize
```

## 🔄 Data Flow

```mermaid
graph TD
    A[User Input] --> B[API: search()]
    B --> C[Load Data & Model]
    C --> D[Filter Correct Samples]
    D --> E[SearchEngine]
    E --> F[Apply Perturbation]
    F --> G[Evaluate Property]
    G --> H{Property Holds?}
    H -->|Yes| I[Increase Strength]
    H -->|No| J[Decrease Strength]
    I --> E
    J --> E
    E --> K[Find Threshold]
    K --> L[Generate Report]
    L --> M[Return to User]
```

## 🎯 Design Principles

### 1. Modular and Extensible

Each component is independent and can be extended:
- New strategies inherit from `Strategy`
- New properties inherit from `Property`
- New search methods added to `SearchEngine`

### 2. Domain-Agnostic Core

The core is designed to work with any modality:
- `SearchEngine` doesn't know about images
- `Property` is abstract
- Strategies handle domain-specific logic

### 3. Declarative API

Users declare what they want, not how:
```python
# Declarative
search(model, data, perturbation="blur")

# Not imperative
strategy = GaussianBlur(sigma=2.0)
for level in range(0, 10):
    perturbed = strategy.apply(images, level)
    # ... manual search logic
```

### 4. Progressive Disclosure

Simple API for common cases, full control when needed:
```python
# Simple
search(model, data, preset="natural")

# Intermediate
search(model, data, perturbation="gaussian_noise", level_hi=0.15)

# Advanced
search(model, data,
       strategy=lambda l: CustomStrategy(l),
       property_fn=lambda o, p: custom_property(o, p))
```

## 🔌 Extension Points

### Adding a New Perturbation

1. Create strategy class in `strategies/`:
```python
class MyPerturbation(Strategy):
    def __init__(self, strength: float):
        self.strength = strength

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        # Implementation
        return perturbed_images
```

2. Register in `config/perturbations.yaml`:
```yaml
my_perturbation:
  name: "My Perturbation"
  strategy: "MyPerturbation"
  params:
    imagenet: {lo: 0.0, hi: 1.0}
    cifar10: {lo: 0.0, hi: 0.5}
```

3. Export from `strategies/__init__.py`

### Adding a New Preset

Create YAML file in `config/presets/`:
```yaml
name: "my_preset"
display_name: "My Custom Preset"
perturbations:
  - gaussian_noise
  - gaussian_blur
  - my_perturbation
threat_model: "custom_threat"
```

### Adding a New Property

```python
class MyProperty(Property):
    def evaluate(self, original_output, perturbed_output):
        # Custom evaluation logic
        return passes_property
```

## 🔧 Key Algorithms

### Adaptive Search

```python
def adaptive_search(evaluate_fn, lo, hi, tolerance):
    """
    Two-phase search:
    1. Coarse search to find approximate region
    2. Fine search to find exact threshold
    """
    # Phase 1: Coarse grid search
    coarse_points = np.linspace(lo, hi, num=10)
    for point in coarse_points:
        if not evaluate_fn(point):
            # Found failure region
            break

    # Phase 2: Binary search refinement
    return binary_search(evaluate_fn,
                         point - delta,
                         point + delta,
                         tolerance)
```

### Normalization-Aware Perturbation

```python
def apply_perturbation_with_normalization(images, strategy, norm_params):
    """
    Critical for correct perturbation application:
    1. Denormalize to [0,1] range
    2. Apply perturbation in [0,1] space
    3. Renormalize for model input
    """
    # Denormalize
    denormalized = (images * std + mean)

    # Apply perturbation
    perturbed = strategy.apply(denormalized)

    # Clip to valid range
    perturbed = torch.clamp(perturbed, 0, 1)

    # Renormalize
    normalized = (perturbed - mean) / std

    return normalized
```

## 🚀 Performance Considerations

### Caching
- Model outputs cached during search
- Perturbation results cached when possible

### Batching
- Process multiple samples in parallel
- Configurable batch size

### GPU Utilization
- Automatic device selection
- Memory-efficient operations

## 🔐 Security Considerations

- Input validation for all user inputs
- Safe YAML loading (no arbitrary code execution)
- Sandboxed strategy execution
- No network access in core library

## 📚 Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **torchvision**: Image transformations
- **NumPy**: Numerical operations

### Optional Dependencies
- **ART**: Adversarial attacks
- **Streamlit**: Dashboard UI
- **Plotly**: Interactive visualizations

## 🎓 Learning Resources

- [Strategy Pattern](https://en.wikipedia.org/wiki/Strategy_pattern)
- [Property-Based Testing](https://hypothesis.works/articles/what-is-property-based-testing/)
- [Adaptive Search Algorithms](https://en.wikipedia.org/wiki/Adaptive_algorithm)

## 🔮 Future Architecture Plans

- **Plugin System**: Dynamic loading of strategies
- **Distributed Testing**: Multi-GPU/multi-node support
- **Result Database**: SQLite backend for results
- **Web API**: REST API for cloud deployment
- **Multi-Modal Support**: Extend beyond vision (NLP, audio)