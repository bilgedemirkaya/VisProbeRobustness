# API Reference

Complete API reference for VisProbe functions, classes, and modules.

## Core Functions

### `search()`

The main entry point for robustness testing.

```python
def search(
    model: torch.nn.Module,
    data: List[Tuple[torch.Tensor, int]],
    preset: Optional[str] = None,
    perturbation: Optional[str] = None,
    strategy: Optional[Callable] = None,
    property_fn: Optional[Callable] = None,
    normalization: Optional[Union[str, Dict]] = "imagenet",
    search_method: str = "adaptive",
    level_lo: float = None,
    level_hi: float = None,
    num_steps: int = 10,
    tolerance: float = 0.01,
    pass_threshold: float = 0.9,
    batch_size: int = 32,
    device: str = None,
    verbose: bool = True
) -> Report
```

#### Parameters

- **model** (`torch.nn.Module`): PyTorch model to test
- **data** (`List[Tuple[torch.Tensor, int]]`): List of (image, label) tuples
- **preset** (`str`, optional): Name of preset to use (e.g., "natural", "adversarial")
- **perturbation** (`str`, optional): Single perturbation to test (e.g., "gaussian_blur")
- **strategy** (`Callable`, optional): Custom strategy function for advanced usage
- **property_fn** (`Callable`, optional): Custom property function
- **normalization** (`str | Dict`): Normalization scheme ("imagenet", "cifar10", or custom dict)
- **search_method** (`str`): Search algorithm ("adaptive", "binary", "bayesian")
- **level_lo** (`float`): Lower bound for perturbation strength
- **level_hi** (`float`): Upper bound for perturbation strength
- **num_steps** (`int`): Number of search steps
- **tolerance** (`float`): Search tolerance for threshold
- **pass_threshold** (`float`): Fraction of samples that must pass (default 0.9)
- **batch_size** (`int`): Batch size for processing
- **device** (`str`): Device to use ("cuda", "cpu", or None for auto)
- **verbose** (`bool`): Print progress information

#### Returns

- **Report**: Report object containing results

#### Examples

```python
# Basic usage with preset
report = search(model, data, preset="natural")

# Single perturbation
report = search(model, data, perturbation="gaussian_noise")

# Custom parameters
report = search(
    model, data,
    perturbation="gaussian_blur",
    level_lo=0.0,
    level_hi=5.0,
    search_method="binary",
    tolerance=0.001
)
```

---

### `list_presets()`

List all available presets.

```python
def list_presets(verbose: bool = False) -> Union[Dict[str, str], None]
```

#### Parameters

- **verbose** (`bool`): If True, print detailed information

#### Returns

- **Dict[str, str]** or **None**: Dictionary of preset names and descriptions (if verbose=False)

#### Example

```python
# Get preset dictionary
presets = list_presets()
print(presets.keys())  # ['natural', 'adversarial', ...]

# Print detailed info
list_presets(verbose=True)
```

---

### `list_perturbations()`

List all available perturbations.

```python
def list_perturbations() -> List[str]
```

#### Returns

- **List[str]**: List of perturbation names

#### Example

```python
perturbations = list_perturbations()
print(perturbations)  # ['gaussian_noise', 'gaussian_blur', ...]
```

---

### `get_perturbation()`

Get detailed information about a perturbation.

```python
def get_perturbation(name: str) -> Dict[str, Any]
```

#### Parameters

- **name** (`str`): Perturbation name

#### Returns

- **Dict**: Perturbation specification including parameters

---

### `get_preset_info()`

Get detailed information about a preset.

```python
def get_preset_info(name: str) -> Dict[str, Any]
```

#### Parameters

- **name** (`str`): Preset name

#### Returns

- **Dict**: Preset configuration

---

## Classes

### `Report`

Results from robustness testing.

```python
class Report:
    def __init__(self, data: Dict[str, Any])
```

#### Attributes

- **score** (`float`): Overall robustness score (0-100)
- **metrics** (`Dict`): Detailed metrics including failure thresholds
- **passed_samples** (`List`): Indices of samples that passed
- **failed_samples** (`List`): Indices of samples that failed
- **search_history** (`List`): Search progression history

#### Methods

##### `show()`

Display formatted report.

```python
def show(self, detailed: bool = False) -> None
```

##### `save()`

Save report to file.

```python
def save(self, filepath: str = None) -> str
```

##### `load()`

Load report from file (class method).

```python
@classmethod
def load(cls, filepath: str) -> Report
```

##### `to_dict()`

Convert to dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

#### Example

```python
# Run test
report = search(model, data, preset="natural")

# Access results
print(f"Score: {report.score}%")
print(f"Failure threshold: {report.metrics['failure_threshold']}")

# Save and load
report.save("results.json")
loaded_report = Report.load("results.json")
```

---

### `Strategy`

Base class for all perturbation strategies.

```python
class Strategy(ABC):
    @abstractmethod
    def apply(self, images: torch.Tensor) -> torch.Tensor:
        """Apply perturbation to images."""
        pass
```

#### Implementing Custom Strategies

```python
from visprobe.strategies import Strategy

class MyCustomNoise(Strategy):
    def __init__(self, intensity: float):
        self.intensity = intensity

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(images) * self.intensity
        return torch.clamp(images + noise, 0, 1)

# Use with search
report = search(
    model, data,
    strategy=lambda level: MyCustomNoise(level),
    level_lo=0.0,
    level_hi=1.0
)
```

---

### `Property`

Base class for robustness properties.

```python
class Property(ABC):
    @abstractmethod
    def evaluate(self,
                 original_outputs: torch.Tensor,
                 perturbed_outputs: torch.Tensor,
                 labels: torch.Tensor) -> float:
        """Evaluate if property holds."""
        pass
```

#### Implementing Custom Properties

```python
from visprobe.properties import Property

class Top5Preservation(Property):
    def evaluate(self, original_outputs, perturbed_outputs, labels):
        orig_top5 = original_outputs.topk(5, dim=1).indices
        pert_top5 = perturbed_outputs.topk(5, dim=1).indices

        # Check if original prediction stays in top-5
        preserved = 0
        for i in range(len(orig_top5)):
            if orig_top5[i, 0] in pert_top5[i]:
                preserved += 1

        return preserved / len(orig_top5)

# Use with search
report = search(
    model, data,
    perturbation="gaussian_noise",
    property_fn=Top5Preservation()
)
```

---

### `Perturbation` (Enum)

Named constants for perturbations with IDE autocomplete.

```python
class Perturbation:
    GAUSSIAN_NOISE = "gaussian_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"
    BRIGHTNESS_INCREASE = "brightness_increase"
    BRIGHTNESS_DECREASE = "brightness_decrease"
    CONTRAST_INCREASE = "contrast_increase"
    CONTRAST_DECREASE = "contrast_decrease"
    ROTATION = "rotation"
    # ... and more
```

#### Example

```python
from visprobe import search, Perturbation

# Use with autocomplete
report = search(model, data, perturbation=Perturbation.GAUSSIAN_BLUR)
```

---

### `NormalizationHandler`

Handles image normalization and denormalization.

```python
class NormalizationHandler:
    def __init__(self, mean: List[float], std: List[float])
```

#### Methods

##### `normalize()`

```python
def normalize(self, tensor: torch.Tensor) -> torch.Tensor
```

##### `denormalize()`

```python
def denormalize(self, tensor: torch.Tensor) -> torch.Tensor
```

##### `apply_perturbation_with_normalization()`

```python
def apply_perturbation_with_normalization(
    self,
    images: torch.Tensor,
    strategy: Strategy,
    level: float
) -> torch.Tensor
```

---

## Module Reference

### `visprobe.strategies`

Perturbation strategy implementations.

#### Available Strategies

**Noise Strategies:**
- `GaussianNoise(std_dev: float)`
- `SaltPepperNoise(amount: float)`
- `UniformNoise(max_val: float)`
- `SpeckleNoise(variance: float)`

**Blur Strategies:**
- `GaussianBlur(sigma: float)`
- `MotionBlur(kernel_size: int, angle: float)`
- `DefocusBlur(radius: float)`
- `BoxBlur(kernel_size: int)`

**Lighting Strategies:**
- `Brightness(factor: float)`
- `Contrast(factor: float)`
- `Gamma(gamma: float)`
- `Saturation(factor: float)`

**Spatial Strategies:**
- `Rotation(angle: float)`
- `Scale(factor: float)`
- `Translation(dx: float, dy: float)`
- `Shear(shear_x: float, shear_y: float)`

**Adversarial Strategies** (requires `adversarial-robustness-toolbox`):
- `FGSMStrategy(eps: float)`
- `PGDStrategy(eps: float, eps_step: float, max_iter: int)`
- `BIMStrategy(eps: float, eps_step: float, max_iter: int)`

---

### `visprobe.properties`

Property definitions for testing.

#### Available Properties

- `AccuracyPreservation`: Original prediction maintained
- `ConfidenceThreshold`: Confidence above threshold
- `TopKPreservation`: Prediction in top-K

---

### `visprobe.presets`

Preset configurations.

#### Functions

- `list_available_presets()`: Get all presets
- `load_preset(name: str)`: Load preset configuration
- `requires_art(preset_name: str)`: Check if preset needs ART

---

## Configuration

### Normalization Presets

```python
NORMALIZATION_PRESETS = {
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010]
    },
    # ... more presets
}
```

### Custom Normalization

```python
# Use custom normalization
report = search(
    model, data,
    perturbation="blur",
    normalization={
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    }
)
```

---

## CLI Interface

### Commands

```bash
# Run tests
visprobe run <script.py>

# Visualize results
visprobe visualize <script.py>

# List presets
visprobe list-presets

# List perturbations
visprobe list-perturbations
```

---

## Exceptions

### `VisProbeError`

Base exception for all VisProbe errors.

### `PerturbationError`

Raised when perturbation application fails.

### `PropertyError`

Raised when property evaluation fails.

### `SearchError`

Raised when search algorithm fails.

---

## Type Definitions

```python
# Type aliases used in the API
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch

ImageData = List[Tuple[torch.Tensor, int]]
NormalizationParams = Dict[str, List[float]]
StrategyFactory = Callable[[float], Strategy]
PropertyFunction = Callable[[torch.Tensor, torch.Tensor], bool]
```