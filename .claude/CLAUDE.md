# VisProbe Project - Claude Development Guidelines

## Project Overview

**VisProbe** is a property-based robustness testing framework for vision models. The framework enables systematic testing of ML model robustness through declarative property specifications and adaptive search strategies.

### Core Innovation
- **Property-first testing** (not attack-first)
- **Adaptive threshold search** (not fixed grid sweeps)
- **Declarative specifications** for robustness properties
- **Modular, extensible architecture**

### Target Domain
- Primary: Computer Vision models (classification, detection, segmentation)
- Future: Extensible to NLP, Audio, and other ML domains

---

## Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: PyTorch
- **Architecture Pattern**: Decorator-based API with strategy pattern
- **Testing Paradigm**: Property-based testing (inspired by Hypothesis, QuickCheck)

---

## Architecture Principles

### 1. Core Components (Domain-Agnostic)

The framework is designed with modality-agnostic core components:

#### Primary API
```python
search()        # Single-strategy threshold finding
```

#### Core Classes
```python
SearchEngine    # Unified search algorithm (core/search_engine.py)
Strategy        # Perturbation strategies (image, text, audio, etc.)
Property        # Test oracle / assertions on model outputs
Report          # Results with score, failures, metrics
```

#### Search Mechanism
- SearchEngine provides adaptive and binary search modes
- Modality-agnostic - doesn't care about input type

### 2. Modality-Specific Implementations

Current implementations are vision-specific but follow extensible patterns:

```
visprobe/
├── strategies/
│   └── image.py        # Vision perturbations (future: text.py, audio.py)
├── properties/
│   └── classification.py  # Classification properties (future: generation.py)
└── core/               # Domain-agnostic core
```

### 3. Extension Pattern

When adding new modalities:
- Implement new `Strategy` subclass in `strategies/{modality}.py`
- Implement new `Property` subclass in `properties/{task_type}.py`
- Core SearchEngine logic remains unchanged

---

## Code Style Guidelines

### Python Best Practices

#### 1. Object-Oriented Design

**Use `__init__` for object initialization:**
```python
class SearchStrategy:
    def __init__(self, max_iterations: int, tolerance: float):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._cache = {}
```

**Property decorators for validation and computed values:**
```python
class PerturbationStrength:
    def __init__(self, value: float):
        self._value = value

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, val: float):
        if val < 0 or val > 1:
            raise ValueError("Strength must be in [0, 1]")
        self._value = val

    @property
    def normalized(self) -> float:
        """Computed property"""
        return self._value / self.max_value
```

#### 2. Type Hints

Always use type hints for function signatures:
```python
from typing import List, Dict, Optional, Union, Callable
import torch

def evaluate_property(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    property_fn: Callable[[torch.Tensor], bool],
    threshold: float = 0.5
) -> Dict[str, Union[bool, float]]:
    """
    Evaluate a property on model outputs.

    Args:
        model: PyTorch model to test
        inputs: Input tensor
        property_fn: Property function returning bool
        threshold: Confidence threshold

    Returns:
        Dict containing result and confidence score
    """
    pass
```

#### 3. Naming Conventions

**Follow Python conventions:**
```python
# Classes: PascalCase
class AdaptiveSearch:
    pass

# Functions/methods: snake_case
def find_minimal_perturbation():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_SEARCH_ITERATIONS = 100
DEFAULT_TOLERANCE = 1e-5

# Private members: leading underscore
class Strategy:
    def __init__(self):
        self._internal_state = {}

    def _compute_gradient(self):  # Private method
        pass
```

**Domain-specific naming:**
```python
# Be explicit about what values represent
perturbation_strength  # Good
strength              # Too vague

# Use domain terminology
property_fn           # Good (property-based testing term)
test_function        # Less clear

# Clarity over brevity
adaptive_threshold_search  # Good
ats                       # Bad
```

#### 4. Docstrings

Use Google-style docstrings:
```python
def adaptive_search(
    model: torch.nn.Module,
    property_oracle: PropertyOracle,
    search_strategy: SearchStrategy,
    initial_threshold: float = 0.5
) -> SearchResult:
    """
    Perform adaptive search to find minimal perturbation thresholds.

    This implements the core adaptive search algorithm (RQ2) that efficiently
    explores the perturbation space to find failure boundaries.

    Args:
        model: The vision model under test
        property_oracle: Oracle defining the robustness property
        search_strategy: Strategy for exploring perturbation space
        initial_threshold: Starting point for binary search

    Returns:
        SearchResult containing:
            - minimal_threshold: Smallest perturbation causing failure
            - perturbation: The failing perturbation
            - iterations: Number of search iterations
            - confidence: Confidence in the result

    Raises:
        ValueError: If initial_threshold is outside [0, 1]
        RuntimeError: If search fails to converge

    Example:
        >>> strategy = BinarySearchStrategy(max_iters=50)
        >>> result = adaptive_search(resnet18, property, strategy)
        >>> print(f"Failure at threshold: {result.minimal_threshold}")
    """
    pass
```

### PyTorch-Specific Patterns

#### 1. Model Handling

```python
def test_model(model: torch.nn.Module, device: str = "cuda"):
    """Always handle device placement and eval mode properly"""
    model = model.to(device)
    model.eval()  # Disable dropout, batch norm, etc.

    with torch.no_grad():  # Disable gradient computation for testing
        outputs = model(inputs)

    return outputs
```

#### 2. Tensor Operations

```python
# Prefer functional API for transformations
import torch.nn.functional as F

def apply_perturbation(
    image: torch.Tensor,
    perturbation: torch.Tensor,
    strength: float
) -> torch.Tensor:
    """Apply perturbation with proper tensor handling"""
    # Ensure tensors are on same device
    perturbation = perturbation.to(image.device)

    # Clone to avoid in-place modification
    perturbed = image.clone()

    # Apply perturbation
    perturbed = perturbed + strength * perturbation

    # Clamp to valid range [0, 1] for images
    perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed
```

#### 3. Batching

```python
def evaluate_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    batch_size: int = 32
) -> torch.Tensor:
    """Process large datasets in batches to manage memory"""
    results = []

    for i in range(0, len(batch), batch_size):
        batch_subset = batch[i:i + batch_size]
        with torch.no_grad():
            outputs = model(batch_subset)
        results.append(outputs)

    return torch.cat(results, dim=0)
```

### Error Handling

```python
class PerturbationError(Exception):
    """Base exception for perturbation-related errors"""
    pass

class PropertyViolation(PerturbationError):
    """Raised when a property is violated"""
    def __init__(self, property_name: str, threshold: float, actual: float):
        self.property_name = property_name
        self.threshold = threshold
        self.actual = actual
        super().__init__(
            f"Property '{property_name}' violated: "
            f"threshold={threshold}, actual={actual}"
        )

def evaluate_with_error_handling(
    model: torch.nn.Module,
    inputs: torch.Tensor
) -> torch.Tensor:
    """Robust evaluation with proper error handling"""
    try:
        outputs = model(inputs)
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Handle OOM gracefully
            torch.cuda.empty_cache()
            raise PerturbationError("GPU out of memory") from e
        raise
    except Exception as e:
        raise PerturbationError(f"Model evaluation failed: {e}") from e

    return outputs
```

---

## Testing Guidelines

### Property-Based Testing Patterns

```python
from hypothesis import given, strategies as st
import pytest

@given(
    strength=st.floats(min_value=0.0, max_value=1.0),
    image_size=st.integers(min_value=32, max_value=512)
)
def test_perturbation_strength_bounds(strength: float, image_size: int):
    """Properties should hold for all valid inputs"""
    perturbation = generate_perturbation(image_size, strength)

    # Property: perturbation magnitude should not exceed strength
    assert torch.abs(perturbation).max() <= strength

    # Property: output image should be in valid range
    image = torch.rand(3, image_size, image_size)
    perturbed = apply_perturbation(image, perturbation, strength)
    assert perturbed.min() >= 0.0
    assert perturbed.max() <= 1.0
```

### Unit Tests

```python
class TestAdaptiveSearch:
    @pytest.fixture
    def mock_model(self):
        """Fixture for reproducible test model"""
        model = SimpleClassifier()
        model.eval()
        return model

    def test_search_converges(self, mock_model):
        """Search should converge within max iterations"""
        strategy = BinarySearchStrategy(max_iters=50, tolerance=0.01)
        result = adaptive_search(mock_model, property_fn, strategy)

        assert result.converged
        assert result.iterations <= 50

    def test_search_finds_minimal_threshold(self, mock_model):
        """Found threshold should be minimal"""
        result = adaptive_search(mock_model, property_fn, strategy)

        # Property holds at threshold - ε
        assert property_fn(result.threshold - 0.01)

        # Property fails at threshold
        assert not property_fn(result.threshold)
```

---

## Performance Considerations

### 1. Caching

```python
from functools import lru_cache
import hashlib

class PerturbationCache:
    """Cache perturbation evaluations to avoid recomputation"""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, torch.Tensor] = {}
        self.max_size = max_size

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """Create hash of tensor for cache key"""
        return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()

    def get(self, inputs: torch.Tensor, strength: float) -> Optional[torch.Tensor]:
        """Retrieve cached result if available"""
        key = f"{self._hash_tensor(inputs)}_{strength}"
        return self._cache.get(key)

    def set(self, inputs: torch.Tensor, strength: float, result: torch.Tensor):
        """Cache a result"""
        if len(self._cache) >= self.max_size:
            # Simple FIFO eviction
            self._cache.pop(next(iter(self._cache)))

        key = f"{self._hash_tensor(inputs)}_{strength}"
        self._cache[key] = result
```

### 2. Parallelization

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List

def evaluate_multiple_properties_parallel(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    properties: List[PropertyOracle],
    max_workers: int = 4
) -> List[bool]:
    """Evaluate multiple properties in parallel"""

    def eval_single(prop: PropertyOracle) -> bool:
        with torch.no_grad():
            return prop.evaluate(model, inputs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(eval_single, properties))

    return results
```

### 3. Memory Management

```python
def process_large_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32
):
    """Process large datasets with proper memory management"""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True  # Faster GPU transfer
    )

    results = []
    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            outputs = model(batch)

        # Move results to CPU to free GPU memory
        results.append(outputs.cpu())

        # Explicit cleanup
        del batch, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(results, dim=0)
```

---

## Documentation Standards

### README Structure

Each module should have a README covering:
1. **Purpose**: What this module does
2. **Usage**: Basic examples
3. **API**: Key classes and functions
4. **Examples**: Common use cases

### Code Comments

```python
# Good comments explain WHY, not WHAT
def adaptive_binary_search(start: float, end: float) -> float:
    # Binary search is optimal here because we need O(log n) convergence
    # for large perturbation spaces (10^6+ possible values)
    midpoint = (start + end) / 2

    # Cache this evaluation - it's expensive (forward pass + property check)
    if not self._is_cached(midpoint):
        result = self._evaluate(midpoint)
        self._cache[midpoint] = result

    return self._cache[midpoint]
```

### API Documentation

```python
"""
visprobe.strategies.image
~~~~~~~~~~~~~~~~~~~~~~~~~

Image perturbation strategies for vision model testing.

This module provides various perturbation strategies for testing robustness
of vision models, including:

- Gaussian noise
- Adversarial perturbations (FGSM, PGD)
- Geometric transformations (rotation, scaling)
- Color space manipulations

Example:
    >>> from visprobe.strategies.image import GaussianNoise
    >>> strategy = GaussianNoise(mean=0, std=0.1)
    >>> perturbed = strategy.apply(image, strength=0.5)
"""
```

---

## Git Workflow

### Branch Naming
```
feature/adaptive-search-optimization
fix/property-validation-bug
refactor/strategy-interface
docs/api-reference
```

### Commit Messages

Follow conventional commits:
```
feat(search): implement parallel property evaluation
fix(strategies): correct normalization in Gaussian noise
refactor(core): simplify SearchEngine implementation
docs(readme): add examples for custom properties
test(properties): add edge cases for classification oracle
perf(cache): optimize tensor hashing for cache keys
```

### PR Guidelines

**PR Title**: `[Component] Brief description`

**PR Description Template**:
```markdown
## Changes
- Brief bullet points of what changed

## Motivation
Why this change is needed

## Testing
How this was tested

## Breaking Changes
Any backwards incompatible changes

## Related Issues
Closes #123
```

---

## Research & Academic Context

### Positioning the Framework

When discussing VisProbe in academic or professional contexts:

**Focus Areas:**
- Property-based testing methodology (RQ1)
- Adaptive threshold search algorithms (RQ2)
- Minimal perturbation discovery
- Declarative robustness specifications

**Key Differentiators:**
- **vs. Attack Libraries (ART, Foolbox)**: Property-first, not attack-first
- **vs. Traditional Testing**: Adaptive search, not grid sweep
- **vs. Formal Verification**: Practical testing, not proof-based

**Applications:**
- Model debugging and development
- Robustness certification
- Safety-critical system validation
- MLOps quality assurance

---

## Common Patterns & Anti-patterns

### ✅ Good Patterns

```python
# 1. Composition over inheritance
class CompositePerturbation:
    def __init__(self, strategies: List[Strategy]):
        self.strategies = strategies

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        result = image
        for strategy in self.strategies:
            result = strategy.apply(result)
        return result

# 2. Strategy pattern for extensibility
class SearchStrategy(ABC):
    @abstractmethod
    def search(self, evaluate_fn: Callable) -> float:
        pass

# 3. Context managers for resource management
class GPUMemoryManager:
    def __enter__(self):
        torch.cuda.empty_cache()
        return self

    def __exit__(self, *args):
        torch.cuda.empty_cache()
```

### ❌ Anti-patterns to Avoid

```python
# 1. Avoid global state
# Bad:
global_model = None
def set_model(model):
    global global_model
    global_model = model

# Good:
class ModelEvaluator:
    def __init__(self, model):
        self.model = model

# 2. Don't mix concerns
# Bad:
def evaluate_and_log_and_save(model, data, filename):
    result = model(data)
    logger.info(f"Result: {result}")
    torch.save(result, filename)
    return result

# Good:
def evaluate(model, data):
    return model(data)
# Separate logging and saving

# 3. Avoid magic numbers
# Bad:
if strength > 0.7:  # What does 0.7 mean?
    pass

# Good:
PERTURBATION_THRESHOLD = 0.7  # Maximum allowed perturbation strength
if strength > PERTURBATION_THRESHOLD:
    pass
```

---

## Development Workflow

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in editable mode

# Run tests
pytest tests/

# Run with coverage
pytest --cov=visprobe tests/

# Type checking
mypy visprobe/

# Linting
pylint visprobe/
black visprobe/  # Auto-format
```

### Pre-commit Checks

```bash
# Format code
black visprobe/ tests/

# Sort imports
isort visprobe/ tests/

# Type check
mypy visprobe/

# Run tests
pytest tests/
```

---

## Troubleshooting Common Issues

### PyTorch Device Issues

```python
# Always check device compatibility
def safe_to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Safely move tensor to device"""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, using CPU")
        device = "cpu"
    return tensor.to(device)
```

### Memory Issues

```python
# Monitor GPU memory
def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

# Clear memory periodically
def clear_gpu_cache_if_needed():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        if allocated / total > 0.8:  # 80% threshold
            torch.cuda.empty_cache()
```

---

## Claude-Specific Instructions

When writing code for VisProbe:

1. **Always read relevant sections of this file** before starting implementation
2. **Follow the established patterns** for SearchEngine, strategies, and properties
3. **Use type hints** for all function signatures
4. **Write docstrings** in Google style
5. **Consider extensibility** - will this work for other modalities?
6. **Optimize for readability** over cleverness
7. **Test property-based** when possible (using Hypothesis)
8. **Document decisions** - why this approach over alternatives?
9. **Keep core modality-agnostic** - only implementation details should be domain-specific
10. **Performance matters** - cache evaluations, use batching, manage GPU memory

### When Adding New Features

**Before implementing:**
- [ ] Read relevant sections of claude.md
- [ ] Check existing similar implementations
- [ ] Verify it fits the modular architecture
- [ ] Consider backward compatibility

**During implementation:**
- [ ] Follow naming conventions
- [ ] Add type hints
- [ ] Write docstrings
- [ ] Handle errors properly
- [ ] Add logging where appropriate

**After implementation:**
- [ ] Write unit tests
- [ ] Add property-based tests if applicable
- [ ] Update documentation
- [ ] Test with real models
- [ ] Check performance impact

### Code Review Checklist

- [ ] Follows established patterns
- [ ] Has appropriate type hints
- [ ] Has Google-style docstrings
- [ ] Includes error handling
- [ ] Has unit tests
- [ ] Maintains modular architecture
- [ ] Considers performance
- [ ] Updates relevant documentation

---

## Resources & References

### Property-Based Testing
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- QuickCheck papers (original PBT framework)

### ML Robustness
- Adversarial Robustness Toolbox (ART)
- Foolbox library
- Papers on model robustness evaluation

### PyTorch
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Best Practices Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## Version History

- **v1.0** (Initial): Core framework with vision model support
- **Future**: NLP support, additional search strategies, distributed testing
