# VisProbe - Claude Development Guidelines

## Project Overview

**VisProbe** is a simple, production-ready robustness testing tool for vision models. The framework tests model robustness under compositional conditions (environmental degradation + adversarial attack) with automatic checkpointing, memory management, and fast execution.

### Core Philosophy
- **Simple over complex** - No hidden magic, transparent execution
- **Production-ready** - Checkpointing and memory management built-in
- **One clear goal** - Compositional robustness testing
- **Clean architecture** - ~2000 lines in 7 core files

### Target Use Case
Test computer vision models (classification) against:
- Environmental degradations (blur, noise, lighting)
- Adversarial attacks (AutoAttack, PGD)
- Compositional conditions (environment + attack combined)

---

## Architecture

### Clean File Structure (7 Core Files)
```
src/visprobe/
├── __init__.py          # Clean API exports
├── experiment.py        # Main CompositionalExperiment class
├── checkpoint.py        # Automatic checkpointing & resumption
├── memory.py           # GPU memory management
├── attacks.py          # AutoAttack & PGD integration
├── perturbations.py    # Environmental perturbations
├── results.py          # Results container & analysis
└── analysis.py         # Evaluation functions
```

### Core Components

#### 1. CompositionalExperiment (`experiment.py`)
Main runner that orchestrates everything:
- Manages checkpointing
- Handles memory swapping
- Runs evaluations
- Auto-resumes from crashes

#### 2. CheckpointManager (`checkpoint.py`)
Automatic save/resume functionality:
- Saves after each severity level
- Tracks progress in JSON
- Stores results as pickle files
- Resume from exact stopping point

#### 3. ModelMemoryManager (`memory.py`)
Prevents OOM errors:
- Swaps models between CPU/GPU
- One model on GPU at a time
- Tracks memory usage
- Automatic cleanup

#### 4. AttackFactory (`attacks.py`)
First-class attack integration:
- AutoAttack (standard & APGD-CE)
- PGD attack
- Cached attack instances
- eps=0 optimization

#### 5. Environmental Perturbations (`perturbations.py`)
Simple callable classes:
- GaussianBlur, MotionBlur
- GaussianNoise, SaltPepperNoise
- Brightness, Contrast, LowLight
- Compose for combinations

#### 6. CompositionalResults (`results.py`)
Results with integrated analysis:
- Protection gap analysis
- Crossover detection
- Disagreement analysis
- Save/load for offline analysis
- Visualization

---

## Code Style Guidelines

### Python Best Practices

#### 1. Simplicity First
```python
# GOOD: Simple, clear intent
class GaussianBlur:
    def __init__(self, sigma_max: float = 3.0):
        self.sigma_max = sigma_max

    def __call__(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        sigma = severity * self.sigma_max
        # Apply blur...

# BAD: Over-engineered
class GaussianBlurStrategy(BaseStrategy):
    def generate(self, images, model=None, level=None):
        # Complex inheritance chain...
```

#### 2. Type Hints Always
```python
def evaluate_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 50
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Always include type hints for clarity."""
    pass
```

#### 3. Explicit Error Handling
```python
# Handle specific cases
if eps < 1e-10:
    return images  # Skip attack for tiny epsilon

# Clear error messages
if model_name not in self.models:
    raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
```

#### 4. Memory Management Pattern
```python
# Always clean up after GPU operations
with torch.no_grad():
    outputs = model(images)

# Explicit cache clearing
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Documentation Standards

Every function needs clear docstring:
```python
def save_checkpoint(
    self,
    result: Any,
    model_name: str,
    scenario: str,
    severity: float
):
    """
    Save a single evaluation result.

    Args:
        result: The evaluation result to save
        model_name: Name of the model
        scenario: Perturbation scenario name
        severity: Severity level
    """
```

---

## API Design Principles

### 1. Single Entry Point
```python
# One main class, one main method
experiment = CompositionalExperiment(...)
results = experiment.run()
```

### 2. Sensible Defaults
```python
# Most parameters have good defaults
CompositionalExperiment(
    models=models,
    images=images,
    labels=labels,
    # Everything else optional with smart defaults
)
```

### 3. Progressive Disclosure
```python
# Simple case
experiment = CompositionalExperiment(
    models={"model": model},
    images=images,
    labels=labels
)

# Advanced case
experiment = CompositionalExperiment(
    models=models,
    images=images,
    labels=labels,
    env_strategies=custom_perturbations,
    attack="autoattack-apgd-ce",
    severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    eps_fn=lambda s: (8/255) * s,
    checkpoint_dir="./checkpoints",
    batch_size=50,
    device="cuda"
)
```

---

## Performance Optimizations

### Built-in Optimizations
1. **Skip attack when eps < 1e-10** - Major speedup
2. **APGD-CE mode** - 5x faster than full AutoAttack
3. **Cached attack instances** - Reuse between evaluations
4. **Batch processing** - Efficient GPU utilization
5. **Model swapping** - Prevents OOM

### Memory Patterns
```python
# Move to CPU when not in use
model.cpu()
torch.cuda.empty_cache()

# Process in batches
for i in range(0, n_samples, batch_size):
    batch = images[i:i+batch_size]
    # Process batch...
```

---

## Testing Guidelines

### Simple Test Structure
```python
def test_component():
    """Test one thing well."""
    # Setup
    component = Component()

    # Execute
    result = component.method()

    # Assert
    assert result == expected
```

### Integration Testing
```python
# Test the full pipeline
experiment = CompositionalExperiment(...)
results = experiment.run()
assert "model" in results.get_models()
```

---

## Common Patterns

### Checkpointing Pattern
```python
# Check if already done
if checkpoint_mgr.is_completed(model, scenario, severity):
    result = checkpoint_mgr.load_checkpoint(model, scenario, severity)
else:
    # Do work
    result = evaluate(...)
    # Save immediately
    checkpoint_mgr.save_checkpoint(result, model, scenario, severity)
```

### Memory Management Pattern
```python
# Load model
model = memory_mgr.load_model(model_name)  # Others moved to CPU

# Use model
result = evaluate(model, ...)

# Clear periodically
torch.cuda.empty_cache()
```

---

## What NOT to Do

### ❌ Don't Add Complexity
- No complex inheritance hierarchies
- No abstract base classes unless necessary
- No design patterns for their own sake
- No hidden state or magic

### ❌ Don't Add Features
- No dashboard/UI
- No complex search algorithms
- No property-based testing framework
- No version management
- Keep it simple and focused

### ❌ Don't Break Core Principles
- Always checkpoint after work
- Always manage memory explicitly
- Always handle eps=0 case
- Always provide clear error messages

---

## Claude-Specific Instructions

When modifying VisProbe:

1. **Maintain simplicity** - Don't add complexity without clear benefit
2. **Follow existing patterns** - Consistency over creativity
3. **Test changes** - Run the test suite in `test_refactored.py`
4. **Document clearly** - Update docstrings and README if needed
5. **Preserve core features**:
   - Automatic checkpointing must work
   - Memory management must prevent OOM
   - AutoAttack must remain first-class
   - Analysis must work offline

### When Adding Features

Ask yourself:
- Does this make the tool simpler or more complex?
- Does this solve a real problem users have?
- Can this be done without changing core files?
- Is this feature worth the added complexity?

If the answer to any is "no", reconsider.

### Code Review Checklist

- [ ] Code is simple and clear
- [ ] Type hints are complete
- [ ] Docstrings are present
- [ ] Memory is managed properly
- [ ] Checkpointing still works
- [ ] No unnecessary complexity added
- [ ] Tests pass

---

## Quick Reference

### Main API
```python
from visprobe import (
    CompositionalExperiment,    # Main runner
    CompositionalResults,       # Results container
    get_standard_perturbations, # 7 standard perturbations
    get_minimal_perturbations,  # 3 key perturbations
    GaussianBlur, GaussianNoise, LowLight,  # Individual perturbations
    AttackFactory              # Create attacks
)
```

### Standard Workflow
```python
# Setup
experiment = CompositionalExperiment(
    models=models,
    images=images,
    labels=labels,
    checkpoint_dir="./checkpoints"
)

# Run
results = experiment.run()

# Analyze
results.print_summary()
results.save("./results")

# Load later
results = CompositionalResults.load("./results")
```

---

## Version History

- **v2.0** (Current): Complete refactor to simple, production-ready tool
- **v1.0** (Legacy): Complex framework with property-based testing

---

## Resources

- Main README: `/README.md`
- Examples: `/examples/`
- Test suite: `/test_refactored.py`
- Refactoring notes: `/REFACTORING_SUMMARY.md`