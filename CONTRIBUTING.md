# Contributing to VisProbe

Thank you for your interest in contributing to VisProbe! This guide will help you get started with contributing to the project.

## 🎯 Ways to Contribute

- **Report bugs** and request features through GitHub Issues
- **Improve documentation** and examples
- **Add new perturbation strategies**
- **Implement new search algorithms**
- **Add support for new model types**
- **Fix bugs** and improve performance

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/VisProbe.git
cd VisProbe
git remote add upstream https://github.com/bilgedemirkaya/VisProbe.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .
pip install black flake8 isort mypy pytest
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

## 📝 Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking (optional but encouraged)

Run formatters before committing:

```bash
# Format code
black src/ examples/ --line-length 100
isort src/ examples/ --profile black

# Check linting
flake8 src/ examples/ --max-line-length=100 --extend-ignore=E203,W503

# Type checking (optional)
mypy src/visprobe --ignore-missing-imports
```

### Naming Conventions

```python
# Classes: PascalCase
class GaussianBlurStrategy:
    pass

# Functions/methods: snake_case
def apply_perturbation():
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_TOLERANCE = 0.01

# Private members: leading underscore
def _internal_method():
    pass
```

### Documentation

All public functions and classes must have docstrings:

```python
def search(model: torch.nn.Module,
           data: List[Tuple[torch.Tensor, int]],
           **kwargs) -> Report:
    """
    Search for robustness failures in a vision model.

    Args:
        model: PyTorch model to test
        data: List of (image, label) tuples
        **kwargs: Additional search parameters

    Returns:
        Report object containing test results

    Raises:
        ValueError: If data is empty
        RuntimeError: If search fails

    Example:
        >>> report = search(resnet18, test_data, preset="natural")
        >>> print(f"Score: {report.score}%")
    """
    pass
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Union, Tuple
import torch

def evaluate_property(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.9
) -> Dict[str, Union[float, bool]]:
    pass
```

## 🔧 Adding New Features

### Adding a New Perturbation

1. **Create the strategy class** in `src/visprobe/strategies/`:

```python
# src/visprobe/strategies/my_perturbation.py
from .base import Strategy
import torch

class MyPerturbation(Strategy):
    """Description of your perturbation."""

    def __init__(self, strength: float):
        """
        Initialize the perturbation.

        Args:
            strength: Perturbation strength [0, 1]
        """
        self.strength = strength

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply perturbation to images.

        Args:
            images: Input images tensor [B, C, H, W]

        Returns:
            Perturbed images tensor
        """
        # Your implementation here
        perturbed = images + self.strength * torch.randn_like(images)
        return torch.clamp(perturbed, 0, 1)
```

2. **Register the perturbation** in `src/visprobe/config/perturbations.yaml`:

```yaml
my_perturbation:
  name: "My Perturbation"
  description: "Description of what it does"
  strategy_class: "MyPerturbation"
  strategy_module: "visprobe.strategies.my_perturbation"
  parameters:
    imagenet:
      level_lo: 0.0
      level_hi: 1.0
      default: 0.5
    cifar10:
      level_lo: 0.0
      level_hi: 0.5
      default: 0.25
```

3. **Export from strategies** in `src/visprobe/strategies/__init__.py`:

```python
from .my_perturbation import MyPerturbation

__all__ = [
    # ... existing exports
    "MyPerturbation",
]
```

4. **Add to Perturbation enum** in `src/visprobe/perturbations.py`:

```python
class Perturbation:
    # ... existing perturbations
    MY_PERTURBATION = "my_perturbation"
```

5. **Create an example** in `examples/`:

```python
from visprobe import search, Perturbation

report = search(
    model, data,
    perturbation=Perturbation.MY_PERTURBATION,
    normalization="imagenet"
)
```

### Adding a New Preset

Create a YAML file in `src/visprobe/config/presets/`:

```yaml
# src/visprobe/config/presets/my_preset.yaml
name: "my_preset"
display_name: "My Custom Preset"
description: "Tests specific robustness properties"
perturbations:
  - gaussian_noise
  - gaussian_blur
  - my_perturbation
parameters:
  pass_threshold: 0.9
  search_method: "adaptive"
threat_model: "custom_threat"
```

### Adding a New Property

```python
# src/visprobe/properties/my_property.py
from .base import Property
import torch

class MyProperty(Property):
    """Custom property for robustness testing."""

    def evaluate(self,
                 original_outputs: torch.Tensor,
                 perturbed_outputs: torch.Tensor,
                 labels: torch.Tensor) -> float:
        """
        Evaluate if the property holds.

        Returns:
            Float between 0 and 1 indicating property satisfaction
        """
        # Your implementation
        pass
```

## 🧪 Testing

Write tests for new features (when test framework is set up):

```python
# tests/test_my_perturbation.py
import pytest
import torch
from visprobe.strategies import MyPerturbation

def test_my_perturbation_apply():
    strategy = MyPerturbation(strength=0.5)
    images = torch.rand(2, 3, 224, 224)

    perturbed = strategy.apply(images)

    assert perturbed.shape == images.shape
    assert perturbed.min() >= 0
    assert perturbed.max() <= 1

def test_my_perturbation_strength():
    # Test that higher strength = more perturbation
    pass
```

## 📦 Submitting Changes

### 1. Commit Your Changes

Write clear commit messages following the conventional commits format:

```bash
# Format: <type>(<scope>): <description>

git commit -m "feat(strategies): add quantum noise perturbation"
git commit -m "fix(search): handle edge case in adaptive search"
git commit -m "docs(readme): update installation instructions"
git commit -m "refactor(core): simplify normalization logic"
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `perf`: Performance improvements
- `style`: Code style changes

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Code runs without errors
- [ ] Added/updated examples if needed
- [ ] Documentation updated if needed

## Related Issues
Closes #123
```

## 📋 Pull Request Guidelines

### PR Requirements

- **Clear description** of changes
- **Follows code style** guidelines
- **Includes documentation** for new features
- **Adds examples** where appropriate
- **Passes CI checks** (when available)

### What We Look For

✅ **Good PR:**
- Focused on a single feature/fix
- Well-documented code
- Follows existing patterns
- Includes examples
- Clear commit history

❌ **Needs Work:**
- Multiple unrelated changes
- No documentation
- Breaks existing functionality
- Poor code quality
- No description

## 🐛 Reporting Issues

### Bug Reports

Include:
1. **Description** of the bug
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment** (Python version, PyTorch version, OS)
6. **Minimal code example**

Example:

```markdown
**Description:**
GaussianBlur fails with batch size > 1

**Steps to reproduce:**
```python
from visprobe import search
report = search(model, data, perturbation="gaussian_blur", batch_size=32)
```

**Expected:** Process batches correctly
**Actual:** RuntimeError: dimension mismatch

**Environment:**
- Python 3.10
- PyTorch 2.0
- Ubuntu 20.04
```

### Feature Requests

Include:
1. **Use case** - why is this needed?
2. **Proposed solution**
3. **Alternatives considered**
4. **Additional context**

## 🤝 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing private information
- Other unprofessional conduct

## 📬 Getting Help

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions and ideas
- **Documentation**: Check [docs/](docs/) first
- **Examples**: See [examples/](examples/) for usage

## 🙏 Recognition

Contributors will be:
- Listed in the contributors file
- Mentioned in release notes
- Given credit in documentation

Thank you for contributing to VisProbe! Your efforts help make vision models more robust and reliable. 🚀