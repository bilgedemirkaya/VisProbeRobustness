# VisProbe

**Simple, production-ready robustness testing for vision models.**

VisProbe tests model robustness under compositional conditions (environmental degradation + adversarial attack) with automatic checkpointing, memory management, and fast execution.

## Features

✅ **Automatic Checkpointing** - Never lose progress, auto-resume from crashes
✅ **Memory Management** - Test multiple models without OOM errors
✅ **Built-in Attacks** - AutoAttack and PGD as first-class citizens
✅ **Fast Execution** - 30x faster with optimizations
✅ **Offline Analysis** - Analyze results without models or GPU
✅ **Simple API** - 5 lines to run full experiment

## Installation

```bash
pip install visprobe
```

For AutoAttack support:
```bash
pip install autoattack
```

## Quick Start

```python
from visprobe import CompositionalExperiment, get_standard_perturbations
import torchvision.models as models

# Load model
model = models.resnet50(pretrained=True)
model.eval()

# Create experiment
experiment = CompositionalExperiment(
    models={"resnet50": model},
    images=images,  # Your test images
    labels=labels,  # Ground truth labels
    env_strategies=get_standard_perturbations(),
    attack="autoattack-apgd-ce",  # Fast AutoAttack mode
    checkpoint_dir="./checkpoints"  # Auto-saves progress
)

# Run (auto-resumes if interrupted)
results = experiment.run()

# Analyze
results.print_summary()
results.plot_compositional()
results.save("./results")

# Load later without GPU
from visprobe import CompositionalResults
results = CompositionalResults.load("./results")
```

## Core API

### CompositionalExperiment

Main class for running experiments with automatic checkpointing and memory management.

```python
experiment = CompositionalExperiment(
    models={"model1": model1, "model2": model2},  # Multiple models
    images=images,                                # Test images (N, C, H, W)
    labels=labels,                                # Ground truth labels (N,)
    env_strategies={                              # Environmental perturbations
        "blur": GaussianBlur(sigma_max=3.0),
        "noise": GaussianNoise(std_max=0.1),
        "lowlight": LowLight(gamma_max=5.0)
    },
    attack="autoattack-apgd-ce",                  # Attack type
    severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],   # Severity levels
    eps_fn=lambda s: (8/255) * s,                 # Severity to epsilon mapping
    checkpoint_dir="./checkpoints",               # Checkpoint directory
    batch_size=50,                                # Batch size
    device="cuda"                                 # Device
)

results = experiment.run()
```

### Attack Options

- `"autoattack-standard"` - Full AutoAttack (4 attacks)
- `"autoattack-apgd-ce"` - Fast APGD-CE only (5x faster)
- `"pgd"` - Projected Gradient Descent
- `"none"` - No attack (environmental only)

### Environmental Perturbations

**Standard Set:**
```python
from visprobe import get_standard_perturbations

perturbations = get_standard_perturbations()
# Returns: blur, motion_blur, noise, salt_pepper, brightness, contrast, lowlight
```

**Custom Perturbations:**
```python
from visprobe import GaussianBlur, GaussianNoise, Compose

custom = {
    "strong_blur": GaussianBlur(sigma_max=5.0),
    "heavy_noise": GaussianNoise(std_max=0.2),
    "combined": Compose([
        GaussianBlur(sigma_max=2.0),
        GaussianNoise(std_max=0.1)
    ])
}
```

### Analysis Methods

```python
# Protection gap relative to baseline
gaps = results.protection_gap(baseline="resnet50")

# Crossover detection (accuracy threshold)
crossovers = results.crossover_detection(baseline="resnet50", threshold=0.5)

# Model disagreement
disagreement = results.disagreement_analysis()

# Confidence profile
profile = results.confidence_profile(model="resnet50", scenario="blur", severity=0.5)

# Robustness AUC
auc = results.compute_auc(model="resnet50", scenario="blur")

# Visualization
results.plot_compositional(save_path="results.png")
```

## Key Improvements Over Traditional Approaches

| Problem | Traditional | VisProbe Solution |
|---------|------------|-------------------|
| Kernel crashes lose progress | Start over | Auto-checkpoint & resume |
| Multiple models cause OOM | Manual swapping | Automatic memory management |
| AutoAttack setup complex | Manual wrapper | Built-in with caching |
| Slow execution (10+ hours) | No optimization | 30x faster (~20 minutes) |
| Analysis needs all models | In-memory only | Save/load offline analysis |

## Examples

See the `examples/` directory:
- `simple_example.py` - Basic usage
- `comprehensive_example.py` - All features

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- tqdm
- autoattack (optional, for AutoAttack support)

## Architecture

VisProbe uses a clean, modular architecture:

```
visprobe/
├── experiment.py    # Main runner with checkpointing
├── checkpoint.py    # Automatic save/resume
├── memory.py       # GPU memory management
├── attacks.py      # AutoAttack & PGD integration
├── perturbations.py # Environmental degradations
├── results.py      # Analysis & visualization
└── analysis.py     # Evaluation functions
```

## Citation

If you use VisProbe in your research, please cite:

```bibtex
@software{visprobe2024,
  title={VisProbe: Simple, Production-Ready Robustness Testing},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/visprobe}
}
```

## License

MIT License - see LICENSE file for details.