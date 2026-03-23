# VisProbe Examples

This directory contains examples demonstrating how to use VisProbe for compositional robustness testing.

## Examples

### 1. Simple Example (`simple_example.py`)
Basic usage showing:
- Single model testing
- Minimal perturbations (blur, noise, lowlight)
- AutoAttack integration
- Automatic checkpointing

### 2. Comprehensive Example (`comprehensive_example.py`)
Advanced features including:
- Multi-model comparison with automatic memory management
- All standard perturbations
- Protection gap analysis
- Crossover detection
- Model disagreement analysis
- Custom perturbations
- Loading and analyzing saved results without GPU

## Quick Start

```python
from visprobe import CompositionalExperiment, get_standard_perturbations

# Create experiment
experiment = CompositionalExperiment(
    models={"resnet50": model},
    images=images,
    labels=labels,
    env_strategies=get_standard_perturbations(),
    attack="autoattack-apgd-ce",
    checkpoint_dir="./checkpoints"
)

# Run with auto-checkpointing
results = experiment.run()

# Analyze
results.print_summary()
results.save("./results")
```

## Key Features

- **Automatic Checkpointing**: Experiments save progress and resume from interruptions
- **Memory Management**: Multiple models are automatically swapped between CPU/GPU
- **Built-in Attacks**: AutoAttack and PGD are first-class citizens
- **Comprehensive Analysis**: Protection gaps, crossover detection, disagreement analysis
- **Offline Analysis**: Load and analyze results without models or GPU