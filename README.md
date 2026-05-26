# VisProbe

**Compositional robustness testing for vision models.**

VisProbe runs a single, focused experiment: sweep your classifier across `environment x adversarial-attack x severity`, save every intermediate result so a crash never costs progress, and swap models between CPU and GPU so multiple models do not OOM. That is the entire tool.

## What you get

- **Compositional eval pipeline** — environmental perturbation, then adversarial attack, then accuracy, across a severity sweep.
- **Checkpoint + auto-resume** — every `(model, scenario, severity)` cell is checkpointed; rerunning picks up exactly where it stopped.
- **GPU memory management** — only one model on the GPU at a time, the rest are swapped to CPU.
- **AutoAttack + APGD-CE built in** — plus PGD, plus a `none` mode for environment-only sweeps.
- **4 environmental perturbations** — Gaussian blur, Gaussian noise, brightness shift, low-light (gamma).
- **Saveable, loadable results** — inspect runs offline without GPUs or model weights.

## Install

```bash
pip install visprobe              # core
pip install visprobe[autoattack]  # adds AutoAttack
```

## Quick start

```python
from visprobe import CompositionalExperiment, get_standard_perturbations

experiment = CompositionalExperiment(
    models={"resnet50": model},
    images=images,                          # (N, C, H, W) in [0, 1]
    labels=labels,                          # (N,)
    env_strategies=get_standard_perturbations(),
    attack="autoattack-apgd-ce",            # fast AutoAttack mode
    severities=[0.0, 0.25, 0.5, 0.75, 1.0],
    eps_fn=lambda s: (8 / 255) * s,
    checkpoint_dir="./checkpoints",
)

results = experiment.run()                  # auto-resumes if interrupted
results.print_summary()
results.plot_compositional()
results.save("./results")
```

Reload later, on any machine:

```python
from visprobe import CompositionalResults
results = CompositionalResults.load("./results")
```

A runnable end-to-end walkthrough lives in [examples/visprobe_walkthrough.ipynb](examples/visprobe_walkthrough.ipynb).

## Attack modes

| `attack=` | What it does |
|---|---|
| `"autoattack-standard"` | Full AutoAttack (APGD-CE + APGD-DLR + FAB + Square). Most thorough, slowest. |
| `"autoattack-apgd-ce"` | APGD-CE only. ~5x faster, the right default for sweeps. |
| `"pgd"` | Standard PGD-Linf. |
| `"none"` | No attack; environmental-only robustness. |

`eps_fn(severity)` controls the attack budget per severity step. For Linf, `lambda s: (8/255) * s` is a sensible default.

## Perturbations

`get_standard_perturbations()` returns the four supported degradations:

```python
{
    "blur":       GaussianBlur(sigma_max=3.0),
    "noise":      GaussianNoise(std_max=0.1),
    "brightness": Brightness(delta_max=0.3),
    "lowlight":   LowLight(gamma_max=5.0),
}
```

Each is a callable `(images, severity) -> images` where `severity=0` is a no-op. Pass your own dictionary into `env_strategies=` to use custom perturbations.

## Architecture

```
src/visprobe/
├── experiment.py     # CompositionalExperiment runner
├── checkpoint.py     # CheckpointManager: per-cell save/resume
├── memory.py         # ModelMemoryManager: CPU<->GPU swapping
├── attacks.py        # AttackFactory: AutoAttack, APGD-CE, PGD, none
├── perturbations.py  # 4 environmental perturbations
└── results.py        # CompositionalResults: save/load, summary, plotting
```

## Roadmap

The next planned feature is **RobustBench leaderboard integration** for v2 — see [ROADMAP.md](ROADMAP.md) for the full design.

## License

MIT. See [LICENSE](LICENSE).
