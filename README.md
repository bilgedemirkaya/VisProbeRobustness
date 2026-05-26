# VisProbe

**Test your vision model the way it will actually fail — under real-world conditions, not lab conditions.**

Your image classifier has 94% accuracy on the test set. What does it do when the camera is noisy *and* the lighting drops *and* someone is adversarially crafting input at the same time? Pure AutoAttack benchmarks answer none of those — they assume pristine images. Real deployments rarely give you pristine images.

VisProbe runs the experiment that does: sweep your model across `environment × adversarial-attack × severity` in one command and tell you where it actually breaks. One call. Auto-resuming. No OOM. Save the results once, share them anywhere.

## What you get

**For evaluating one model.** Run a sweep with one command. Get accuracy curves across blur, noise, low-light, and brightness — each combined with adversarial attack at multiple severities. The interaction effects are where models fail in practice and where pure attack benchmarks are silent.

**For long experiments.** Every `(model, scenario, severity)` cell is checkpointed as soon as it finishes. Kernel crash, session timeout, manual cancel — rerun and it picks up exactly where it stopped. We learned this the hard way.

**For multi-model comparisons.** One model on the GPU, the rest swapped to CPU automatically. Compare 4-5 architectures on a single 24GB card without OOM.

**For sharing results.** Save once, load anywhere. Analyze on a laptop without GPUs or model weights — useful when your run finished on a server and you want to do the writeup on the train.

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
results.save("./results")
```

**Example output:**

```
VisProbe Compositional Robustness Report
========================================
Model: resnet50 (24M params)
Clean accuracy:          94.2%
Robust accuracy (AA):    12.4%

Compositional degradation:
  blur + AA:        94.2% → 8.1%    (-86 pp at max severity)
  noise + AA:       94.2% → 11.3%   (-83 pp at max severity)
  brightness + AA:  94.2% → 9.7%    (-84 pp at max severity)
  lowlight + AA:    94.2% →  2.1%   (-92 pp at max severity)   ← weakest

Results saved to ./results/   •   3m 42s
```

The `lowlight + AA` row is the kind of failure mode a pure AutoAttack benchmark never surfaces — *the model is two-thirds more fragile under low light than under a clean attack*. That's the entire point of running the composition.

Reload later, on any machine:

```python
from visprobe import CompositionalResults
results = CompositionalResults.load("./results")
results.plot_compositional()
```

A full walkthrough lives in [examples/visprobe_walkthrough.ipynb](examples/visprobe_walkthrough.ipynb).

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

## Roadmap — coming in v3

**RobustBench leaderboard integration.** After your run finishes, get back something like:

> *Your model would rank #14 on the ImageNet Linf leaderboard. Closest comparable is `Liu2023_Swin-B` at +4.1 pp robust accuracy. Head-to-head on your data: your model beats `Liu2023_Swin-B` under `noise + AA` but loses by 11 pp under `lowlight + AA`.*

Two distinct comparisons in one command: the **official rank** (under RobustBench's strict protocol) and a **head-to-head** that re-evaluates the top-k published robust models *on your data*, so you see how they hold up under the same compositional conditions yours just failed. See [ROADMAP.md](ROADMAP.md) for the full design.

## Architecture

```
src/visprobe/
├── experiment.py     # CompositionalExperiment runner
├── checkpoint.py     # CheckpointManager: per-cell save/resume
├── memory.py         # ModelMemoryManager: CPU <-> GPU swapping
├── attacks.py        # AttackFactory: AutoAttack, APGD-CE, PGD, none
├── perturbations.py  # 4 environmental perturbations
└── results.py        # CompositionalResults: save/load, summary, plotting
```

## License

MIT. See [LICENSE](LICENSE).
