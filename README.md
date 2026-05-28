# VisProbe Testing

**Rank your vision model against the RobustBench leaderboard under their exact protocol — then find the real world failure modes the leaderboard doesn't measure.**

Every robust-vision paper reports a "robust accuracy" number, and few are directly comparable unless the protocol matches exactly. Different sample counts. Different attack subsets. Different epsilons. Different test-time augmentations. The published rankings on RobustBench mean something specific — and your evaluation only matches them if you ran the *exact* same thing.

VisProbe does two things: a defensible rank, plus the failure modes the rank misses.

1. **Compute a leaderboard rank under strict protocol enforcement.** You either match the RobustBench protocol byte-for-byte and get a comparable rank, or VisProbe refuses to produce one. No silent disagreement with the published numbers.
2. **Sweep that same model across `environment × attack × severity`.** A model that ranks well on pristine images can collapse on inputs a real camera produces — those failure modes don't show up on the leaderboard. The rank tells you where you sit on paper; the sweep tells you what your users will actually hit.

## 1. Where do I sit on the leaderboard?

```python
from visprobe import robustbench_eval, CompositionalResults

# CIFAR-10 ~5h, ~$15 on A100  |  ImageNet ~25h, ~$70. confirm=True is required so you see the cost first.
result = robustbench_eval(model, "cifar10", "Linf", confirm=True)

results = CompositionalResults()
results.add_result("my_model", "none", 0.0, result)
print(results.compare_to_leaderboard("my_model", "cifar10", "Linf"))
```

```
RobustBench cifar10/Linf — my_model
====================================
Rank:         14 of 99   (top 14.1%)
Robust acc:   0.6789
Protocol:     autoattack-standard, eps=8/255 (full RobustBench Linf)
Snapshot:     2026-05-27

Neighbors above (better):
  #11   Wang2024Foo                            0.6912   (+1.23 pp)
  #12   Bai2024Bar                             0.6856   (+0.67 pp)
  #13   Cui2023Baz                             0.6823   (+0.34 pp)

Neighbors below (worse):
  #15   Gowal2023Qux                           0.6745   (-0.44 pp)
  #16   Rebuffi2022Quux                        0.6710   (-0.79 pp)
  #17   Carmon2021Corge                        0.6680   (-1.09 pp)
```

That's the number you can put in a paper and defend.

## Why the protocol matters

"Robust accuracy under AutoAttack" varies between papers in ways that quietly destroy comparability:

- APGD-CE only, or the full AutoAttack suite (4 sub-attacks). APGD-CE typically reports 1-3 pp higher.
- 1000 samples, or 10000. Standard error scales with √N.
- eps=8/255, eps=4/255, or something else entirely.
- With or without test-time augmentation.

"I beat the leaderboard" stops meaning what it sounds like. It becomes "I beat my version of it."

VisProbe pins the protocol per `(dataset, threat)` pair and validates it on every rank call.

### The protocols

Each RobustBench leaderboard has its own fixed evaluation:

| Leaderboard | Attack | eps | Samples | Ships in v3? |
|---|---|---|---|---|
| `cifar10` / `Linf` | autoattack-standard | 8/255 | 10000 | ✓ |
| `cifar100` / `Linf` | autoattack-standard | 8/255 | 10000 | ✓ |
| `imagenet` / `Linf` | autoattack-standard | 4/255 | 5000 | ✓ |
| `cifar10` / `L2` | autoattack-standard | 0.5 | 10000 | later |
| `cifar10` / `corruptions` | *no attack — uses CIFAR-10-C* | n/a | per corruption | later |

`autoattack-standard` is the full AutoAttack suite (APGD-CE + APGD-DLR + FAB + Square). The `corruptions` threat is a different evaluation entirely — the model is graded on pre-computed corrupted images rather than an adversary — so its protocol shape is fundamentally different and v3 doesn't cover it yet.

### What the gate checks

On every `compare_to_leaderboard()` call, VisProbe validates your `EvaluationResult` against the protocol for the requested `(dataset, threat)`:

- `metadata.protocol == "robustbench"` — only `robustbench_eval()` sets this, so arbitrary `CompositionalExperiment` outputs can't accidentally claim a rank.
- `attack` matches the expected attack for that threat.
- `eps` matches the threat's eps (within float tolerance).
- `n_samples` matches the protocol sample count.

Any mismatch raises `ProtocolError` with the full list of violations and no attempt to approximate:

```
ProtocolError: Cannot rank against RobustBench cifar10/Linf — protocol mismatch:
  - attack='autoattack-apgd-ce', expected 'autoattack-standard'
  - n_samples=1000, expected 10000
  - eps=0.0156862745, expected 0.0313725490

Fix: use robustbench_eval(model, dataset='cifar10', threat='Linf') to produce
a protocol-compliant result, or check that result.metadata survived any
serialization round-trip that might have stripped it.
```

There's no honest way to convert a 1000-sample APGD-CE number into a 10000-sample full-AutoAttack number, so the gate just refuses. The number you publish is the number you can defend.

## 2. What the leaderboard rank misses

A leaderboard rank is one point in a much larger evaluation space. It says nothing about how your model behaves when the camera is noisy, when the lighting drops, when an attacker exploits both at once. Real deployments rarely give you the pristine inputs RobustBench evaluates on.

For that, sweep the same model across `environment × adversarial-attack × severity` and look at where it actually breaks:

```python
from visprobe import CompositionalExperiment, get_standard_perturbations

experiment = CompositionalExperiment(
    models={"my_model": model},
    images=images,
    labels=labels,
    env_strategies=get_standard_perturbations(),   # blur, noise, brightness, lowlight
    attack="autoattack-apgd-ce",
    severities=[0.0, 0.25, 0.5, 0.75, 1.0],
    eps_fn=lambda s: (8 / 255) * s,
    checkpoint_dir="./checkpoints",                # auto-resumes if interrupted
)
results = experiment.run()                     # prints cost estimate; pass confirm=True for sweeps > ~1h or ~$5
results.save("./results")
```

Run both — the leaderboard rank tells you where you sit, the compositional sweep tells you what your deployment faces. In our pilot on CIFAR-10, **Wang2023, the #1 ranked model on RobustBench Linf, drops from 74% robust accuracy on clean inputs to 45% once Gaussian noise is added** (pilot: APGD-CE on 1000 samples — the strict-protocol leaderboard number is 71%). **Gowal2020 (~#30 on the leaderboard) ties Wang2023 under that noise condition** — the #1-vs-#30 distinction collapses on inputs any real camera would produce. The gap between official rank and compositional behavior is where the interesting failure modes hide. Full pilot table: [pilot_grid.csv](pilot_grid.csv).

## Install

```bash
pip install "visprobe[all]"
```

That pulls in AutoAttack + RobustBench, which you need for both leaderboard rank and adversarial sweeps. (Bare `pip install visprobe` works but only gives you PGD and environment-only eval.)

> If `pip install` can't find `autoattack` (the PyPI package occasionally lags), install it from GitHub instead:
> ```bash
> pip install visprobe[robustbench]
> pip install git+https://github.com/fra31/auto-attack
> ```

## Attack modes

| `attack=` | Use |
|---|---|
| `"autoattack-standard"` | Full AutoAttack. Required for `robustbench_eval`. |
| `"autoattack-apgd-ce"` | APGD-CE only. ~5x faster. For compositional sweeps; not for leaderboard rank. |
| `"pgd"` | Standard PGD-Linf. Debugging or speed-sensitive sweeps. |
| `"none"` | Identity. Environment-only robustness. |

## v3.1

**Head-to-head on your data.** Download the top-k published robust models, re-evaluate them on *your* images under *your* compositional protocol, and rank yourself alongside. Distinct from the official rank: the output carries a `data_source: user` label everywhere so the two cannot be confused.

## Architecture

```
src/visprobe/
├── experiment.py     # CompositionalExperiment + robustbench_eval
├── leaderboard.py    # validate_protocol + RobustBenchClient + LeaderboardComparison
├── checkpoint.py     # per-cell save/resume (module functions)
├── memory.py         # one model on GPU at a time, rest swapped to CPU
├── attacks.py        # attacks.build(): AutoAttack standard/APGD-CE, PGD, none
├── perturbations.py  # 4 environmental perturbations: blur, noise, brightness, lowlight
└── results.py        # CompositionalResults: save/load, summary, ranking
```

Leaderboard snapshots ship in `src/visprobe/data/`. CI refreshes them weekly via [.github/workflows/refresh-leaderboard.yml](.github/workflows/refresh-leaderboard.yml).

## Examples

- [examples/visprobe_walkthrough.ipynb](examples/visprobe_walkthrough.ipynb) — full compositional-eval workflow end to end.
- [examples/plotting.ipynb](examples/plotting.ipynb) — matplotlib recipes (faceted accuracy curves + heatmap).

## License

MIT.
