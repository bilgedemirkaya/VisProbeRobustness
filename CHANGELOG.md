# Changelog

All notable changes to VisProbe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-05-26

First PyPI release. A focused, minimum-viable compositional robustness tool.

### Added
- `CompositionalExperiment` runner: sweeps `(model, env perturbation, severity)` with optional adversarial attack per cell.
- `CheckpointManager`: automatic per-cell save and resume; reruns pick up exactly where they stopped.
- `ModelMemoryManager`: CPU<->GPU swapping so multiple models do not OOM a single GPU.
- `AttackFactory`: AutoAttack (standard + APGD-CE), PGD, and `none` modes; caches attack instances; short-circuits when eps < 1e-10.
- Four environmental perturbations: `GaussianBlur`, `GaussianNoise`, `Brightness`, `LowLight`.
- `CompositionalResults`: save/load to disk, `print_summary`, `compute_auc`, `plot_compositional`.
- One end-to-end walkthrough notebook at `examples/visprobe_walkthrough.ipynb`.
- pytest suite covering checkpointing, memory swapping, perturbations, attacks, results round-trip, and experiment resume.

### Not included (see ROADMAP.md)
- RobustBench leaderboard integration (planned for v3.0).

## [1.0.0] - 2024-02-01

Pre-PyPI prototype. Never published; superseded entirely by 2.0.0.

### Removed in 2.0.0 refactor
- Property-based testing framework.
- Adaptive / Binary / Bayesian search strategies.
- Streamlit dashboard.
- CLI entry point.
- Preset / JSON-config system.
