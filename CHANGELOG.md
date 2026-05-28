# Changelog

All notable changes to VisProbe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-05-27

### Added — Feature A: RobustBench leaderboard ranking
- `robustbench_eval(model, dataset, threat, *, confirm=True)` — strict-protocol eval that produces a `EvaluationResult` tagged with `metadata.protocol = "robustbench"`.
- `CompositionalResults.compare_to_leaderboard(model_name, dataset, threat)` — protocol-gated ranking. Returns a `LeaderboardComparison` or raises `ProtocolError`.
- `RobustBenchClient(dataset, threat)` — snapshot reader with `rank()`, `top_k()`, `neighbors()`, `snapshot_date`.
- `LeaderboardComparison` dataclass — pleasant-to-print rank output with neighbors above/below and snapshot date.
- `ProtocolError` + `validate_protocol(result, dataset, threat)` — the gate function. Refuses to rank unless `attack`, `eps`, `n_samples`, and `metadata.protocol` all match the per-(dataset, threat) spec.
- Snapshot data shipped in the wheel: `src/visprobe/data/robustbench_cifar10_linf.json`, `robustbench_imagenet_linf.json`.
- Weekly CI snapshot refresh workflow: `.github/workflows/refresh-leaderboard.yml`.
- Maintainer-only refresher script: `scripts/refresh_leaderboard.py`.

### Added — Cost estimate + gate
- `CompositionalExperiment.run(*, confirm=False)` now prints a rough cost estimate (calibrated against the May 2026 pilot) and raises if the *remaining* work exceeds ~1 hour or ~$5 of A100-80GB time. Tiny sweeps run silently; long sweeps require explicit `confirm=True`.
- New module `src/visprobe/cost.py`: `estimate()`, `format_estimate()`, `is_expensive()`.

### Changed — internal refactor (breaking)
- `AttackFactory.create(name, eps, ...)` → `attacks.build(name, eps, ...)` (module function instead of factory class).
- `CheckpointManager` class → `checkpoint.*` module functions (`save_cell`, `load_cell`, `is_completed`, `load_all`, `save_metadata`, `load_metadata`).
- Five classes collapsed to three domain entities (`CompositionalExperiment`, `ModelMemoryManager`, `CompositionalResults`) plus two namespaces (`attacks`, `checkpoint`).
- ~600 LOC of dead code removed: `AutoAttackAPGDDLR`, `create_adaptive_attack`, `AdaptiveAttack`, `BatchMemoryOptimizer`, verbose memory-logging methods.

### Removed (breaking)
- `AttackFactory` class — use `attacks.build()`.
- `CheckpointManager` class — use `checkpoint` module functions.
- `CompositionalResults.plot_compositional()` — `matplotlib` is no longer a runtime dependency. Copy the snippet from `examples/plotting.ipynb` and edit to taste.
- `CompositionalExperiment(experiment_id=...)` kwarg — removed; resume now works purely off `checkpoint_dir` file existence (and actually works across processes, which it didn't before).
- `quick_test()` (already gone in 2.0; explicitly enumerated here for migration completeness).
- `matplotlib` from required dependencies.

### Fixed
- AutoAttack API drift: `aa.eps` AttributeError (modern AutoAttack stores eps on sub-attacks, not on the wrapper) and `aa.run_standard` rename to `aa.run_standard_evaluation`.
- Cross-process checkpoint resume was broken in v2.0 because each `CompositionalExperiment(...)` auto-generated a new `experiment_id`, namespacing checkpoints into a fresh subdirectory. Now resume hinges only on `checkpoint_dir`, so the second invocation actually picks up the first invocation's work.

### Migration from 2.0.0
| Removed in 3.0.0 | Replace with |
|---|---|
| `from visprobe import AttackFactory` | `from visprobe import attacks; attacks.build(name, eps, ...)` |
| `from visprobe.checkpoint import CheckpointManager` | `from visprobe import checkpoint` + module functions |
| `results.plot_compositional()` | Copy from `examples/plotting.ipynb` |
| `CompositionalExperiment(experiment_id=...)` | Drop the kwarg; resume works automatically |

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
