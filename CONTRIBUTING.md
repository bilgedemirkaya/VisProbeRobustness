# Contributing to VisProbe

Thanks for considering a contribution. This guide is short on purpose, because VisProbe is a small library and the contributor surface should match.

## What VisProbe is

A focused tool for two things:

1. Ranking your vision model against the RobustBench leaderboard under their exact protocol.
2. Sweeping a model across `environment × adversarial-attack × severity` to surface failure modes the leaderboard doesn't measure.

Read the [README](README.md) first; the rest of this file assumes you have.

## Where to contribute

Useful contributions, roughly ordered by how much we want them:

1. **Bug reports with reproducers.** Open a GitHub issue with the smallest snippet that reproduces the problem, plus your Python and PyTorch versions.
2. **Documentation improvements.** README clarity, docstrings, examples.
3. **New environmental perturbations** in `src/visprobe/perturbations.py`. Keep them simple callables with signature `(images, severity) -> images` where `severity=0` is a no-op.
4. **L2 / CIFAR-100 / common-corruptions leaderboard support.** See [ROADMAP.md](ROADMAP.md) for the v3.1 plan.
5. **Head-to-head feature** (v3.1 milestone in ROADMAP.md): top-k model loading plus side-by-side evaluation against the user's data.

We are *not* looking for:

- Complex search strategies (Bayesian, Binary, Adaptive). We deliberately cut those in v2.
- A web UI or dashboard.
- New abstractions, factories, or "manager" classes unless they earn their keep against the existing surface.

## Local setup

```bash
git clone https://github.com/bilgedemirkaya/VisProbeRobustness.git
cd VisProbeRobustness

python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

pip install -e ".[all,dev]"      # core + AutoAttack + RobustBench + pytest/black/isort/mypy
```

If `pip install autoattack` fails (the PyPI package occasionally lags), install it from GitHub:

```bash
pip install -e ".[robustbench,dev]"
pip install git+https://github.com/fra31/auto-attack
```

Run the test suite:

```bash
pytest tests/ -v
```

Smoke-check that the leaderboard snapshots load from your local install:

```bash
python -c "from visprobe import RobustBenchClient; print(RobustBenchClient('cifar10', 'Linf').snapshot_date)"
```

You should see a date string. If you see `FileNotFoundError`, the data files aren't reaching your install; check `git status src/visprobe/data/` and make sure the JSON snapshots are present.

## Branches and pull requests

```bash
git checkout -b fix/short-description
# ... edits ...
git commit -m "fix: short description of what changed"
git push origin fix/short-description
```

Open a PR against `main`. Keep one logical change per PR. Bundle README and docstring polish together if it's the same logical change.

## Code conventions

- Black, line length 100: `black src/ tests/`
- Type hints on every public function signature.
- Docstrings on public functions where the *why* is not obvious from the code. Skip docstrings on trivial helpers.
- No emoji in code, docstrings, or commit messages.
- Avoid adding new runtime dependencies. If you must add one, raise it in the PR description before writing the code so we can discuss the trade-off.

## Tests

- Tests live under `tests/`. Use pytest's `tmp_path` fixture for anything that touches disk.
- New code in the protocol gate (`leaderboard.py`), the attack builders (`attacks.py`), or any other safety-critical path must include tests that assert on **specific substring contracts** in error messages. If the message can silently regress, the gate is not doing its job. See `tests/test_leaderboard.py` for the pattern.
- CI runs the suite on Ubuntu, macOS, and Windows across Python 3.9 / 3.10 / 3.11. If your PR passes locally on one OS but fails CI, check whether `.gitignore` is dropping a file from the build, or whether the wheel is missing data shipped from `src/visprobe/data/`.

## Reporting bugs

Open a GitHub issue. Include:

1. What you ran.
2. What happened.
3. What you expected.
4. Python version, PyTorch version, OS.
5. The smallest snippet that reproduces the problem.

If you do not have a reproducer, that is okay. Say so. Half the work of fixing a bug is bisecting to a reproducer, so we appreciate having one when you can produce it.

## Code of conduct

Be civil. Do not be hostile. Do not share private information. We will close issues and PRs that violate this without much ceremony.

## Recognition

Contributors are listed in the GitHub repo's contributors graph. Significant contributions get credit in the relevant `CHANGELOG.md` entry.
