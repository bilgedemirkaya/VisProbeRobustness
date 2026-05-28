# Testing

## Surface

| Suite | Tests | Covers |
|---|---:|---|
| `tests/test_visprobe.py` | 10 | v2 core: checkpoint, memory, perturbations, `attacks.build`, results save/load, experiment resume |
| `tests/test_leaderboard.py` | 120 | v3 leaderboard, protocol gate, `robustbench_eval`, `compare_to_leaderboard` |
| **Total** | **130** | All green on Ubuntu/macOS/Windows × Python 3.9/3.10/3.11 |

```bash
pytest tests/
```

## What each section covers

### Protocol gate — `validate_protocol`

The safety mechanism. The user's first impression of how strict VisProbe is. One test per failure mode, each asserting a specific substring of the error message so message wording is pinned, not vibes-checked.

- Missing `protocol=robustbench` metadata → message names `use robustbench_eval()`
- Wrong `n_samples` → message includes `n_samples=` and `expected`
- Wrong `eps` → message includes `eps=`
- Wrong attack → message names the expected `autoattack-standard`
- Env perturbation present → message includes `severity=`
- Compound violations all listed (no fail-fast)
- Float tolerance: `1e-12` passes, `1e-7` fails. Symmetric on eps and severity.
- Unknown `(dataset, threat)` raises `ValueError`, not `ProtocolError` — distinguishes "user error" from "protocol violation" so callers can handle them differently.

### `RobustBenchClient` — snapshot reader

33 tests. Rank math, top-k, neighbors, schema versioning, entry validation, defensive normalization.

Rank math edge cases:
- Better than all entries → rank 1
- Worse than all entries → rank `total + 1`, OR ties with last entry if a Standard baseline is present in the snapshot (real CIFAR-10 case)
- Tied with rank-1 entry → rank 1 (competition ranking)
- Multiple ties → only strictly-better entries count

Type rejection (these would silently work without explicit checks because Python's loose semantics):
- `bool` rejected as `robust_acc` input (bool is `int` subclass)
- `bool` rejected as `rank` field in entries
- `NaN` rejected (uses `x != x` check)

Schema + entry validation:
- `schema_version` rejects `bool`, `0`, negatives, and anything `> _MAX_KNOWN_SCHEMA`
- Per-entry: null `robust_acc`, missing `name`, empty `name`, out-of-range `clean_acc`, `bool` as rank — each raises with the bad field named
- `eps` rejected as non-numeric or `bool`

Defensive behavior:
- Snapshot re-sorted and re-ranked on load (defends against hand-edited files)
- `top_k` returns fresh dict copies (caller mutation cannot poison internal state)
- `from_dict` does not mutate the caller's dicts (verified by identity check)

### `LeaderboardComparison` — rendering

28 tests on the render contract.

- Required fields present in output: model name, dataset, threat, snapshot date, protocol attack, eps
- Percentile uses the colloquial "top X%" reading: rank 1 of 100 → top 1.0%, not 100.0%
- Section collapse: `Neighbors above` section disappears when at rank 1; `Neighbors below` disappears at last rank
- Frozen contract: `neighbors_above` and `neighbors_below` stored as tuples, mutation raises `AttributeError`
- `eps` rendered as `N/255` when applicable (Linf convention), float otherwise — `_format_eps_for_display` handles both
- Format stability: byte-identical output for byte-identical inputs

### `robustbench_eval` — Feature A entry point

17 tests.

- `confirm=False` raises `RuntimeError` carrying the full cost message. No expensive work done before the gate.
- Cost message names the sample count, dataset, threat, and "AutoAttack"
- Unknown `(dataset, threat)` raises `ValueError` *before* the cost gate, so users fix typos first
- Case-insensitive inputs — `"CIFAR10"` / `"linf"` work identically to `"cifar10"` / `"Linf"`
- Missing `robustbench` package → `ImportError` with `pip install visprobe[robustbench]` in the message
- `_tag_robustbench_metadata` unit-tested in isolation: adds protocol/dataset/threat keys, preserves existing keys, handles `metadata=None`
- End-to-end: a tagged result passes `validate_protocol`

### `compare_to_leaderboard` — Feature A wiring (the M7 integration)

13 tests.

- Happy path against the *real shipped* CIFAR-10 snapshot
- Returns `LeaderboardComparison` carrying the protocol fields (`attack`, `eps`) from the underlying cell
- Snapshot date comes from the shipped data (not the test execution date — CI auto-refresh would otherwise corrupt this assertion)
- Top-score (accuracy = 1.0) → rank 1
- Bottom-score (accuracy = 0.0) → ties with the undefended `Standard` baseline; rank == total
- Missing cell → `ValueError` naming `robustbench_eval` as the fix
- Protocol violation (untagged cell, wrong eps, env perturbation) → `ProtocolError` before any rank computation
- End-to-end render contains all paper-pasteable fields (rank, accuracy, protocol, snapshot date, both neighbor sections)

## A TDD pattern that paid off

During M7 implementation, the test `test_compare_to_leaderboard_returns_correct_rank_for_bottom_score` asserted `rank == total + 1` for an accuracy of 0.0. It failed.

Initial reading: bug in `RobustBenchClient.rank`. Actual cause: the shipped RobustBench leaderboard includes an undefended `Standard` baseline at `robust_acc = 0.0` (rank 99). Under competition ranking, accuracy = 0.0 ties with the Standard entry, so rank = total, not total + 1.

The implementation was right. The test's mental model of the world was wrong. Fixed the test, learned the property, moved on.

The discipline: when a test fails, ask first whether the test's assumption about the world is the bug, not just the implementation. Both are equally likely.

## CI

### `.github/workflows/build.yml` — runs on push/PR

- Matrix: Ubuntu, macOS, Windows × Python 3.9, 3.10, 3.11 (9 combinations)
- Builds the wheel
- Installs the built wheel (cross-platform: uses `glob` instead of bash globbing so it works on Windows)
- Smoke-tests the full v3 public API and verifies the leaderboard snapshots load from the installed wheel
- Runs `pytest tests/ -v`
- On GitHub Release: publishes to PyPI via `PYPI_API_TOKEN` repo secret

### `.github/workflows/refresh-leaderboard.yml` — Mondays 06:00 UTC

- Runs `scripts/refresh_leaderboard.py` against the upstream RobustBench repo
- Opens a PR only if `src/visprobe/data/*.json` actually changed
- Refresher preserves `snapshot_date` when entries are byte-identical, so date-only PRs cannot happen
- Manually triggerable from the Actions tab via `workflow_dispatch`

### `.github/workflows/security.yml`

- Bandit scanner on push/PR + weekly cron

### `.github/workflows/lint.yml`

- Black, isort, flake8 on push/PR with `continue-on-error: true` (advisory, not blocking)
