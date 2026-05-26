# VisProbe Roadmap

The current shipped surface (v2.0) is intentionally small: compositional eval, checkpointing, memory management, AutoAttack/APGD-CE, four perturbations, save/load results. Anything else gets debated against the rule *"every feature added is a feature that can break, that users will misuse, that we will need to maintain."*

The next planned release is **v3.0 — RobustBench leaderboard integration**.

---

## v3.0: RobustBench Integration

### Motivation

The killer use case for a deployment engineer is not "evaluate my model" — it is **"evaluate my model and tell me where it would rank against the best published models."** Without leaderboard context, robust accuracy is a number with no meaning. With it, VisProbe becomes the tool a robustness researcher uses to contextualize their results before writing them up.

### Design constraint: ranking must not be misleading

RobustBench leaderboards are only valid under their exact protocol — full test set, specific eps per threat model, full AutoAttack (not APGD-CE), no environmental perturbation. If we let arbitrary `CompositionalExperiment` runs claim a rank, every misconfiguration becomes a misleading number. The integration therefore gates ranking behind a strict protocol check rather than bolting it onto every run.

### Two distinct features

| Feature | Question it answers | Source of truth | Caveat |
|---|---|---|---|
| **A. Official rank** | "Where do I sit on the published leaderboard?" | Snapshot JSON | Only valid under exact RobustBench protocol |
| **B. Head-to-head on your data** | "On my deployment data, am I better than published robust models?" | Re-evaluation of downloaded weights on user's images | NOT the official rank — same number on different data |

Both ship in Phase 1. The distinction is enforced in the output format so users cannot conflate them.

### Module layout

```
src/visprobe/
├── leaderboard.py            # NEW (~350 LOC)
│   ├── RobustBenchClient     # snapshot reader, rank(), top_k()
│   ├── LeaderboardComparison # dataclass: rank + neighbors
│   └── HeadToHeadComparison  # dataclass: side-by-side table
├── model_zoo.py              # NEW (~150 LOC, lazy-imports robustbench)
│   ├── load_robustbench_model(name, dataset, threat)
│   └── load_top_k(k, dataset, threat) -> Dict[name, model]
├── data/
│   ├── robustbench_cifar10_linf.json
│   └── robustbench_imagenet_linf.json
├── results.py                # MODIFIED: add compare_to_leaderboard()
└── experiment.py             # MODIFIED: add robustbench_eval() + compare_against_top_k()
scripts/
└── refresh_leaderboard.py    # maintainer-only snapshot refresher
tests/
└── test_leaderboard.py
```

### Data source: frozen snapshot

The leaderboard is shipped as a JSON snapshot inside the package, refreshed by a maintainer-run script. Format:

```json
{
  "dataset": "cifar10",
  "threat": "Linf",
  "eps": 0.03137254901960784,
  "snapshot_date": "2026-05-01",
  "entries": [
    {
      "rank": 1,
      "name": "Wang2023Better_WRN-70-16",
      "paper": "Better Diffusion Models Further Improve...",
      "venue": "ICML 2023",
      "architecture": "WideResNet-70-16",
      "clean_acc": 0.9305,
      "robust_acc": 0.7088
    },
    ...
  ]
}
```

Why snapshot over a live `robustbench` dep for *leaderboard data*: zero new runtime deps, fully offline, deterministic, immune to network flakiness. The leaderboard moves on a months timescale so staleness is acceptable; `snapshot_date` is printed in every comparison output so users see freshness. A maintainer-run script regenerates the JSON by scraping RobustBench's `model_info/` directory on GitHub.

### Model zoo: optional dep

For Feature B, we need *weights*, not just numbers. Those come from the `robustbench` package's `load_model()`. It is an optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
robustbench = ["robustbench>=1.1"]
```

`model_zoo.py` lazy-imports it and raises a helpful install message if missing.

### Public API

```python
# ---- Feature A: official rank (strict protocol) ----
from visprobe import robustbench_eval, RobustBenchClient

result = robustbench_eval(
    model,
    dataset="cifar10",
    threat="Linf",
    data_dir="./cifar",  # full RobustBench test split is loaded for you
)
# Returns an EvaluationResult tagged with protocol="robustbench"

comparison = results.compare_to_leaderboard(
    model_name="my_model",
    dataset="cifar10",
    threat="Linf",
)
# Returns rank, percentile, 3 neighbors above + 3 below.
# Raises ProtocolError if the eval was not run under robustbench_eval.

# ---- Feature B: head-to-head on user data ----
from visprobe import CompositionalExperiment

experiment = CompositionalExperiment(
    models={"my_model": model}, images=images, labels=labels, ...
)
results = experiment.run()
h2h = experiment.compare_against_top_k(k=5, dataset="cifar10", threat="Linf")
# Downloads top-5 published models, evaluates each on the same images/labels,
# returns a side-by-side table including all 6 models, clearly labeled
# "head-to-head on user data" so it cannot be mistaken for the official rank.
```

### Key implementation notes

1. **Protocol validation** for Feature A checks: full test set size (10k CIFAR / 5k ImageNet), eps matches threat (8/255 Linf, 0.5 L2), `attack="autoattack-standard"` (not APGD-CE), no env perturbation. Any mismatch raises `ProtocolError` with the specific violation.
2. **`robustbench_eval` auto-pins** all of the above so the common case is a one-liner.
3. **Head-to-head reuses `ModelMemoryManager`** — top-k published models are loaded one at a time and swapped, exactly like user-supplied models. No new memory machinery.
4. **Output flags.** Every head-to-head output carries a `data_source: "user"` flag in the rendered table so the result cannot be mistaken for the official RobustBench rank.
5. **Snapshot refresher** is maintainer-only, lives under `scripts/`, parses RobustBench's GitHub `model_info/` directory and writes the JSON files committed to the repo.
6. **Snapshot freshness is visible.** Every comparison printout includes `Snapshot: 2026-05-01 (Wang2023Better and 66 others)` so staleness is never hidden.

### Phase 1 scope (ships in v3.0)

- Snapshot JSON for CIFAR-10 Linf and ImageNet Linf.
- `RobustBenchClient` (rank, top_k, neighbors).
- `compare_to_leaderboard()` on `CompositionalResults` with protocol validation.
- `robustbench_eval()` one-liner helper.
- `model_zoo.py` with `load_robustbench_model()` and `load_top_k()` behind the optional `robustbench` extra.
- `compare_against_top_k()` on `CompositionalExperiment`.
- Test coverage for snapshot parsing, rank calc, and protocol validation.

### Explicitly out of scope (for now)

- Live network fetch of leaderboard data — snapshot only.
- L2 / CIFAR-100 / common-corruptions leaderboards — add once Phase 1 is validated.
- Submission JSON generator (a file users could PR to RobustBench).
- Custom model-zoo caching — defer to `robustbench`'s own caching.

### Risks

- **`robustbench` package API drift.** Their `load_model` signature has shifted historically. We pin to a known-good range and test against it. ~1 day of pain if they break it.
- **Head-to-head GPU cost.** Re-evaluating 5 ImageNet robust models with full AutoAttack on a user's 1000-image set is hours of GPU time. Default `k=5` but document the cost and allow `k=3` or by-name selection.
- **User confusion** between "official rank" and "head-to-head on your data." Mitigated by separate methods, separate output formats, and explicit labels.

---

## After v3.0

Open questions, not commitments:

- L2 and common-corruptions leaderboards.
- A submission helper that produces RobustBench-compatible JSON.
- Per-class robustness breakdown in `CompositionalResults`.

Anything beyond the above will be added only when a concrete user need shows up. New features need to clear the same bar: simpler total system, real problem, no hidden complexity.
