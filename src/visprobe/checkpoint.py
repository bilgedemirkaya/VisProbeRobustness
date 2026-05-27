"""
Per-cell checkpoint persistence for compositional experiments.

Module-level functions; no state held except the caller-supplied root path.
A cell is identified by (model, scenario, severity) and stored as a pickle file
named ``{model}_{scenario}_{severity:.3f}.pkl`` under the supplied results_dir.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


def _cell_path(results_dir: Path, model: str, scenario: str, severity: float) -> Path:
    return Path(results_dir) / f"{model}_{scenario}_{severity:.3f}.pkl"


def save_cell(
    result: Any,
    results_dir: Path,
    model: str,
    scenario: str,
    severity: float,
) -> None:
    """Persist a single cell result."""
    p = _cell_path(results_dir, model, scenario, severity)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(result, f)


def load_cell(
    results_dir: Path,
    model: str,
    scenario: str,
    severity: float,
) -> Optional[Any]:
    """Return the saved cell result, or None if it does not exist."""
    p = _cell_path(results_dir, model, scenario, severity)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def is_completed(
    results_dir: Path,
    model: str,
    scenario: str,
    severity: float,
) -> bool:
    return _cell_path(results_dir, model, scenario, severity).exists()


def load_all(results_dir: Path) -> Dict[str, Dict[str, Dict[float, Any]]]:
    """Scan ``results_dir`` and return nested {model: {scenario: {severity: result}}}."""
    results: Dict[str, Dict[str, Dict[float, Any]]] = {}
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return results
    for filepath in results_dir.glob("*.pkl"):
        parts = filepath.stem.split("_")
        if len(parts) < 3:
            continue
        model = "_".join(parts[:-2])
        scenario = parts[-2]
        severity = float(parts[-1])
        with open(filepath, "rb") as f:
            result = pickle.load(f)
        results.setdefault(model, {}).setdefault(scenario, {})[severity] = result
    return results


def save_metadata(metadata: Dict[str, Any], metadata_path: Path) -> None:
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(metadata_path: Path) -> Optional[Dict[str, Any]]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)
