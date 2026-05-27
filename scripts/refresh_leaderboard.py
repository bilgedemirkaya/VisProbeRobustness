#!/usr/bin/env python
"""
Refresh VisProbe's RobustBench snapshot files.

Maintainer-only script. Downloads RobustBench's repository tarball, extracts
the ``model_info/<dataset>/<threat>/*.json`` files, transforms them into
VisProbe's schema v1, and writes:

    src/visprobe/data/robustbench_cifar10_linf.json
    src/visprobe/data/robustbench_imagenet_linf.json

Run with:

    python scripts/refresh_leaderboard.py

Network only — one HTTPS GET for the tarball. No GitHub auth needed.
Idempotent: re-running with the same upstream state writes byte-identical
output (modulo `snapshot_date`).

CI invokes this via .github/workflows/refresh-leaderboard.yml (M12); a diff in
src/visprobe/data/*.json opens a PR for maintainer review.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Optional

TARBALL_URL = "https://github.com/RobustBench/robustbench/archive/refs/heads/master.tar.gz"
TARBALL_PREFIX = "robustbench-master"
DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "visprobe" / "data"
SCHEMA_VERSION = 1

# (dataset, threat) pairs we ship snapshots for. Add to this list to expand
# coverage; both the snapshot file and the leaderboard._PROTOCOL spec table
# need matching entries.
SUPPORTED = [
    ("cifar10",  "Linf", 8 / 255),
    ("imagenet", "Linf", 4 / 255),
]


# ---------------------------------------------------------------------------
# Parsers — RobustBench stores accuracies as percentage strings ("93.25") and
# eps as fractions ("8/255"). Coerce defensively; malformed entries are dropped
# with a warning rather than failing the whole refresh.
# ---------------------------------------------------------------------------

def parse_percentage(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value) / 100.0
    s = str(value).strip().rstrip("%")
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def parse_eps(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        if "/" in s:
            num, denom = s.split("/")
            return float(num) / float(denom)
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Fetch + transform
# ---------------------------------------------------------------------------

def fetch_tarball(url: str) -> tarfile.TarFile:
    print(f"Fetching {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "visprobe-refresh-leaderboard"})
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = response.read()
    print(f"  downloaded {len(payload) / (1024 * 1024):.1f} MB")
    return tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz")


def transform_entries(tar: tarfile.TarFile, dataset: str, threat: str) -> list[dict]:
    """Pull every model_info JSON for (dataset, threat) and project to v1 schema.

    Sorts the result by robust_acc descending and assigns 1-indexed ranks.
    """
    prefix = f"{TARBALL_PREFIX}/model_info/{dataset}/{threat}/"
    rows: list[dict] = []

    for member in tar.getmembers():
        if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".json"):
            continue
        model_name = Path(member.name).stem
        try:
            data = json.loads(tar.extractfile(member).read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"  WARN: cannot parse {member.name}: {e}")
            continue

        clean_acc = parse_percentage(data.get("clean_acc"))
        # Prefer the published AutoAttack number; fall back to "reported" if AA missing.
        robust_acc = parse_percentage(data.get("AA")) or parse_percentage(data.get("reported"))
        if clean_acc is None or robust_acc is None:
            print(f"  WARN: skipping {model_name} (missing clean_acc or robust_acc)")
            continue

        rows.append({
            "name": model_name,
            "paper": str(data.get("link", "") or ""),
            "venue": str(data.get("venue", "") or ""),
            "architecture": str(data.get("architecture", "") or ""),
            "clean_acc": round(clean_acc, 6),
            "robust_acc": round(robust_acc, 6),
        })

    rows.sort(key=lambda r: r["robust_acc"], reverse=True)

    # Emit in a stable field order for clean diffs.
    return [
        {
            "rank": i,
            "name": r["name"],
            "paper": r["paper"],
            "venue": r["venue"],
            "architecture": r["architecture"],
            "clean_acc": r["clean_acc"],
            "robust_acc": r["robust_acc"],
        }
        for i, r in enumerate(rows, start=1)
    ]


def write_snapshot(dataset: str, threat: str, eps: float, entries: list[dict], out_path: Path) -> None:
    """Write the snapshot, preserving ``snapshot_date`` when entries are unchanged.

    The weekly CI workflow refreshes this file every Monday. Without this
    check, ``snapshot_date`` would update every run and produce a spurious
    PR with only a date diff. Preserving the old date when nothing else
    changed makes the CI quiet by default — PRs land only when RobustBench
    itself published something new.
    """
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "threat": threat,
        "eps": eps,
        "snapshot_date": date.today().isoformat(),
        "entries": entries,
    }

    if out_path.exists():
        try:
            with open(out_path) as f:
                old = json.load(f)
        except (json.JSONDecodeError, OSError):
            old = None  # corrupt or unreadable; overwrite with today's date
        if (
            isinstance(old, dict)
            and old.get("entries") == entries
            and old.get("eps") == eps
            and old.get("dataset") == dataset
            and old.get("threat") == threat
            and old.get("schema_version") == SCHEMA_VERSION
        ):
            snapshot["snapshot_date"] = old.get("snapshot_date", snapshot["snapshot_date"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")  # POSIX-friendly trailing newline


# ---------------------------------------------------------------------------
# Diff summary
# ---------------------------------------------------------------------------

def diff_summary(old_path: Path, new_entries: list[dict]) -> None:
    if not old_path.exists():
        print(f"  new snapshot: {len(new_entries)} entries")
        return

    with open(old_path) as f:
        old = json.load(f)
    old_entries = old.get("entries", [])
    old_by_name = {e["name"]: e for e in old_entries}
    new_by_name = {e["name"]: e for e in new_entries}

    added = sorted(set(new_by_name) - set(old_by_name))
    removed = sorted(set(old_by_name) - set(new_by_name))
    rank_changes = [
        (name, old_by_name[name]["rank"], new_by_name[name]["rank"])
        for name in sorted(set(old_by_name) & set(new_by_name))
        if old_by_name[name]["rank"] != new_by_name[name]["rank"]
    ]

    print(f"  {len(new_entries)} entries (was {len(old_entries)})")
    if added:
        sample = ", ".join(added[:5])
        more = " ..." if len(added) > 5 else ""
        print(f"  + added   ({len(added)}): {sample}{more}")
    if removed:
        sample = ", ".join(removed[:5])
        more = " ..." if len(removed) > 5 else ""
        print(f"  - removed ({len(removed)}): {sample}{more}")
    if rank_changes:
        print(f"  ~ rank changes ({len(rank_changes)}): showing first 5")
        for name, old_r, new_r in rank_changes[:5]:
            print(f"    {name}: {old_r} -> {new_r}")
    if not (added or removed or rank_changes):
        print("  (no changes to ranking)")


def delete_samples() -> None:
    for sample in DATA_DIR.glob("*.sample.json"):
        sample.unlink()
        print(f"Deleted sample file: {sample.relative_to(DATA_DIR.parent.parent.parent)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with fetch_tarball(TARBALL_URL) as tar:
        for dataset, threat, eps in SUPPORTED:
            print(f"\nProcessing {dataset}/{threat} ...")
            entries = transform_entries(tar, dataset, threat)
            if not entries:
                print(f"  ERROR: no entries found for {dataset}/{threat}")
                sys.exit(1)
            out_path = DATA_DIR / f"robustbench_{dataset}_{threat.lower()}.json"
            diff_summary(out_path, entries)
            write_snapshot(dataset, threat, eps, entries, out_path)
            print(f"  wrote {out_path.relative_to(DATA_DIR.parent.parent.parent)}")

    print()
    delete_samples()
    print("\nDone.")


if __name__ == "__main__":
    main()
