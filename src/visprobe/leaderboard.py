"""
RobustBench leaderboard integration.

This module is the *safety mechanism* for v3's ranking feature. Its central
function — ``validate_protocol`` — is what makes ranking against RobustBench
honest: it refuses to return a rank unless the user's evaluation matches
RobustBench's published protocol exactly.

There is intentionally no warn-and-continue path. Either the eval matches the
strict protocol or no rank is returned. Misconfiguration becomes a loud error
instead of a misleading number.

Public surface:
    ProtocolError       — raised when an eval does not satisfy the protocol
    validate_protocol   — the gate function; call it before producing a rank
    RobustBenchClient   — read-only client over a frozen snapshot;
                          provides rank, top_k, neighbors
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .results import EvaluationResult


_MAX_KNOWN_SCHEMA = 1
"""Highest snapshot schema_version this code can read.

Bump in lockstep when ``scripts/refresh_leaderboard.py`` writes a schema
with newer field semantics. The ``RobustBenchClient`` constructor raises on
anything higher than this so an old visprobe install cannot silently
misread a future snapshot.
"""


# Float tolerance for any scalar protocol field (severity, eps).
# Picked tight enough to catch real misconfiguration (e.g. eps=4/255 vs 8/255)
# but loose enough to absorb fp drift from arithmetic in user code.
_TOL = 1e-9


# RobustBench's published protocol per (dataset, threat) pair.
# Values come straight from https://github.com/RobustBench/robustbench README.
# Update only when RobustBench itself changes its evaluation protocol.
_PROTOCOL = {
    ("cifar10",  "Linf"): {"n_samples": 10000, "eps": 8 / 255, "attack": "autoattack-standard"},
    ("imagenet", "Linf"): {"n_samples":  5000, "eps": 4 / 255, "attack": "autoattack-standard"},
}


class ProtocolError(Exception):
    """Raised when an ``EvaluationResult`` does not satisfy the RobustBench protocol.

    The message enumerates every violation and tells the user how to produce a
    compliant result. This is the primary failure mode users see from the
    ranking feature, so the message contract is pinned in tests; see
    ``tests/test_leaderboard.py``.
    """


def validate_protocol(result: "EvaluationResult", dataset: str, threat: str) -> None:
    """Raise ``ProtocolError`` if ``result`` does not satisfy RobustBench's
    protocol for ``(dataset, threat)``.

    The protocol covers four scalar checks and one tag check:
      * ``metadata['protocol'] == "robustbench"`` — proves the result came from
        ``robustbench_eval`` (M6), not an arbitrary ``CompositionalExperiment`` run.
      * ``n_samples`` matches the full RobustBench test split (10000 for CIFAR-10,
        5000 for ImageNet).
      * ``eps`` matches the threat model's published budget (within 1e-9).
      * ``metadata['attack'] == "autoattack-standard"`` — APGD-CE alone is not
        the published RobustBench protocol, no matter how convenient.
      * ``severity`` is within 1e-9 of zero — no environmental perturbation.

    Args:
        result: the EvaluationResult to validate.
        dataset: "cifar10" or "imagenet".
        threat: "Linf" (others added in v3.1).

    Raises:
        ValueError: if ``(dataset, threat)`` is not a known protocol pair.
            This indicates a programmer error, not a protocol violation.
        ProtocolError: if any protocol field does not match. Multi-line message
            lists every violation; do not split a single check across multiple calls.
    """
    key = (dataset, threat)
    if key not in _PROTOCOL:
        raise ValueError(
            f"No RobustBench protocol defined for dataset={dataset!r}, threat={threat!r}. "
            f"Known: {sorted(_PROTOCOL)}"
        )
    expected = _PROTOCOL[key]
    metadata = result.metadata or {}
    violations = []

    if metadata.get("protocol") != "robustbench":
        violations.append(
            "result is not tagged 'protocol=robustbench' "
            "— use robustbench_eval() to produce a protocol-compliant result"
        )

    if result.n_samples != expected["n_samples"]:
        violations.append(
            f"n_samples={result.n_samples}, expected {expected['n_samples']}"
        )

    if abs(result.eps - expected["eps"]) > _TOL:
        violations.append(
            f"eps={result.eps:.10f}, expected {expected['eps']:.10f}"
        )

    if metadata.get("attack") != expected["attack"]:
        violations.append(
            f"attack={metadata.get('attack')!r}, expected {expected['attack']!r}"
        )

    if abs(result.severity) > _TOL:
        violations.append(
            f"env perturbation present (severity={result.severity})"
        )

    if violations:
        raise ProtocolError(
            f"Cannot rank against RobustBench {dataset}/{threat} — protocol mismatch:\n  - "
            + "\n  - ".join(violations)
            + f"\n\nFix: use robustbench_eval(model, dataset={dataset!r}, threat={threat!r}) "
              "to produce a protocol-compliant result."
        )


# ---------------------------------------------------------------------------
# RobustBenchClient: read-only client over the shipped snapshots
# ---------------------------------------------------------------------------

def _validate_entries(entries: list, label: str) -> None:
    """Per-entry validation. Required fields: name (non-empty str), rank (int),
    clean_acc and robust_acc (numeric in [0,1]). Display fields (paper, venue,
    architecture) are not validated — they may be empty or missing.

    Raises ValueError naming the bad entry index and field. The label identifies
    the source (filename or test fixture).
    """
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{label} entries[{i}] is not a dict")

        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{label} entries[{i}].name must be a non-empty string (got {name!r})"
            )

        rank = entry.get("rank")
        # bool is an int subclass — explicitly reject so True/False can't pose as a rank.
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(
                f"{label} entries[{i}].rank must be int (got {type(rank).__name__})"
            )

        for field in ("clean_acc", "robust_acc"):
            value = entry.get(field)
            if value is None:
                raise ValueError(f"{label} entries[{i}].{field} is missing or null")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{label} entries[{i}].{field} must be numeric "
                    f"(got {type(value).__name__})"
                )
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{label} entries[{i}].{field} out of [0,1]: {value}"
                )


class RobustBenchClient:
    """Read-only client over a frozen RobustBench leaderboard snapshot.

    Loads ``src/visprobe/data/robustbench_<dataset>_<threat>.json`` via
    ``importlib.resources``, validates ``schema_version`` and every entry,
    and exposes ranking helpers.

    Entries are sorted by ``robust_acc`` descending; ranks are 1-indexed in
    that order. Rank semantics are **competition ranking** — tied entries
    share a rank (rank N means N-1 entries are strictly better).

    Args:
        dataset: "cifar10" or "imagenet"
        threat: "Linf" (others land in v3.1)

    Raises:
        FileNotFoundError: no snapshot ships for this ``(dataset, threat)``.
        ValueError: snapshot ``schema_version`` is newer than this code knows,
            or any entry fails per-field validation.
    """

    def __init__(self, dataset: str, threat: str):
        self.dataset = dataset
        self.threat = threat
        fname = f"robustbench_{dataset}_{threat.lower()}.json"
        resource = files("visprobe") / "data" / fname
        try:
            content = resource.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No snapshot for {dataset}/{threat}: expected visprobe/data/{fname}"
            ) from e
        self._init_from_data(json.loads(content), label=fname)

    @classmethod
    def from_dict(cls, data: dict, *, label: str = "<dict>") -> "RobustBenchClient":
        """Construct a client from an in-memory dict, bypassing file loading.

        Used by tests to exercise validation against synthetic snapshots
        without writing fixture files to disk.
        """
        obj = cls.__new__(cls)
        obj.dataset = data.get("dataset", "")
        obj.threat = data.get("threat", "")
        obj._init_from_data(data, label=label)
        return obj

    def _init_from_data(self, data: dict, *, label: str) -> None:
        version = data.get("schema_version")
        if not isinstance(version, int) or version > _MAX_KNOWN_SCHEMA:
            raise ValueError(
                f"Snapshot {label} has schema_version={version!r}; "
                f"this visprobe knows up to schema_version={_MAX_KNOWN_SCHEMA}. "
                "Upgrade visprobe to read newer snapshots."
            )

        entries = data.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Snapshot {label} has no entries")

        _validate_entries(entries, label)

        # Defensive: sort desc by robust_acc and re-assign ranks in case the
        # snapshot file was hand-edited and got the ranking out of sync with
        # the robust_acc ordering.
        entries = sorted(entries, key=lambda e: float(e["robust_acc"]), reverse=True)
        for i, e in enumerate(entries, start=1):
            e["rank"] = i

        self._entries = entries
        self._snapshot_date = str(data.get("snapshot_date", ""))
        self._eps = data.get("eps")

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def snapshot_date(self) -> str:
        return self._snapshot_date

    @property
    def eps(self) -> Optional[float]:
        return self._eps

    def entries(self) -> list[dict]:
        """All entries as a new list of dicts (caller may mutate without side-effects)."""
        return [dict(e) for e in self._entries]

    def rank(self, robust_acc: float) -> int:
        """Where ``robust_acc`` lands in this leaderboard, 1-indexed.

        Competition ranking: tied entries share a rank. ``rank(x) == r`` means
        ``r - 1`` entries are strictly better than ``x``.

        Examples — leaderboard ``[0.71, 0.68, 0.66, 0.59]``:
            rank(0.75) == 1   # better than all
            rank(0.71) == 1   # ties with rank 1
            rank(0.69) == 2   # between ranks 1 and 2
            rank(0.50) == 5   # worse than all; len + 1
        """
        if not isinstance(robust_acc, (int, float)) or isinstance(robust_acc, bool):
            raise ValueError(f"robust_acc must be a real number, got {robust_acc!r}")
        if robust_acc != robust_acc:  # NaN check (NaN != NaN)
            raise ValueError("robust_acc must not be NaN")
        return 1 + sum(1 for e in self._entries if e["robust_acc"] > robust_acc)

    def top_k(self, k: int) -> list[dict]:
        """First ``k`` entries by robust_acc descending. ``k > len(self)`` returns all."""
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")
        return [dict(e) for e in self._entries[:k]]

    def neighbors(self, robust_acc: float, k: int = 3) -> tuple[list[dict], list[dict]]:
        """Entries ranked just above and just below ``robust_acc``.

        Returns ``(above, below)``:
          * ``above`` — up to k entries with strictly higher robust_acc, in
            DESC order (matches natural leaderboard display: highest first).
          * ``below`` — up to k entries with strictly lower robust_acc, in
            DESC order (closest below appears first).

        Tied entries are excluded from both lists — they are peers, not
        neighbors. To find peers, filter ``entries()`` directly.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError(f"k must be a non-negative int, got {k!r}")
        if k == 0:
            return [], []
        above = [e for e in self._entries if e["robust_acc"] > robust_acc]
        below = [e for e in self._entries if e["robust_acc"] < robust_acc]
        return ([dict(e) for e in above[-k:]], [dict(e) for e in below[:k]])
