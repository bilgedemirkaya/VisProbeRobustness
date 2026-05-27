"""
RobustBench leaderboard integration.

This module is the *safety mechanism* for v3's ranking feature. Its central
function — ``validate_protocol`` — is what makes ranking against RobustBench
honest: it refuses to return a rank unless the user's evaluation matches
RobustBench's published protocol exactly.

There is intentionally no warn-and-continue path. Either the eval matches the
strict protocol or no rank is returned. Misconfiguration becomes a loud error
instead of a misleading number.

Public surface (M3 scope):
    ProtocolError      — raised when an eval does not satisfy the protocol
    validate_protocol  — the gate function; call it before producing a rank
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import EvaluationResult


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
