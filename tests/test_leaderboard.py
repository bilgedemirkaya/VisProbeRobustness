"""
Tests for the protocol gate (v3 M3).

Every test that exercises a protocol violation asserts a specific substring of
the error message. The protocol gate is the user's first impression of the
safety mechanism — pin the message contract here so it cannot silently regress.
"""

import numpy as np
import pytest

from visprobe.leaderboard import ProtocolError, validate_protocol
from visprobe.results import EvaluationResult


def _result(**overrides) -> EvaluationResult:
    """Construct an EvaluationResult that PASSES cifar10/Linf protocol by default.

    Each test overrides exactly the field it wants to break, so failures are
    localized to the field under test.
    """
    defaults = dict(
        accuracy=0.7,
        mean_confidence=0.8,
        mean_loss=0.5,
        correct_mask=np.array([True]),
        predictions=np.array([0]),
        confidences=np.array([0.8]),
        model_name="my_model",
        scenario="none",
        severity=0.0,
        eps=8 / 255,
        n_samples=10000,
        metadata={"protocol": "robustbench", "attack": "autoattack-standard"},
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_protocol_happy_path_cifar10():
    """A correctly-tagged result passes silently — no exception, no return value."""
    assert validate_protocol(_result(), "cifar10", "Linf") is None


def test_protocol_happy_path_imagenet():
    """ImageNet protocol passes with its own n_samples and eps."""
    r = _result(n_samples=5000, eps=4 / 255)
    assert validate_protocol(r, "imagenet", "Linf") is None


# ---------------------------------------------------------------------------
# Individual violations — each asserts a substring of the error message
# ---------------------------------------------------------------------------

def test_protocol_missing_tag():
    """Result without 'protocol=robustbench' metadata is rejected."""
    r = _result(metadata={"attack": "autoattack-standard"})  # no protocol key
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    assert "use robustbench_eval()" in str(exc.value)


def test_protocol_wrong_n_samples():
    r = _result(n_samples=1000)
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    msg = str(exc.value)
    assert "n_samples=" in msg
    assert "expected " in msg


def test_protocol_wrong_eps():
    """ImageNet eps used on CIFAR-10 protocol fails."""
    r = _result(eps=4 / 255)
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    assert "eps=" in str(exc.value)


def test_protocol_wrong_attack():
    """APGD-CE is faster but is NOT the published RobustBench protocol."""
    r = _result(metadata={"protocol": "robustbench", "attack": "autoattack-apgd-ce"})
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    assert "autoattack-standard" in str(exc.value)


def test_protocol_env_perturbation_present():
    """Any non-zero severity means env was added to the eval. Reject."""
    r = _result(severity=0.5)
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    assert "severity=" in str(exc.value)


# ---------------------------------------------------------------------------
# Float tolerance — symmetric on eps and severity
# ---------------------------------------------------------------------------

def test_protocol_eps_tolerance_passes_at_1e_minus_12():
    """1e-12 drift on eps is well within fp tolerance and must pass."""
    r = _result(eps=8 / 255 + 1e-12)
    assert validate_protocol(r, "cifar10", "Linf") is None


def test_protocol_eps_tolerance_fails_at_1e_minus_7():
    """1e-7 drift exceeds tolerance — that's bigger than fp noise, real drift."""
    r = _result(eps=8 / 255 + 1e-7)
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    assert "eps=" in str(exc.value)


def test_protocol_severity_tolerance_passes_at_1e_minus_12():
    r = _result(severity=1e-12)
    assert validate_protocol(r, "cifar10", "Linf") is None


def test_protocol_severity_tolerance_fails_at_1e_minus_7():
    r = _result(severity=1e-7)
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    assert "severity=" in str(exc.value)


# ---------------------------------------------------------------------------
# Compound failures — every violation must be listed
# ---------------------------------------------------------------------------

def test_protocol_multiple_violations_all_listed():
    """Don't fail-fast on the first violation. Surface every problem at once
    so the user fixes everything in one round-trip instead of N."""
    r = _result(
        n_samples=500,
        eps=0.001,
        metadata={"protocol": "robustbench", "attack": "pgd"},
    )
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    msg = str(exc.value)
    assert "n_samples=" in msg
    assert "eps=" in msg
    assert "autoattack-standard" in msg


def test_protocol_error_message_contains_fix_instruction():
    """Every ProtocolError ends with a 'Fix:' line naming the helper to use."""
    r = _result(n_samples=42)
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    msg = str(exc.value)
    assert "Fix:" in msg
    assert "robustbench_eval" in msg
    assert "cifar10" in msg


# ---------------------------------------------------------------------------
# Programmer errors vs protocol errors
# ---------------------------------------------------------------------------

def test_protocol_unknown_dataset_raises_value_error_not_protocol_error():
    """Unknown (dataset, threat) is a bug in the caller, not a protocol mismatch.
    Distinguishing the two lets users handle them differently:
        try:
            validate_protocol(...)
        except ProtocolError as e:
            # user needs to re-run with robustbench_eval
        except ValueError as e:
            # developer typed the wrong dataset name
    """
    r = _result()
    with pytest.raises(ValueError) as exc:
        validate_protocol(r, "unknown_dataset", "Linf")
    assert "unknown_dataset" in str(exc.value)


def test_protocol_metadata_none_is_treated_as_missing_tag():
    """EvaluationResult.metadata defaults to None — must not crash."""
    r = _result(metadata=None)
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    assert "use robustbench_eval()" in str(exc.value)


# ===========================================================================
# RobustBenchClient (M4)
# ===========================================================================

from visprobe.leaderboard import RobustBenchClient


def _snapshot(entries=None, schema_version=1, dataset="cifar10", threat="Linf"):
    """Build a synthetic snapshot dict for from_dict()-based tests."""
    if entries is None:
        # Default: 4-entry leaderboard with a gap-and-tie pattern useful for
        # rank-math edge cases. Sorted desc by robust_acc.
        entries = [
            {"rank": 1, "name": "A", "paper": "", "venue": "", "architecture": "",
             "clean_acc": 0.90, "robust_acc": 0.71},
            {"rank": 2, "name": "B", "paper": "", "venue": "", "architecture": "",
             "clean_acc": 0.88, "robust_acc": 0.68},
            {"rank": 3, "name": "C", "paper": "", "venue": "", "architecture": "",
             "clean_acc": 0.85, "robust_acc": 0.66},
            {"rank": 4, "name": "D", "paper": "", "venue": "", "architecture": "",
             "clean_acc": 0.80, "robust_acc": 0.59},
        ]
    return {
        "schema_version": schema_version,
        "dataset": dataset,
        "threat": threat,
        "eps": 8 / 255,
        "snapshot_date": "2026-05-27",
        "entries": entries,
    }


# ---- Basic load against shipped snapshots --------------------------------

def test_client_loads_shipped_cifar10_snapshot():
    """The real CIFAR-10 snapshot (from M2) loads cleanly."""
    c = RobustBenchClient("cifar10", "Linf")
    assert len(c) > 0
    assert c.snapshot_date  # non-empty
    assert abs(c.eps - 8 / 255) < 1e-9


def test_client_loads_shipped_imagenet_snapshot():
    c = RobustBenchClient("imagenet", "Linf")
    assert len(c) > 0
    assert abs(c.eps - 4 / 255) < 1e-9


def test_client_missing_snapshot_raises_file_not_found():
    with pytest.raises(FileNotFoundError) as exc:
        RobustBenchClient("cifar100", "Linf")
    assert "cifar100" in str(exc.value)


# ---- Rank math (M4 acceptance: tied / lower than all / higher than top) --

def test_rank_better_than_all():
    """robust_acc above the top entry returns rank 1."""
    c = RobustBenchClient.from_dict(_snapshot())
    assert c.rank(0.99) == 1


def test_rank_exactly_at_top():
    """robust_acc tied with rank-1 still returns rank 1 (competition ranking)."""
    c = RobustBenchClient.from_dict(_snapshot())
    assert c.rank(0.71) == 1


def test_rank_in_between():
    """robust_acc between rank-1 and rank-2 returns rank 2."""
    c = RobustBenchClient.from_dict(_snapshot())
    assert c.rank(0.69) == 2


def test_rank_tied_with_middle():
    """Tying rank-2 returns rank 2 (one entry strictly better)."""
    c = RobustBenchClient.from_dict(_snapshot())
    assert c.rank(0.68) == 2


def test_rank_worse_than_all():
    """Below the bottom entry returns len(client) + 1."""
    c = RobustBenchClient.from_dict(_snapshot())
    assert c.rank(0.10) == 5  # 4 entries, all strictly better


def test_rank_with_multiple_ties():
    """A run of ties only counts strictly-better entries."""
    entries = [
        {"rank": 1, "name": "A", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 0.70},
        {"rank": 2, "name": "B", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 0.65},
        {"rank": 3, "name": "C", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 0.65},
        {"rank": 4, "name": "D", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 0.65},
    ]
    c = RobustBenchClient.from_dict(_snapshot(entries=entries))
    # Tied with B/C/D — only A (0.70) is strictly better → rank 2
    assert c.rank(0.65) == 2


def test_rank_rejects_non_numeric():
    c = RobustBenchClient.from_dict(_snapshot())
    with pytest.raises(ValueError):
        c.rank("not a number")  # type: ignore[arg-type]


def test_rank_rejects_nan():
    c = RobustBenchClient.from_dict(_snapshot())
    with pytest.raises(ValueError, match="NaN"):
        c.rank(float("nan"))


def test_rank_rejects_bool():
    """Bool is a Python int subclass — explicit reject to avoid surprise."""
    c = RobustBenchClient.from_dict(_snapshot())
    with pytest.raises(ValueError):
        c.rank(True)  # type: ignore[arg-type]


# ---- top_k ----------------------------------------------------------------

def test_top_k_returns_first_k_in_desc_order():
    c = RobustBenchClient.from_dict(_snapshot())
    top = c.top_k(2)
    assert [e["name"] for e in top] == ["A", "B"]
    assert top[0]["robust_acc"] >= top[1]["robust_acc"]


def test_top_k_larger_than_total_returns_all():
    c = RobustBenchClient.from_dict(_snapshot())
    assert len(c.top_k(99)) == len(c) == 4


def test_top_k_rejects_zero_or_negative():
    c = RobustBenchClient.from_dict(_snapshot())
    with pytest.raises(ValueError):
        c.top_k(0)
    with pytest.raises(ValueError):
        c.top_k(-1)


def test_top_k_returns_copies_not_references():
    """Mutating a returned entry must not affect the client's state."""
    c = RobustBenchClient.from_dict(_snapshot())
    top = c.top_k(1)
    top[0]["name"] = "MUTATED"
    assert c.top_k(1)[0]["name"] == "A"


# ---- neighbors ------------------------------------------------------------

def test_neighbors_in_middle():
    """robust_acc between rank-2 and rank-3 returns both above and below."""
    c = RobustBenchClient.from_dict(_snapshot())
    above, below = c.neighbors(0.67, k=3)
    assert [e["name"] for e in above] == ["A", "B"]  # 0.71, 0.68 both strictly above
    assert [e["name"] for e in below] == ["C", "D"]  # 0.66, 0.59 both strictly below


def test_neighbors_at_top():
    """robust_acc better than all entries: above empty, below has top-k."""
    c = RobustBenchClient.from_dict(_snapshot())
    above, below = c.neighbors(0.99, k=2)
    assert above == []
    assert [e["name"] for e in below] == ["A", "B"]


def test_neighbors_at_bottom():
    """robust_acc worse than all: above has bottom-k closest, below empty."""
    c = RobustBenchClient.from_dict(_snapshot())
    above, below = c.neighbors(0.10, k=2)
    # bottom 2 closest above = C (0.66) and D (0.59); presented in desc order
    assert [e["name"] for e in above] == ["C", "D"]
    assert below == []


def test_neighbors_excludes_ties():
    """Tied entries are peers, not neighbors — appear in neither list."""
    c = RobustBenchClient.from_dict(_snapshot())
    above, below = c.neighbors(0.68, k=3)
    # 0.68 ties with B. B should NOT appear in either list.
    names_above = [e["name"] for e in above]
    names_below = [e["name"] for e in below]
    assert "B" not in names_above
    assert "B" not in names_below
    assert names_above == ["A"]
    assert names_below == ["C", "D"]


def test_neighbors_k_zero_returns_empty():
    c = RobustBenchClient.from_dict(_snapshot())
    above, below = c.neighbors(0.65, k=0)
    assert above == []
    assert below == []


def test_neighbors_rejects_negative_k():
    c = RobustBenchClient.from_dict(_snapshot())
    with pytest.raises(ValueError):
        c.neighbors(0.65, k=-1)


# ---- Schema version check (M4 acceptance) --------------------------------

def test_schema_version_too_new_raises():
    """Loading a future-version snapshot must fail loudly, not silently."""
    snap = _snapshot(schema_version=99)
    with pytest.raises(ValueError) as exc:
        RobustBenchClient.from_dict(snap, label="test.json")
    msg = str(exc.value)
    assert "schema_version" in msg
    assert "99" in msg
    assert "Upgrade visprobe" in msg or "upgrade" in msg.lower()


def test_schema_version_missing_raises():
    snap = _snapshot()
    del snap["schema_version"]
    with pytest.raises(ValueError, match="schema_version"):
        RobustBenchClient.from_dict(snap, label="test.json")


def test_schema_version_non_int_raises():
    snap = _snapshot()
    snap["schema_version"] = "1"  # string, not int
    with pytest.raises(ValueError, match="schema_version"):
        RobustBenchClient.from_dict(snap, label="test.json")


# ---- Per-entry validation (M4 acceptance) ---------------------------------

def test_corrupt_entry_null_robust_acc_raises_with_field_named():
    """robust_acc: null must raise with the bad field identified."""
    entries = [
        {"rank": 1, "name": "A", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": None},
    ]
    with pytest.raises(ValueError) as exc:
        RobustBenchClient.from_dict(_snapshot(entries=entries), label="bad.json")
    msg = str(exc.value)
    assert "robust_acc" in msg
    assert "bad.json" in msg


def test_corrupt_entry_missing_name_raises_with_field_named():
    entries = [
        {"rank": 1, "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 0.7},
    ]
    with pytest.raises(ValueError) as exc:
        RobustBenchClient.from_dict(_snapshot(entries=entries), label="bad.json")
    assert "name" in str(exc.value)


def test_corrupt_entry_empty_name_raises():
    entries = [
        {"rank": 1, "name": "", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 0.7},
    ]
    with pytest.raises(ValueError, match="name"):
        RobustBenchClient.from_dict(_snapshot(entries=entries), label="bad.json")


def test_corrupt_entry_robust_acc_out_of_range_raises():
    entries = [
        {"rank": 1, "name": "A", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 1.5},  # out of [0,1]
    ]
    with pytest.raises(ValueError) as exc:
        RobustBenchClient.from_dict(_snapshot(entries=entries), label="bad.json")
    msg = str(exc.value)
    assert "robust_acc" in msg
    assert "1.5" in msg


def test_corrupt_entry_clean_acc_negative_raises():
    entries = [
        {"rank": 1, "name": "A", "paper": "", "venue": "", "architecture": "",
         "clean_acc": -0.1, "robust_acc": 0.7},
    ]
    with pytest.raises(ValueError, match="clean_acc"):
        RobustBenchClient.from_dict(_snapshot(entries=entries), label="bad.json")


def test_corrupt_entry_bool_as_rank_raises():
    """bool is technically int — must not be accepted as a rank."""
    entries = [
        {"rank": True, "name": "A", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 0.7},
    ]
    with pytest.raises(ValueError, match="rank"):
        RobustBenchClient.from_dict(_snapshot(entries=entries), label="bad.json")


def test_corrupt_snapshot_no_entries_raises():
    with pytest.raises(ValueError, match="no entries"):
        RobustBenchClient.from_dict(_snapshot(entries=[]), label="bad.json")


# ---- Defensive normalization: ranks recomputed from robust_acc order ------

def test_ranks_recomputed_after_load():
    """If the snapshot's entries are out of order or have stale ranks, the
    client re-sorts and re-ranks defensively. Trust the data, not the labels."""
    entries = [
        # Deliberately mis-ordered and mis-ranked.
        {"rank": 99, "name": "Z", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.8, "robust_acc": 0.50},
        {"rank": 5, "name": "Y", "paper": "", "venue": "", "architecture": "",
         "clean_acc": 0.9, "robust_acc": 0.75},
    ]
    c = RobustBenchClient.from_dict(_snapshot(entries=entries), label="messy.json")
    top = c.top_k(2)
    assert top[0]["name"] == "Y"  # higher robust_acc
    assert top[0]["rank"] == 1     # re-ranked
    assert top[1]["name"] == "Z"
    assert top[1]["rank"] == 2


# ===========================================================================
# LeaderboardComparison (M5)
# ===========================================================================

from visprobe.leaderboard import LeaderboardComparison


def _neighbor(rank: int, name: str, robust_acc: float, clean_acc: float = 0.9):
    """Construct a neighbor entry dict matching RobustBenchClient.neighbors() output."""
    return {
        "rank": rank,
        "name": name,
        "paper": "",
        "venue": "",
        "architecture": "",
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
    }


def _comparison(**overrides) -> LeaderboardComparison:
    """A reasonable LeaderboardComparison for tests; override any field."""
    defaults = dict(
        model_name="my_model",
        robust_acc=0.6789,
        rank=14,
        total=99,
        neighbors_above=[
            _neighbor(11, "Wang2024Foo", 0.6912),
            _neighbor(12, "Bai2024Bar", 0.6856),
            _neighbor(13, "Cui2023Baz", 0.6823),
        ],
        neighbors_below=[
            _neighbor(15, "Gowal2023Qux", 0.6745),
            _neighbor(16, "Rebuffi2022Quux", 0.6710),
            _neighbor(17, "Carmon2021Corge", 0.6680),
        ],
        snapshot_date="2026-05-27",
        dataset="cifar10",
        threat="Linf",
        attack="autoattack-standard",
        eps=8 / 255,
    )
    defaults.update(overrides)
    return LeaderboardComparison(**defaults)


# ---- Required fields appear in the render --------------------------------

def test_comparison_render_includes_model_name():
    out = str(_comparison())
    assert "my_model" in out


def test_comparison_render_includes_dataset_and_threat():
    """Acceptance: dataset+threat present so the leaderboard origin is unambiguous."""
    out = str(_comparison())
    assert "cifar10" in out
    assert "Linf" in out


def test_comparison_render_includes_snapshot_date():
    """Acceptance: snapshot date is always shown so staleness is never hidden."""
    out = str(_comparison())
    assert "2026-05-27" in out


def test_comparison_render_includes_rank_and_total():
    out = str(_comparison())
    assert "14" in out
    assert "99" in out
    assert "of" in out  # "Rank: 14 of 99" idiom


def test_comparison_render_includes_robust_acc_to_4_decimals():
    out = str(_comparison(robust_acc=0.6789))
    assert "0.6789" in out


def test_comparison_render_includes_percentile():
    """Percentile uses the colloquial "top X%" formula: rank/total*100.
    Rank 14 of 99 → top ~14.1%."""
    out = str(_comparison())
    assert "14.1" in out or "14.2" in out  # rounding


# ---- Percentile math (colloquial "top X%" reading) ------------------------

def test_percentile_rank_1_of_100_is_top_1_percent():
    """The TDD test: rank 1 of 100 renders as 'top 1.0%', not 'top 100.0%'."""
    c = _comparison(rank=1, total=100, neighbors_above=[], neighbors_below=[])
    assert abs(c.percentile - 1.0) < 1e-9, (
        f"rank 1 of 100 must be 'top 1.0%'; got top {c.percentile}%"
    )
    assert "(top 1.0%)" in str(c)


def test_percentile_rank_total_of_total_is_top_100():
    """rank = total → top 100% (you're last, no one ranks worse)."""
    c = _comparison(rank=100, total=100)
    assert abs(c.percentile - 100.0) < 1e-9


def test_percentile_rank_median():
    """rank 50 of 100 → top 50% — the median."""
    c = _comparison(rank=50, total=100)
    assert abs(c.percentile - 50.0) < 1e-9


def test_percentile_zero_total_does_not_divide_by_zero():
    c = _comparison(rank=1, total=0)
    assert c.percentile == 0.0


# ---- Neighbors rendering --------------------------------------------------

def test_comparison_render_shows_all_neighbors_above():
    out = str(_comparison())
    assert "Wang2024Foo" in out
    assert "Bai2024Bar" in out
    assert "Cui2023Baz" in out


def test_comparison_render_shows_all_neighbors_below():
    out = str(_comparison())
    assert "Gowal2023Qux" in out
    assert "Rebuffi2022Quux" in out
    assert "Carmon2021Corge" in out


def test_comparison_render_shows_pp_deltas():
    """Each neighbor row carries a percentage-point delta vs the user's model."""
    out = str(_comparison())
    # +1.23 pp = (0.6912 - 0.6789) * 100; check the formatted token appears
    assert "pp" in out
    # At least one positive and one negative delta
    assert "+" in out  # neighbors above are positive deltas
    assert "-" in out  # neighbors below are negative deltas


def test_comparison_render_omits_above_section_when_rank_1():
    """User at rank 1 has no neighbors above — that section disappears entirely."""
    out = str(_comparison(rank=1, neighbors_above=[]))
    assert "Neighbors above" not in out
    assert "Neighbors below" in out  # below still present


def test_comparison_render_omits_below_section_at_bottom():
    out = str(_comparison(rank=99, neighbors_below=[]))
    assert "Neighbors below" not in out
    assert "Neighbors above" in out


def test_comparison_render_omits_both_sections_for_single_entry_leaderboard():
    """Degenerate case: only one entry exists, the user is it."""
    out = str(_comparison(rank=1, total=1, neighbors_above=[], neighbors_below=[]))
    assert "Neighbors above" not in out
    assert "Neighbors below" not in out
    assert "Rank:" in out  # core info still rendered


# ---- Format stability -----------------------------------------------------

def test_comparison_render_is_stable():
    """Two equal LeaderboardComparison objects render to byte-identical strings.
    Pins the format contract so unintended formatting drift fails CI."""
    a = _comparison()
    b = _comparison()
    assert str(a) == str(b)


def test_comparison_render_starts_with_titled_header():
    """Output is paper-pasteable: starts with a clearly-titled header line
    followed by a same-width ruler of '='."""
    out = str(_comparison())
    first_line = out.splitlines()[0]
    second_line = out.splitlines()[1]
    assert "RobustBench" in first_line
    assert set(second_line) == {"="}  # the ruler is all =
    assert len(first_line) == len(second_line)


def test_comparison_render_does_not_crash_on_long_neighbor_names():
    """The longest RobustBench identifiers are ~40 chars. Render must not
    raise — names are allowed to spill the column rather than truncate."""
    long_name = "Singh2023Revisiting_ConvNeXt-L-ConvStem-X-VeryLong"
    c = _comparison(
        neighbors_above=[_neighbor(13, long_name, 0.6823)],
    )
    out = str(c)
    assert long_name in out


# ---- Frozen dataclass invariants ------------------------------------------

def test_comparison_is_frozen():
    """LeaderboardComparison is immutable; mutation raises FrozenInstanceError."""
    import dataclasses
    c = _comparison()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.rank = 1  # type: ignore[misc]


def test_comparison_neighbors_stored_as_tuple():
    """Honor the frozen contract: neighbor sequences are tuples even when
    constructed from lists."""
    c = _comparison()
    assert isinstance(c.neighbors_above, tuple)
    assert isinstance(c.neighbors_below, tuple)


def test_comparison_neighbors_above_cannot_be_appended():
    """Honor the frozen contract: tuples don't have .append()."""
    c = _comparison()
    with pytest.raises(AttributeError):
        c.neighbors_above.append({})  # type: ignore[attr-defined]


# ---- Protocol line in render (post-M5 review fix) ------------------------

def test_render_includes_protocol_line():
    """The rendered output names the protocol that produced the number —
    AutoAttack standard, eps in N/255 format. Reinforces the protocol-gating
    thesis right where the user is looking."""
    out = str(_comparison())
    assert "Protocol:" in out
    assert "autoattack-standard" in out


def test_render_eps_formatted_as_linf_fraction():
    """Linf eps values are conventionally written as N/255; render that form."""
    out = str(_comparison(eps=8 / 255))
    assert "8/255" in out


def test_render_eps_4_over_255_for_imagenet():
    out = str(_comparison(threat="Linf", eps=4 / 255))
    assert "4/255" in out


def test_render_eps_falls_back_to_float_for_non_linf_values():
    """An eps that isn't N/255 falls back to a 6-decimal float."""
    out = str(_comparison(eps=0.12345))
    assert "0.12" in out  # 6-decimal float form


# ---- Total no longer duplicated (post-M5 review fix) ---------------------

def test_render_does_not_duplicate_total():
    """'14 of 99' is enough; the Snapshot line no longer repeats '(99 entries)'."""
    out = str(_comparison())
    # The total appears exactly once after "of" — and not in "(N entries)" form
    assert "99 entries" not in out
    assert "of 99" in out


def test_render_snapshot_line_is_just_a_date():
    """Snapshot: line carries the date only, no entry count."""
    out = str(_comparison(snapshot_date="2026-05-27"))
    snapshot_lines = [ln for ln in out.splitlines() if ln.startswith("Snapshot:")]
    assert len(snapshot_lines) == 1
    assert snapshot_lines[0].strip() == "Snapshot:     2026-05-27"


# ===========================================================================
# Regression: defenses added after M5 review
# ===========================================================================

# ---- schema_version: bool / zero / negative ------------------------------

def test_schema_version_bool_true_rejected():
    """bool is an int subclass. schema_version=True passing a naive isinstance(int)
    check would silently read as version 1. Reject explicitly."""
    snap = _snapshot()
    snap["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        RobustBenchClient.from_dict(snap, label="bool.json")


def test_schema_version_zero_rejected():
    snap = _snapshot(schema_version=0)
    with pytest.raises(ValueError, match="schema_version"):
        RobustBenchClient.from_dict(snap, label="zero.json")


def test_schema_version_negative_rejected():
    snap = _snapshot(schema_version=-1)
    with pytest.raises(ValueError, match="schema_version"):
        RobustBenchClient.from_dict(snap, label="neg.json")


# ---- Case insensitivity: same input produces consistent result ----------

def test_client_accepts_lowercase_threat():
    """RobustBenchClient('cifar10', 'linf') must load the same snapshot as 'Linf'."""
    c_lower = RobustBenchClient("cifar10", "linf")
    c_canonical = RobustBenchClient("cifar10", "Linf")
    assert len(c_lower) == len(c_canonical)
    assert c_lower.threat == c_canonical.threat == "Linf"


def test_client_accepts_uppercase_threat():
    c = RobustBenchClient("CIFAR10", "LINF")
    assert c.threat == "Linf"
    assert c.dataset == "cifar10"


def test_client_rejects_unknown_threat_with_helpful_error():
    with pytest.raises(ValueError, match="Unknown threat"):
        RobustBenchClient("cifar10", "Linfinity")


def test_validate_protocol_accepts_lowercase_threat():
    """The protocol gate must agree with RobustBenchClient on case handling."""
    validate_protocol(_result(), "cifar10", "linf")
    validate_protocol(_result(), "CIFAR10", "LINF")


def test_validate_protocol_and_client_agree_on_unknown_threat():
    """Both APIs reject the same unknown threat the same way."""
    with pytest.raises(ValueError):
        validate_protocol(_result(), "cifar10", "L99")
    with pytest.raises(ValueError):
        RobustBenchClient("cifar10", "L99")


# ---- from_dict must not mutate caller's data ----------------------------

def test_from_dict_does_not_mutate_caller_entries():
    """The re-rank step happens on a copy. Caller's dicts must be untouched.

    This matters because tests construct synthetic data inline:
        data = {"entries": [{"rank": 99, ...}]}
        client = RobustBenchClient.from_dict(data)
        # data["entries"][0]["rank"] must still be 99
    """
    original_entry = {
        "rank": 99,  # deliberately wrong; client should re-rank to 1 internally
        "name": "Z",
        "paper": "",
        "venue": "",
        "architecture": "",
        "clean_acc": 0.8,
        "robust_acc": 0.7,
    }
    snap = _snapshot(entries=[original_entry])
    _ = RobustBenchClient.from_dict(snap, label="immutability.json")
    assert original_entry["rank"] == 99  # unchanged


def test_from_dict_does_not_mutate_caller_entries_list_order():
    """The sort step also operates on a copy. Caller's list must be untouched."""
    e1 = {"rank": 1, "name": "A", "paper": "", "venue": "", "architecture": "",
          "clean_acc": 0.9, "robust_acc": 0.50}  # lower
    e2 = {"rank": 2, "name": "B", "paper": "", "venue": "", "architecture": "",
          "clean_acc": 0.9, "robust_acc": 0.75}  # higher
    original_entries = [e1, e2]
    snap = _snapshot(entries=original_entries)
    _ = RobustBenchClient.from_dict(snap, label="immutability.json")
    # Original list order preserved (e1 first, e2 second) — even though the
    # client internally sorts B above A.
    assert original_entries[0] is e1
    assert original_entries[1] is e2


# ---- eps validation -----------------------------------------------------

def test_eps_missing_rejected():
    snap = _snapshot()
    del snap["eps"]
    with pytest.raises(ValueError, match="eps"):
        RobustBenchClient.from_dict(snap, label="noeps.json")


def test_eps_non_numeric_rejected():
    snap = _snapshot()
    snap["eps"] = "8/255"  # string, not parsed
    with pytest.raises(ValueError, match="eps"):
        RobustBenchClient.from_dict(snap, label="strings.json")


def test_eps_bool_rejected():
    """bool sneaks past naive isinstance(int, float) checks."""
    snap = _snapshot()
    snap["eps"] = True
    with pytest.raises(ValueError, match="eps"):
        RobustBenchClient.from_dict(snap, label="bool.json")


# ---- ProtocolError message hints at serialization round-trips ------------

def test_protocol_error_mentions_serialization_in_fix():
    """The 'Fix:' footer covers the contrived-but-real case where a user did
    call robustbench_eval but their metadata got stripped by a buggy pickle
    or JSON round-trip. Don't lie to them by saying 'use robustbench_eval()'
    when they already did."""
    r = _result(metadata=None)
    with pytest.raises(ProtocolError) as exc:
        validate_protocol(r, "cifar10", "Linf")
    assert "serialization" in str(exc.value).lower() or "pickle" in str(exc.value).lower()


# ===========================================================================
# get_protocol_spec (M6 prerequisite, in leaderboard.py)
# ===========================================================================

def test_get_protocol_spec_cifar10():
    from visprobe.leaderboard import get_protocol_spec
    spec = get_protocol_spec("cifar10", "Linf")
    assert spec["n_samples"] == 10000
    assert spec["attack"] == "autoattack-standard"
    assert abs(spec["eps"] - 8 / 255) < 1e-9


def test_get_protocol_spec_imagenet():
    from visprobe.leaderboard import get_protocol_spec
    spec = get_protocol_spec("imagenet", "Linf")
    assert spec["n_samples"] == 5000
    assert abs(spec["eps"] - 4 / 255) < 1e-9


def test_get_protocol_spec_case_insensitive():
    from visprobe.leaderboard import get_protocol_spec
    assert get_protocol_spec("cifar10", "Linf") == get_protocol_spec("CIFAR10", "linf")


def test_get_protocol_spec_unknown_raises():
    from visprobe.leaderboard import get_protocol_spec
    with pytest.raises(ValueError):
        get_protocol_spec("cifar10", "L99")


def test_get_protocol_spec_returns_fresh_dict():
    """Mutating the returned dict must not affect future calls."""
    from visprobe.leaderboard import get_protocol_spec
    spec = get_protocol_spec("cifar10", "Linf")
    spec["n_samples"] = 999
    assert get_protocol_spec("cifar10", "Linf")["n_samples"] == 10000


# ===========================================================================
# robustbench_eval (M6)
# ===========================================================================

def test_robustbench_eval_confirm_false_raises_with_cost_message():
    """Without confirm=True the function must NOT do any expensive work — it
    raises RuntimeError carrying the cost estimate, so users see what they're
    about to commit to."""
    from visprobe.experiment import robustbench_eval
    import torch.nn as nn

    model = nn.Linear(10, 10)  # any nn.Module — won't be invoked
    with pytest.raises(RuntimeError) as exc:
        robustbench_eval(model, "cifar10", "Linf", confirm=False)
    msg = str(exc.value)
    assert "10000" in msg, "cost message must show sample count from _PROTOCOL"
    assert "cifar10" in msg
    assert "Linf" in msg
    assert "confirm=True" in msg


def test_robustbench_eval_cost_message_names_full_autoattack():
    """Spell out what 'full AutoAttack' means so users see why the cost is high
    and aren't tempted to substitute APGD-CE 'for speed'."""
    from visprobe.experiment import robustbench_eval
    import torch.nn as nn

    model = nn.Linear(10, 10)
    with pytest.raises(RuntimeError) as exc:
        robustbench_eval(model, "cifar10", "Linf", confirm=False)
    msg = str(exc.value)
    assert "AutoAttack" in msg


def test_robustbench_eval_imagenet_shows_5000_samples():
    """Different (dataset, threat) → different sample count in the cost message."""
    from visprobe.experiment import robustbench_eval
    import torch.nn as nn

    model = nn.Linear(10, 10)
    with pytest.raises(RuntimeError) as exc:
        robustbench_eval(model, "imagenet", "Linf", confirm=False)
    assert "5000" in str(exc.value)


def test_robustbench_eval_case_insensitive_inputs():
    """Consistent with RobustBenchClient and validate_protocol — any case works."""
    from visprobe.experiment import robustbench_eval
    import torch.nn as nn

    model = nn.Linear(10, 10)
    with pytest.raises(RuntimeError) as exc:
        robustbench_eval(model, "CIFAR10", "linf", confirm=False)
    # canonicalized to 'cifar10'/'Linf' before formatting the cost
    msg = str(exc.value)
    assert "cifar10" in msg
    assert "Linf" in msg


def test_robustbench_eval_unknown_dataset_raises_value_error():
    """Unknown (dataset, threat) is a programmer error — raise ValueError
    BEFORE the cost gate so the user fixes the typo, not the cost confirmation."""
    from visprobe.experiment import robustbench_eval
    import torch.nn as nn

    model = nn.Linear(10, 10)
    with pytest.raises(ValueError):
        robustbench_eval(model, "cifar100", "Linf", confirm=False)


def test_robustbench_eval_missing_robustbench_package_raises_with_install_hint(monkeypatch):
    """If `robustbench` is not installed, surface a helpful ImportError
    that names the install command. confirm=True is required so we reach the
    lazy-import path."""
    import sys
    import torch.nn as nn

    # Force any import of robustbench / robustbench.data to fail with ImportError.
    monkeypatch.setitem(sys.modules, "robustbench", None)
    monkeypatch.setitem(sys.modules, "robustbench.data", None)

    from visprobe.experiment import robustbench_eval
    model = nn.Linear(10, 10)
    with pytest.raises(ImportError) as exc:
        robustbench_eval(model, "cifar10", "Linf", confirm=True)
    assert "pip install visprobe[robustbench]" in str(exc.value)


# ---- Cost-message formatting helper --------------------------------------

def test_format_robustbench_cost_includes_samples_and_dataset():
    from visprobe.experiment import _format_robustbench_cost
    msg = _format_robustbench_cost("cifar10", "Linf", 10000)
    assert "10000" in msg
    assert "cifar10" in msg
    assert "AutoAttack" in msg
    assert "A100" in msg  # mentions the assumed hardware


def test_format_robustbench_cost_distinguishes_imagenet():
    """ImageNet cost range differs from CIFAR-10 — render it distinctly."""
    from visprobe.experiment import _format_robustbench_cost
    cifar = _format_robustbench_cost("cifar10", "Linf", 10000)
    imagenet = _format_robustbench_cost("imagenet", "Linf", 5000)
    assert cifar != imagenet  # messages should not be byte-identical


# ---- Metadata-tagging helper ---------------------------------------------

def test_tag_robustbench_metadata_adds_required_keys():
    """The keys validate_protocol expects to find."""
    from visprobe.experiment import _tag_robustbench_metadata
    r = _result(metadata={"attack": "autoattack-standard"})
    _tag_robustbench_metadata(r, "cifar10", "Linf")
    assert r.metadata["protocol"] == "robustbench"
    assert r.metadata["dataset"] == "cifar10"
    assert r.metadata["threat"] == "Linf"


def test_tag_robustbench_metadata_preserves_existing_keys():
    from visprobe.experiment import _tag_robustbench_metadata
    r = _result(metadata={"attack": "autoattack-standard", "custom": "value"})
    _tag_robustbench_metadata(r, "cifar10", "Linf")
    assert r.metadata["attack"] == "autoattack-standard"
    assert r.metadata["custom"] == "value"


def test_tag_robustbench_metadata_handles_none_metadata():
    """EvaluationResult.metadata defaults to None — must not crash."""
    from visprobe.experiment import _tag_robustbench_metadata
    r = _result(metadata=None)
    _tag_robustbench_metadata(r, "cifar10", "Linf")
    assert r.metadata["protocol"] == "robustbench"


def test_tag_robustbench_metadata_satisfies_validate_protocol():
    """The whole point of this helper: a result tagged by it must pass the gate
    (assuming the other protocol fields — n_samples, eps, attack, severity —
    are already correct, which they will be when robustbench_eval pins them)."""
    from visprobe.experiment import _tag_robustbench_metadata
    # Start with a result that satisfies everything EXCEPT the protocol tag.
    r = _result(metadata={"attack": "autoattack-standard"})  # missing 'protocol'
    _tag_robustbench_metadata(r, "cifar10", "Linf")
    # Should now satisfy the gate.
    validate_protocol(r, "cifar10", "Linf")  # raises ProtocolError if broken


# ===========================================================================
# compare_to_leaderboard (M7)
# ===========================================================================

from visprobe.results import CompositionalResults


def _results_with_protocol_compliant_cell(
    model_name: str = "my_model",
    accuracy: float = 0.6789,
) -> CompositionalResults:
    """A CompositionalResults containing one cell that passes validate_protocol
    for cifar10/Linf. This is what robustbench_eval (M6) would produce."""
    results = CompositionalResults()
    cell = _result(
        accuracy=accuracy,
        model_name=model_name,
        scenario="none",
        severity=0.0,
        eps=8 / 255,
        n_samples=10000,
        metadata={
            "protocol": "robustbench",
            "attack": "autoattack-standard",
            "dataset": "cifar10",
            "threat": "Linf",
        },
    )
    results.add_result(model_name, "none", 0.0, cell)
    return results


def test_compare_to_leaderboard_happy_path():
    """A protocol-tagged cell → LeaderboardComparison with correct rank.

    accuracy=0.6789 against the real shipped CIFAR-10 Linf leaderboard should
    sit somewhere in the middle (between rank 1 at ~0.71 and rank 99 at ~0.30)."""
    results = _results_with_protocol_compliant_cell(accuracy=0.6789)
    comparison = results.compare_to_leaderboard("my_model", "cifar10", "Linf")
    assert comparison.model_name == "my_model"
    assert abs(comparison.robust_acc - 0.6789) < 1e-9
    assert comparison.dataset == "cifar10"
    assert comparison.threat == "Linf"
    assert 1 <= comparison.rank <= comparison.total
    assert comparison.total > 0


def test_compare_to_leaderboard_returns_leaderboard_comparison_type():
    from visprobe.leaderboard import LeaderboardComparison
    results = _results_with_protocol_compliant_cell()
    comparison = results.compare_to_leaderboard("my_model", "cifar10", "Linf")
    assert isinstance(comparison, LeaderboardComparison)


def test_compare_to_leaderboard_carries_protocol_fields_into_comparison():
    """The Protocol: render line needs attack and eps; those come from the cell."""
    results = _results_with_protocol_compliant_cell()
    comparison = results.compare_to_leaderboard("my_model", "cifar10", "Linf")
    assert comparison.attack == "autoattack-standard"
    assert abs(comparison.eps - 8 / 255) < 1e-9


def test_compare_to_leaderboard_snapshot_date_comes_from_shipped_data():
    """LeaderboardComparison.snapshot_date is the shipped snapshot's date,
    NOT the date the EvaluationResult was produced."""
    results = _results_with_protocol_compliant_cell()
    comparison = results.compare_to_leaderboard("my_model", "cifar10", "Linf")
    # Snapshot date is whatever's in src/visprobe/data/robustbench_cifar10_linf.json.
    # We don't pin a specific date — that'd break weekly auto-refresh. Just
    # assert it's non-empty (M4 guarantees that for the shipped snapshots).
    assert comparison.snapshot_date != ""


def test_compare_to_leaderboard_returns_correct_rank_for_top_score():
    """An accuracy above every leaderboard entry must rank 1."""
    # Construct a cell with an unrealistically high robust_acc (1.0) that
    # outranks every real entry.
    results = _results_with_protocol_compliant_cell(accuracy=1.0)
    comparison = results.compare_to_leaderboard("my_model", "cifar10", "Linf")
    assert comparison.rank == 1


def test_compare_to_leaderboard_returns_correct_rank_for_bottom_score():
    """robust_acc=0.0 ties with the undefended 'Standard' baseline that
    RobustBench ships at the bottom of the leaderboard. Competition ranking
    → tied with last → rank == total."""
    results = _results_with_protocol_compliant_cell(accuracy=0.0)
    comparison = results.compare_to_leaderboard("my_model", "cifar10", "Linf")
    assert comparison.rank == comparison.total


def test_compare_to_leaderboard_neighbors_populated_for_middle_rank():
    """A middle-rank score should have at least one above and one below."""
    results = _results_with_protocol_compliant_cell(accuracy=0.55)
    comparison = results.compare_to_leaderboard("my_model", "cifar10", "Linf")
    # At a "middle" robust_acc there should be entries on both sides.
    # The shipped CIFAR-10 leaderboard has 99 entries spanning ~0.30-0.71;
    # 0.55 should be well-surrounded.
    assert len(comparison.neighbors_above) > 0
    assert len(comparison.neighbors_below) > 0


# ---- Failure modes -------------------------------------------------------

def test_compare_to_leaderboard_missing_cell_raises_with_actionable_message():
    """No cell at (model_name, 'none', 0.0) → ValueError naming the helper to call."""
    empty = CompositionalResults()
    with pytest.raises(ValueError) as exc:
        empty.compare_to_leaderboard("never_evaluated", "cifar10", "Linf")
    msg = str(exc.value)
    assert "never_evaluated" in msg
    assert "robustbench_eval" in msg


def test_compare_to_leaderboard_wrong_model_name_raises():
    """Cell exists but under a different model_name → ValueError."""
    results = _results_with_protocol_compliant_cell(model_name="actual_model")
    with pytest.raises(ValueError) as exc:
        results.compare_to_leaderboard("typo_model", "cifar10", "Linf")
    assert "typo_model" in str(exc.value)


def test_compare_to_leaderboard_protocol_violation_raises_protocol_error():
    """Cell exists but its metadata doesn't claim 'protocol=robustbench' →
    ProtocolError, not silent ranking."""
    results = CompositionalResults()
    cell = _result(
        accuracy=0.5,
        scenario="none",
        severity=0.0,
        eps=8 / 255,
        n_samples=10000,
        # No 'protocol' tag — simulates a regular CompositionalExperiment result
        # the user wrongly tries to rank against the leaderboard.
        metadata={"attack": "autoattack-standard"},
    )
    results.add_result("ad_hoc_model", "none", 0.0, cell)
    with pytest.raises(ProtocolError) as exc:
        results.compare_to_leaderboard("ad_hoc_model", "cifar10", "Linf")
    assert "use robustbench_eval()" in str(exc.value)


def test_compare_to_leaderboard_wrong_eps_raises_protocol_error():
    """All other protocol fields right, but eps doesn't match → ProtocolError."""
    results = CompositionalResults()
    cell = _result(
        accuracy=0.5,
        scenario="none",
        severity=0.0,
        eps=4 / 255,  # ImageNet eps used on CIFAR-10 protocol
        n_samples=10000,
        metadata={
            "protocol": "robustbench",
            "attack": "autoattack-standard",
        },
    )
    results.add_result("m", "none", 0.0, cell)
    with pytest.raises(ProtocolError) as exc:
        results.compare_to_leaderboard("m", "cifar10", "Linf")
    assert "eps=" in str(exc.value)


def test_compare_to_leaderboard_case_insensitive_inputs():
    """Consistent with the rest of the leaderboard surface."""
    results = _results_with_protocol_compliant_cell()
    comparison = results.compare_to_leaderboard("my_model", "CIFAR10", "linf")
    assert comparison.dataset == "cifar10"
    assert comparison.threat == "Linf"


def test_compare_to_leaderboard_render_is_paper_pasteable():
    """End-to-end: result tagged by robustbench_eval → compare → str() →
    output that contains everything a paper paragraph needs."""
    results = _results_with_protocol_compliant_cell(accuracy=0.6789)
    out = str(results.compare_to_leaderboard("my_model", "cifar10", "Linf"))
    assert "RobustBench cifar10/Linf" in out
    assert "my_model" in out
    assert "0.6789" in out
    assert "Protocol:" in out
    assert "autoattack-standard" in out
    assert "8/255" in out
    assert "Snapshot:" in out
