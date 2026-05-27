"""
Analyze the pilot results — produce a 5x3 grid of robust accuracies and the
APGD-CE-vs-AutoAttack inflation column.

The pilot uses APGD-CE, which is strictly weaker than full AutoAttack. So:
  - Published RobustBench numbers (full AutoAttack) provide a lower bound.
  - Our APGD-CE numbers at severity=0 should be ABOVE published (positive
    inflation), typically 1-3 pp. That's not a bug — it's the expected
    consequence of running a weaker attack.
  - Negative inflation OR >5 pp inflation IS suspicious (negative: APGD-CE
    somehow beat AutoAttack; >5 pp: methodology drift). Only those warn.

The full study would re-run with attack="autoattack-standard" and a tight
0.5 pp tolerance. For the pilot the goal is the cross-model comparison, not
strict leaderboard replication.
"""

import pandas as pd

from visprobe import CompositionalResults

MODEL_NAMES = [
    "Wang2023Better_WRN-70-16",
    "Cui2023Decoupled_WRN-28-10",
    "Rebuffi2021Fixing_70_16_cutmix_extra",
    "Gowal2020Uncovering_70_16_extra",
    "Carmon2019Unlabeled",
]
SCENARIO = "noise"
SEVERITIES = [0.0, 0.5, 1.0]

# Published RobustBench robust accuracies (Linf, eps=8/255, CIFAR-10), under
# full AutoAttack-standard. Update from robustbench.github.io before publishing.
PUBLISHED_ROBUST_ACC_AA_STANDARD = {
    "Wang2023Better_WRN-70-16": 0.7088,
    "Cui2023Decoupled_WRN-28-10": 0.6773,
    "Rebuffi2021Fixing_70_16_cutmix_extra": 0.6658,
    "Gowal2020Uncovering_70_16_extra": 0.6543,
    "Carmon2019Unlabeled": 0.5961,
}

# APGD-CE inflation bounds. APGD-CE is one sub-attack out of four; numbers
# typically 1-3 pp above full AutoAttack.
#   inflation < 0       → suspicious (weaker attack found a stronger adversary)
#   inflation > UPPER   → suspicious (methodology drift, not just attack weakness)
EXPECTED_INFLATION_UPPER_PP = 5.0

results = CompositionalResults.load("./results/pilot_apgd")

# ---- Grid: rows = models in rank order, columns = severities ----
grid_rows = []
for name in MODEL_NAMES:
    row = {"model": name}
    for sev in SEVERITIES:
        r = results.get_result(name, SCENARIO, sev)
        row[f"sev_{sev}"] = round(r.accuracy, 4) if r else None
    measured = row["sev_0.0"]
    published = PUBLISHED_ROBUST_ACC_AA_STANDARD.get(name)
    inflation_pp = (
        round((measured - published) * 100, 2)
        if measured is not None and published is not None
        else None
    )
    row["published_AA_standard"] = published
    row["apgd_ce_inflation_pp"] = inflation_pp
    grid_rows.append(row)

df_grid = pd.DataFrame(grid_rows)
df_grid.to_csv("pilot_grid.csv", index=False)
print("=== Pilot grid ===")
print(df_grid.to_string(index=False))

# Only flag truly suspicious inflation — negative (impossible under correct
# methodology) or above the expected 1-3 pp band. The 1-3 pp positive band
# is expected and NOT a warning.
suspicious = df_grid[
    (df_grid["apgd_ce_inflation_pp"] < 0)
    | (df_grid["apgd_ce_inflation_pp"] > EXPECTED_INFLATION_UPPER_PP)
]
if len(suspicious):
    print(
        f"\nWARNING: APGD-CE inflation outside the expected 0-{EXPECTED_INFLATION_UPPER_PP:.0f} pp "
        "band for the following models:"
    )
    print(
        suspicious[
            ["model", "sev_0.0", "published_AA_standard", "apgd_ce_inflation_pp"]
        ].to_string(index=False)
    )
    print(
        "Negative inflation means APGD-CE found a stronger adversary than "
        "AutoAttack-standard — methodology likely drifted. >5 pp inflation "
        "suggests something else changed (sample subset, eps, normalization)."
    )
else:
    print(
        f"\nAPGD-CE inflation is within the expected 0-{EXPECTED_INFLATION_UPPER_PP:.0f} pp band "
        "for all models (a weaker attack producing slightly higher accuracies is correct behavior)."
    )

# Note on severity_thresholds: removed. With severities [0.0, 0.5, 1.0], the
# threshold "smallest severity s > 0 where acc drops" degenerates to 0.5 for
# every model (every model degrades under any non-zero noise), giving zero
# information. The full study should use 5-7 severity points and either an
# interpolated "severity-at-X-pp-drop" metric or drop the column entirely.
