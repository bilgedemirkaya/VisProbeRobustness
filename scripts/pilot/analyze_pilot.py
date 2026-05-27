"""
Produce the pilot's two decision tables:
  1. pilot_grid.csv          - 5 models x 3 severities of robust accuracies
                               (severity=0.0 column = RobustBench replication check)
  2. severity_thresholds.csv - coarse severity threshold per model
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

# Published RobustBench robust accuracies (Linf, eps=8/255, CIFAR-10).
# Update from robustbench.github.io before publishing -- these shift.
PUBLISHED_ROBUST_ACC = {
    "Wang2023Better_WRN-70-16": 0.7088,
    "Cui2023Decoupled_WRN-28-10": 0.6773,
    "Rebuffi2021Fixing_70_16_cutmix_extra": 0.6658,
    "Gowal2020Uncovering_70_16_extra": 0.6543,
    "Carmon2019Unlabeled": 0.5961,
}

results = CompositionalResults.load("./results/pilot_apgd")

# ---- Grid: rows = models in rank order, columns = severities ----
grid_rows = []
for name in MODEL_NAMES:
    row = {"model": name}
    for sev in SEVERITIES:
        r = results.get_result(name, SCENARIO, sev)
        row[f"sev_{sev}"] = round(r.accuracy, 4) if r else None
    measured = row["sev_0.0"]
    published = PUBLISHED_ROBUST_ACC.get(name)
    delta = round(measured - published, 4) if measured is not None else None
    row["published"] = published
    row["delta_vs_published"] = delta
    grid_rows.append(row)

df_grid = pd.DataFrame(grid_rows)
df_grid.to_csv("pilot_grid.csv", index=False)
print("=== Pilot grid ===")
print(df_grid.to_string(index=False))

# Flag any replication mismatch > 0.5pp -- that means the eval methodology disagrees
# with the published number and the whole audit is suspect.
bad = df_grid[df_grid["delta_vs_published"].abs() > 0.005]
if len(bad):
    print("\nWARNING: replication mismatch > 0.5pp for:")
    print(bad[["model", "sev_0.0", "published", "delta_vs_published"]].to_string(index=False))
    print("Investigate before drawing any conclusions from the pilot.")
else:
    print("\nReplication check passed (all deltas < 0.5pp).")

# ---- Severity threshold per model ----
# Definition: smallest severity s > 0 where acc(env+attack, s) < acc(no env, attack).
# i.e. where adding env makes things worse than pure adversarial attack alone.
thresh_rows = []
for name in MODEL_NAMES:
    baseline = results.get_result(name, SCENARIO, 0.0).accuracy
    threshold = None
    for sev in [s for s in SEVERITIES if s > 0.0]:
        r = results.get_result(name, SCENARIO, sev)
        if r and r.accuracy < baseline:
            threshold = sev
            break
    thresh_rows.append({
        "model": name,
        "robust_acc_no_env": round(baseline, 4),
        "threshold": threshold,
    })

df_thresh = pd.DataFrame(thresh_rows)
df_thresh.to_csv("severity_thresholds.csv", index=False)
print("\n=== Severity thresholds ===")
print(df_thresh.to_string(index=False))
