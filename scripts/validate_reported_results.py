from __future__ import annotations

import csv
import json
from pathlib import Path


root = Path("results/reported")
required = {
    "dataset_statistics.csv": 6,
    "sdrsb_main_results.csv": 10,
    "ablation_pems08.csv": 5,
    "sensitivity_pems08.csv": 10,
    "efficiency_pems04.csv": 6,
    "traffic_condition_pemsbay.csv": 1,
    "interpretability_summary.csv": 4,
    "training_dynamics_summary.csv": 4,
}
for filename, expected_rows in required.items():
    path = root / filename
    assert path.exists(), f"missing {path}"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == expected_rows, f"{filename}: expected {expected_rows}, got {len(rows)}"
    assert all("source" in row or "source_note" in row for row in rows), f"{filename}: missing source field"
    print(f"valid: {filename} ({len(rows)} rows)")

main = list(csv.DictReader((root / "sdrsb_main_results.csv").open(encoding="utf-8")))
lookup = {(row["dataset"], row["horizon"]): row for row in main}
assert lookup[("PEMS03", "one_step")]["mae"] == "11.58"
assert lookup[("PEMS08", "one_step")]["rmse"] == "19.40"
assert lookup[("PEMS-BAY", "60_min")]["mae"] == "1.50"

paper_index = json.loads((root / "paper_metrics.json").read_text(encoding="utf-8"))
assert paper_index["datasets"] == 6
assert paper_index["architecture"]["global_chebyshev_order"] == 3
print("all reported-result checks passed")
