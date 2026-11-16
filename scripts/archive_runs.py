"""Move rows of finished-but-superseded runs out of results.csv into
results/archive_<name>.csv (kept as evidence, excluded from the tables).

  python scripts/archive_runs.py first_sweep retrieval_small retrieval_medium retrieval_medium_rerank
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from relay.config import RESULTS  # noqa: E402

name, scen = sys.argv[1], set(sys.argv[2:])
p = RESULTS / "results.csv"
rows = list(csv.DictReader(p.open(encoding="utf-8")))
keep = [r for r in rows if r["scenario"] not in scen]
gone = [r for r in rows if r["scenario"] in scen]
for path, rs in ((p, keep), (RESULTS / f"archive_{name}.csv", gone)):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rs)
print(f"archived {len(gone)} rows to archive_{name}.csv; {len(keep)} rows stay in results.csv")
