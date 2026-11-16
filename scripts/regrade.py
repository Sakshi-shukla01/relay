"""Re-grade graded runs from the answers saved in traces.jsonl, with the
current grader (relay/grading.py) and data/grade_overrides.json, and print
every grade that changed. Keeps the change auditable.

  python scripts/regrade.py eval_main
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from relay.config import DATA, RESULTS  # noqa: E402
from relay.corpus import PAPERS  # noqa: E402
from relay.grading import grade, load_overrides  # noqa: E402

scen = set(sys.argv[1:]) or {"eval_main"}
qs = {q["question"]: q for q in json.loads((DATA / "questions.json").read_text(encoding="utf-8"))}
title2key = {p["title"]: p["key"] for p in PAPERS}
tr = {}
for line in (RESULTS / "traces.jsonl").read_text(encoding="utf-8").splitlines():
    t = json.loads(line)
    if t["scenario"] in scen:
        tr[t["request_id"]] = t
over = load_overrides()
p = RESULTS / "results.csv"
rows = list(csv.DictReader(p.open(encoding="utf-8")))
changed = []
for r in rows:
    t = tr.get(r["request_id"])
    if not t or r["status"] not in ("ok", "rate_limited"):
        continue
    q = qs[t["question"]]
    ans = t["answer"]
    refused = ans.strip().startswith("The papers in this library do not say")
    src = ans.split("Sources:")[1] if "Sources:" in ans else ""
    cited = sorted({k for title, k in title2key.items() if title in src})
    if t["level"] == "L3":
        cited = sorted({k for title, k in title2key.items() if title in ans})
    g = over.get(f"{r['scenario']}:{q['id']}") or grade(q, ans, refused, t["level"], cited)
    if g != r["answer_grade"]:
        changed.append((r["scenario"], q["id"], r["answer_grade"], g))
        r["answer_grade"] = g
with p.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
for c in changed:
    print(f"{c[0]} {c[1]}: {c[2]} -> {c[3]}")
print(f"{len(changed)} grade(s) changed")