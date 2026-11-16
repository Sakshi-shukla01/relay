"""Add the real probe requests to results.csv as scenario 'probe', so the
real-latency figures are computable from results.csv like everything else.

  python scripts/probe_to_csv.py
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from relay.config import PRICES, RAW, RESULTS  # noqa: E402
from relay.results import COLUMNS  # noqa: E402

p = RESULTS / "results.csv"
rows = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r["scenario"] != "probe"]
probe = [json.loads(l) for l in (RAW / "probe_real.jsonl").read_text(encoding="utf-8").splitlines()]
new = []
for i, r in enumerate(probe):
    pi, po = PRICES.get(r["model"], (0, 0))
    new.append(dict(
        request_id=f"probe{i:04d}", scenario="probe", source="real",
        start_time=datetime.fromtimestamp(r["wall"] - r["total_ms"] / 1000, timezone.utc).isoformat(timespec="milliseconds"),
        first_token_ms="" if r["ttft_ms"] is None else round(r["ttft_ms"], 1), total_ms=r["total_ms"],
        status={"ok": "ok", "rate_limited": "rate_limited", "timeout": "timeout"}.get(r["status"], "error"),
        retries=0, tokens_in=r["tokens_in"], tokens_out=r["tokens_out"], retrieval_setting="none (probe)",
        answer_grade="", question_id=r["phase"], level="", drafter=r["model"], queue_wait_ms=0,
        model_ttft_ms="" if r["ttft_ms"] is None else round(r["ttft_ms"], 1), model_calls=1,
        cost_usd=f"{r['tokens_in'] * pi / 1e6 + r['tokens_out'] * po / 1e6:.6f}", patience_ms="", gave_up=0))
with p.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(new + rows)
print(f"added {len(new)} probe rows; results.csv now has {len(new) + len(rows)} rows")