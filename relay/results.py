"""results.csv (one row per request) and traces.jsonl (spans per request)."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from .config import RESULTS

CSV = RESULTS / "results.csv"
TRACES = RESULTS / "traces.jsonl"
REQUIRED = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status",
            "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
EXTRA = ["question_id", "level", "drafter", "queue_wait_ms", "model_ttft_ms", "model_calls",
         "cost_usd", "patience_ms", "gave_up"]
COLUMNS = REQUIRED + EXTRA


def row(rec, grade: str = "", patience_ms: float | None = None, gave_up: bool = False) -> dict:
    f = lambda x: "" if x is None else round(x, 1)
    return dict(
        request_id=rec.request_id, scenario=rec.scenario,
        source="real" if rec.source == "real" else "stand-in",
        start_time=datetime.fromtimestamp(rec.start_wall, timezone.utc).isoformat(timespec="milliseconds"),
        first_token_ms=f(rec.first_token_ms), total_ms=f(rec.total_ms), status=rec.status,
        retries=rec.retries, tokens_in=rec.tokens_in, tokens_out=rec.tokens_out,
        retrieval_setting=rec.retrieval_setting, answer_grade=grade,
        question_id=rec.qid, level=rec.level, drafter=rec.drafter,
        queue_wait_ms=f(rec.queue_wait_ms), model_ttft_ms=f(rec.model_ttft_ms),
        model_calls=rec.model_calls, cost_usd=f"{rec.cost_usd:.6f}",
        patience_ms=f(patience_ms), gave_up=int(gave_up))


def write(rows: list[dict], recs: list, scenario: str):
    """Replace any earlier rows of this scenario, then append."""
    old = []
    if CSV.exists():
        with CSV.open(encoding="utf-8") as fh:
            old = [r for r in csv.DictReader(fh) if r["scenario"] != scenario]
    with CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in old + rows:
            w.writerow(r)
    keep = []
    if TRACES.exists():
        keep = [l for l in TRACES.read_text(encoding="utf-8").splitlines() if json.loads(l).get("scenario") != scenario]
    with TRACES.open("w", encoding="utf-8") as fh:
        for l in keep:
            fh.write(l + "\n")
        for rec in recs:
            fh.write(json.dumps(dict(request_id=rec.request_id, scenario=rec.scenario, question=rec.question,
                                     level=rec.level, status=rec.status, notes=rec.notes, spans=rec.spans,
                                     tokens=rec.tokens, answer=rec.answer[:1500])) + "\n")
