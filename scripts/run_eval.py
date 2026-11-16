"""Answer quality.

  python scripts/run_eval.py --main            # real: full pipeline (120b drafter), graded
  python scripts/run_eval.py --sweep --settings small,medium,medium_rerank   # real, 20b drafter, graded
  python scripts/run_eval.py --main --standin  # wiring test only (stand-in answers are fake)

Eval runs use generous deadlines so the pipeline waits for budget instead of
degrading: here we measure what the answer is *worth*, not how fast it is
under load (that is run_load.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.config import DATA, FALLBACK_DRAFT_MODEL, SLO, PipelineConfig  # noqa: E402
from relay.runner import StandInProcess, make_relay, run_schedule  # noqa: E402

QUESTIONS = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
EVAL_SLO = SLO(ttft_target_ms=60000, ttft_deadline_ms=90000, total_deadline_ms=120000, max_model_calls=6)
SWEEP = ["small", "medium", "medium_rerank", "large"]


async def run(scenario, cfg, source, spacing_s):
    relay = make_relay(source, cfg)
    # one question at a time: quality must not depend on how fast we send.
    # (first version sent one every 12 s; at 8K tokens/min that queued, and
    # 7 of 26 answers timed out into excerpts before the model saw them)
    arrivals = [0.0] * len(QUESTIONS)
    recs = await run_schedule(relay, scenario, arrivals, QUESTIONS, patience=None, grade_answers=True,
                              sequential=True)
    await relay.client.close()
    import csv
    rows = [r for r in csv.DictReader(open("results/results.csv", encoding="utf-8")) if r["scenario"] == scenario]
    print(scenario, Counter(r["answer_grade"] for r in rows), "levels", Counter(r["level"] for r in rows))
    return recs


async def main(a):
    source = "standin" if a.standin else "real"
    jobs = []
    if a.main:
        # 8K tokens/min on the 120b model ~ 3 full answers a minute: space them
        jobs.append(("eval_main", PipelineConfig(slo=EVAL_SLO), 20.0))
    if a.final:
        # the configuration chosen from the sweep: small pieces, 120b drafter.
        # No 20b router/verifier calls, so it fits the 120b daily budget
        # alone (the 20b budget goes to the sweep).
        jobs.append(("eval_final", PipelineConfig(slo=EVAL_SLO, retrieval="small", llm_verify=False,
                                                  force_level="L1"), 20.0))
    if a.sweep:
        for s in (a.settings.split(",") if a.settings else SWEEP):
            # small drafter, no LLM verifier: isolates retrieval, and keeps
            # the sweep inside the daily token budget (200K/model/day)
            jobs.append((f"retrieval_{s}", PipelineConfig(slo=EVAL_SLO, retrieval=s, llm_verify=False,
                                                          force_level="L1", draft_models=(FALLBACK_DRAFT_MODEL,)), 12.0))
    if a.standin:
        async with StandInProcess() as si:
            for name, cfg, sp in jobs:
                await si.admin("reset", {"_config_reload": True})
                await run(name + "_standin_wiring", cfg, source, 0.5)
    else:
        for name, cfg, sp in jobs:
            await run(name, cfg, source, sp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--final", action="store_true", help="chosen config: small retrieval + 120b")
    ap.add_argument("--settings", default="")
    ap.add_argument("--standin", action="store_true")
    asyncio.run(main(ap.parse_args()))