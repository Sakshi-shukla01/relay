"""One command for the whole study, in an order that respects the daily budget.

  set GROQ_API_KEY=gsk_...             (Windows cmd; never put it in a file)
  python run_all.py --stages papers,probe,calibrate,offline,standin,analyze   # day 1
  python run_all.py --stages sweep,eval,analyze                              # day 2
  python run_all.py --stages real_load,analyze                               # day 3 morning

Models: openai/gpt-oss-20b (small: router, verifier, fallback drafter) and
openai/gpt-oss-120b (large: drafter). Free tier, each: 30 RPM, 1K RPD,
8K TPM, 200K TPD (documented; the probe measures them).

Stages:
  papers       fetch the 3 PDFs at exact versions, extract, check questions   0 tokens
  probe        real: latency, RPM burst, TPM, recovery                        ~150 req, ~45K tok (mostly 20b)
  calibrate    fit the stand-in to the probe, replay to compare               0 Groq tokens
  offline      retrieval recall@k for every setting                           0 tokens
  sweep        real: 3 retrieval settings x 26 questions, 20b drafter, one at a time   ~140K 20b tokens
  sweep_large  real: optional 4th setting (250-word pieces, no re-rank)        ~45K 20b tokens
  eval         real: full pipeline (120b + 20b) on 26 questions                ~80K 120b, ~50K 20b
  real_load    real: steady 3/min x 5 min, then a 12-request burst            ~80K 120b, ~40K 20b
  standin      stand-in: 8 load shapes                                         0 Groq tokens
  analyze      rebuild every table from results/results.csv

The 20b model's 200K tokens/day are the tight budget, hence the two days.
"""
import argparse
import os
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STAGES = ["papers", "probe", "calibrate", "offline", "sweep", "sweep_large", "eval", "real_load", "standin", "analyze"]


def sh(*args):
    print("\n$", " ".join(args), flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode not in (0,):
        print(f"(exit code {r.returncode})")
    return r.returncode


def drop_wiring_rows():
    p = ROOT / "results" / "results.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    keep = [r for r in rows if not r["scenario"].endswith("_wiring")]
    if len(keep) != len(rows):
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    st = ap.parse_args().stages.split(",")
    os.environ.setdefault("PYTHONUTF8", "1")   # Windows: papers contain non-cp1252 symbols
    drop_wiring_rows()
    if "papers" in st:
        sh("-m", "relay.corpus")
        sh("scripts/check_questions.py")
    if "probe" in st:
        sh("scripts/probe_real.py")
    if "calibrate" in st:
        sh("scripts/calibrate.py")
    if "offline" in st:
        sh("scripts/retrieval_offline.py")
    if "sweep" in st:
        sh("scripts/run_eval.py", "--sweep", "--settings", "small,medium_rerank,large")
    if "sweep_large" in st:
        sh("scripts/run_eval.py", "--sweep", "--settings", "medium")
    if "eval" in st:
        sh("scripts/run_eval.py", "--main")
    if "real_load" in st:
        sh("scripts/run_load.py", "--real", "--scenario", "real_steady")
        sh("scripts/run_load.py", "--real", "--scenario", "real_burst")
    if "standin" in st:
        sh("scripts/run_load.py", "--scenario", "all")
    if "analyze" in st:
        sh("scripts/analyze.py")
