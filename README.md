# The Relay: a cited-answer pipeline that degrades instead of breaking

A small, working version of a multi-step assistant that answers questions
from three arXiv papers, with every answer cited, refusing when the papers do
not say, and built around the Groq free tier's real limits. A stand-in model
server, calibrated from real measurements, covers the loads the free tier
cannot take.

```
question ─► understand (20b router, JSON) ─► retrieve (BM25 + re-rank tool)
        ─► decide (refuse / re-retrieve loop) ─► draft (120b, cited, buffered)
        ─► check (deterministic: citations, numbers) ─► verify (20b) ─► [revise loop ×1]
        ─► stream the checked answer + sources
```

## Quick start (on your own machine)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...         # your own free key. Never commit it.
python run_all.py                   # ~75 min; see run_all.py for the stages
```

No Groq key? `python run_all.py --stages papers,offline,standin,analyze` runs
everything that does not need the real service.

Single stages:

| command | what | Groq cost |
|---|---|---|
| `python -m relay.corpus` | fetch the PDFs at v7 / v1 / v2, check the version stamp, list sections | 0 |
| `python scripts/check_questions.py` | every expected answer phrase must occur in the paper text | 0 |
| `python scripts/probe_real.py` | real latency, RPM burst, TPM, 429 shape, recovery time | ~150 req |
| `python scripts/calibrate.py` | fit the stand-in to the probe, replay, compare | 0 |
| `python scripts/retrieval_offline.py` | recall@k per retrieval setting, misleading cases | 0 |
| `python scripts/run_eval.py --sweep` / `--main` | answer quality, graded | real |
| `python scripts/run_load.py --scenario all` | 8 load shapes on the stand-in (incl. two retry experiments) | 0 |
| `python scripts/run_load.py --real --scenario real_steady` | small real load | real |
| `python scripts/analyze.py` | rebuilds every table into `results/tables.md` | 0 |

## Files

- `relay/pipeline.py` the pipeline, the degradation ladder (L0 to L3), retry policy
- `relay/governor.py` client-side model of every per-model limit, synced from `x-ratelimit-*` headers
- `relay/standin.py` the stand-in: Groq-shaped 429s, lognormal latency, slow-success, stalls, 5xx, load slowdown, streaming
- `relay/retrieval.py`, `relay/corpus.py` chunking with section and page metadata, BM25, re-ranker
- `relay/verify.py` deterministic answer checks; `relay/grading.py` deterministic grader
- `results/results.csv` one row per request (required columns first, extra columns after)
- `results/traces.jsonl` every span of every request (wait / work / retry)
- `raw/probe_real.jsonl`, `raw/provider_events.jsonl` raw real-service observations, headers included

`answer_grade` is blank for stand-in rows: the stand-in's answers are synthetic,
so grading them would mean nothing. "Peak queue length" in Table 2b means the
largest number of people waiting for a first token at the same moment,
computed from `start_time` and `first_token_ms`.

`results.csv` columns after the required twelve: `question_id, level, drafter,
queue_wait_ms, model_ttft_ms, model_calls, cost_usd, patience_ms, gave_up`.

## Testing without arXiv

`RELAY_FIXTURE=1` switches to a tiny synthetic corpus (`tests/make_fixture.py`)
used only to test the code where arXiv is unreachable. Rows produced that way
are named `*_wiring` and deleted by `run_all.py`.
