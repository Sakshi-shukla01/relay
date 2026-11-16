"""Fit the stand-in to the real probe, then replay the probe against the
stand-in and print real-vs-stand-in percentiles side by side.

  python scripts/calibrate.py            # fit + check
"""
from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.config import DOCUMENTED_LIMITS, RAW, RESULTS  # noqa: E402
from relay.llm import LLMClient  # noqa: E402
from relay.runner import StandInProcess  # noqa: E402
from relay.standin import DEFAULT_MODEL  # noqa: E402

PROBE = RAW / "probe_real.jsonl"


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def fit():
    rows = [json.loads(l) for l in PROBE.read_text(encoding="utf-8").splitlines()]
    learned = json.loads((RESULTS / "limits_learned.json").read_text(encoding="utf-8")) if (RESULTS / "limits_learned.json").exists() else {}
    cfg = {"_meta": {"calibrated": True, "source": str(PROBE.name), "assumed": []}}
    for m, doc in DOCUMENTED_LIMITS.items():
        lat = [r for r in rows if r["phase"] == "latency" and r["model"] == m and r["status"] == "ok" and r["ttft_ms"]]
        c = dict(DEFAULT_MODEL)
        c.update(doc)
        c.update(learned.get(m, {}))
        if len(lat) >= 6:
            small = [r for r in lat if not r.get("big_prompt")]
            large = [r for r in lat if r.get("big_prompt")]
            ts = [r["ttft_ms"] for r in small]
            med = statistics.median(ts)
            core = [t for t in ts if t < 3 * med]
            c["ttft_mu_ms"] = round(med, 1)
            c["ttft_sigma"] = round(min(1.0, max(0.12, statistics.pstdev([math.log(t) for t in core]))), 3)
            if large and small:
                dt = statistics.median(r["ttft_ms"] for r in large) - med
                dtok = statistics.mean(r["tokens_in"] for r in large) - statistics.mean(r["tokens_in"] for r in small)
                c["prefill_ms_per_1k"] = round(max(0.0, dt / max(1, dtok) * 1000), 2)
            # gpt-oss: usage.completion_tokens includes hidden reasoning, generated
            # *before* the first visible token. Visible tokens ~ chars/4.
            vis = lambda r: r.get("visible_chars", r["tokens_out"] * 4) / 4
            rates = [vis(r) / ((r["total_ms"] - r["ttft_ms"]) / 1000)
                     for r in lat if vis(r) >= 15 and r["total_ms"] > r["ttft_ms"] + 5]
            if rates:
                c["decode_tok_s"] = round(statistics.median(rates), 1)
            hidden = [max(0.0, r["tokens_out"] - vis(r)) for r in lat if "visible_chars" in r]
            if hidden:
                c["reasoning_tokens"] = round(statistics.median(hidden), 1)
                # the lognormal TTFT is then what is left after reasoning time
                c["ttft_mu_ms"] = round(max(50.0, c["ttft_mu_ms"] - c["reasoning_tokens"] / c["decode_tok_s"] * 1000), 1)
            all_t = [r["ttft_ms"] for r in lat]
            slow = [t / med for t in all_t if t > 3 * med]
            c["p_slow"] = round(max(len(slow) / len(all_t), 0.01), 3)
            if slow:
                c["slow_mult"] = (round(min(slow), 2), round(max(slow), 2))
            else:
                cfg["_meta"]["assumed"].append(f"{m}: no slow calls seen in {len(all_t)}; p_slow=1% assumed")
        else:
            cfg["_meta"]["assumed"].append(f"{m}: <6 latency samples, defaults used")
        errs = [r for r in rows if r["model"] == m and r["status"] == "error"]
        n = len([r for r in rows if r["model"] == m])
        c["p_error"] = round(max(len(errs) / max(1, n), 0.005), 4)
        cfg["_meta"]["assumed"].append(f"{m}: capacity={c['capacity']} and load_alpha={c['load_alpha']} are assumptions "
                                       "(free tier cannot be loaded enough to measure them)")
        cfg[m] = c
    (RESULTS / "standin_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(json.dumps(cfg, indent=1))
    return rows


async def check(rows):
    """Replay the latency phase against the stand-in (limits lifted)."""
    lat = [r for r in rows if r["phase"] == "latency"]
    from scripts.probe_real import msgs  # same prompts
    out = []
    async with StandInProcess(port=8090, seed=3) as si:
        await si.admin("config", {m: {"rpm": 10**6, "tpm": 10**9, "rpd": 10**9, "tpd": 10**12} for m in DOCUMENTED_LIMITS})
        import relay.llm as L
        c = LLMClient("standin", log_events=False)
        c.base = f"{si.base}/openai/v1"
        for r in lat:
            res = await c.chat(r["model"], msgs(r.get("big_prompt", False)), max_tokens=500, ttft_timeout=60, total_timeout=90)
            out.append(dict(model=r["model"], big=r.get("big_prompt"), ttft=res.ttft_ms, total=res.total_ms))
        await c.close()
    lines = ["| model | prompt | n | real TTFT p50/p90 (ms) | stand-in TTFT p50/p90 (ms) | real total p50/p90 | stand-in total p50/p90 |",
             "|---|---|---|---|---|---|---|"]
    for m in DOCUMENTED_LIMITS:
        for big in (False, True):
            rr = [r for r in lat if r["model"] == m and bool(r.get("big_prompt")) == big and r["status"] == "ok"]
            ss = [s for s in out if s["model"] == m and bool(s["big"]) == big and s["ttft"]]
            if not rr:
                continue
            f = lambda xs: f"{pct(xs, 50):.0f} / {pct(xs, 90):.0f}" if xs else "n/a"
            lines.append(f"| {m} | {'~1K tok' if big else 'short'} | {len(rr)} | {f([r['ttft_ms'] for r in rr])} | "
                         f"{f([s['ttft'] for s in ss])} | {f([r['total_ms'] for r in rr])} | {f([s['total'] for s in ss])} |")
    txt = "\n".join(lines)
    (RESULTS / "calibration_check.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    rows = fit()
    asyncio.run(check(rows))
