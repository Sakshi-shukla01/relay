"""Learn how the real Groq free tier behaves. Modest on purpose (~150
requests, ~45K tokens, mostly on the small model): the daily budget is shared
with every other experiment.

  export GROQ_API_KEY=...        # never committed
  python scripts/probe_real.py              # all phases (~6 min)
  python scripts/probe_real.py --phase latency

Writes raw/probe_real.jsonl (one line per request, with headers) and
results/limits_learned.json.
"""
from __future__ import annotations

import argparse
import os
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.config import DRAFT_MODEL, RAW, RESULTS, ROUTER_MODEL  # noqa: E402
from relay.llm import LLMClient  # noqa: E402

OUT = RAW / "probe_real.jsonl"
FILLER = ("The Transformer follows this overall architecture using stacked self-attention and point-wise, "
          "fully connected layers for both the encoder and decoder. ") * 40   # ~1,000 tokens


def log(phase, res, **kw):
    rec = dict(phase=phase, wall=time.time(), model=res.model, status=res.status, http=res.http_status,
               ttft_ms=res.ttft_ms, total_ms=round(res.total_ms, 1), tokens_in=res.tokens_in,
               tokens_out=res.tokens_out, retry_after_s=res.retry_after_s, error=res.error[:800],
               headers={k: v for k, v in res.headers.items() if "ratelimit" in k or k in ("retry-after", "x-groq-region", "date")})
    rec.update(kw)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def msgs(big: bool):
    content = (FILLER if big else "") + "In one sentence: why is self-attention parallelisable?"
    return [{"role": "user", "content": content}]


async def latency(c: LLMClient):
    """Sequential, paced under every limit: the service's own latency."""
    for model, gap in ((ROUTER_MODEL, 4.0), (DRAFT_MODEL, 6.5)):
        for i in range(14):
            big = i % 2 == 1
            res = await c.chat(model, msgs(big), max_tokens=500, ttft_timeout=30, total_timeout=60)
            # visible_chars lets calibrate.py estimate hidden reasoning tokens
            r = log("latency", res, big_prompt=big, i=i, visible_chars=len(res.text),
                    first_reasoning_ms=res.first_reasoning_ms)
            print(f"latency {model} big={big} {r['status']} ttft={r['ttft_ms']} total={r['total_ms']} out={r['tokens_out']}")
            if res.status == "error":
                print(f"  HTTP {res.http_status}: {res.error[:300]}")
                if i >= 1 and all(x.status == "error" for x in [res]):
                    errs = getattr(latency, "_errs", 0) + 1
                    latency._errs = errs
                    if errs >= 3:
                        raise SystemExit("Stopping: Groq keeps returning errors. Run scripts/check_groq.py and fix that first.")
            await asyncio.sleep(gap * float(os.environ.get("PROBE_GAP_SCALE", 1)))


async def wait_clear():
    print("waiting 65 s for minute windows to clear ...")
    await asyncio.sleep(float(os.environ.get("PROBE_CLEAR_S", 65)))


async def recover(c, phase, model, mk, max_s=90):
    """Poll once a second until a request succeeds again."""
    t0 = time.time()
    while time.time() - t0 < max_s:
        res = await c.chat(model, mk(), max_tokens=8, ttft_timeout=20)
        log(phase, res, poll=True, since_first_429_s=time.time() - t0)
        if res.status == "ok":
            return time.time() - t0
        await asyncio.sleep(1.0)
    return None


async def rpm_and_inflight(c: LLMClient):
    """35 tiny requests at the same instant: how many get in (RPM), does
    concurrency itself get refused, and what does a 429 look like."""
    await wait_clear()
    tiny = lambda: [{"role": "user", "content": "hi"}]
    t0 = time.time()
    results = await asyncio.gather(*[c.chat(ROUTER_MODEL, tiny(), max_tokens=8, ttft_timeout=30) for _ in range(35)])
    for i, r in enumerate(results):
        log("rpm_burst", r, i=i, burst_t0=t0)
    ok = sum(r.status == "ok" for r in results)
    print(f"rpm burst: {ok} ok / {35 - ok} refused; statuses={[r.http_status for r in results]}")
    if ok < 35:
        rec_s = await recover(c, "rpm_recover", ROUTER_MODEL, tiny)
        print(f"recovered after {rec_s} s")


async def tpm(c: LLMClient):
    """Back-to-back ~1K-token prompts until the tokens-per-minute limit bites."""
    await wait_clear()
    big = lambda: msgs(True)
    for i in range(10):
        res = await c.chat(ROUTER_MODEL, big(), max_tokens=8, ttft_timeout=30)
        log("tpm_seq", res, i=i)
        print(f"tpm seq {i}: {res.status} in={res.tokens_in} remaining_tokens={res.headers.get('x-ratelimit-remaining-tokens')}")
        if res.status == "rate_limited":
            rec_s = await recover(c, "tpm_recover", ROUTER_MODEL, big)
            print(f"recovered after {rec_s} s")
            break


def summarize():
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    learned = {}
    for m in (ROUTER_MODEL, DRAFT_MODEL):
        hs = [r["headers"] for r in rows if r["model"] == m and r["headers"].get("x-ratelimit-limit-requests")]
        if hs:
            learned[m] = dict(rpd=int(hs[-1]["x-ratelimit-limit-requests"]), tpm=int(hs[-1]["x-ratelimit-limit-tokens"]))
    burst = [r for r in rows if r["phase"] == "rpm_burst"]
    if burst:
        ok = sum(r["status"] == "ok" for r in burst)
        learned.setdefault(ROUTER_MODEL, {})["rpm_observed_in_burst"] = ok
    (RESULTS / "limits_learned.json").write_text(json.dumps({m: {k: v for k, v in d.items() if k in ("rpd", "tpm")}
                                                             for m, d in learned.items()}, indent=1), encoding="utf-8")
    print(json.dumps(learned, indent=1))
    for m in (ROUTER_MODEL, DRAFT_MODEL):
        lat = [r for r in rows if r["phase"] == "latency" and r["model"] == m and r["status"] == "ok"]
        if lat:
            t = sorted(r["ttft_ms"] for r in lat)
            print(m, "ttft median", statistics.median(t), "max", t[-1], "n", len(t))
    for r in rows:
        if r["status"] == "rate_limited":
            print("first 429 body:", r["error"][:400])
            print("headers:", r["headers"])
            break


async def main(phases):
    c = LLMClient("real")
    try:
        if "latency" in phases:
            await latency(c)
        if "rpm" in phases:
            await rpm_and_inflight(c)
        if "tpm" in phases:
            await tpm(c)
    finally:
        await c.close()
    summarize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="latency,rpm,tpm")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if a.summarize_only:
        summarize()
    else:
        asyncio.run(main(a.phase.split(",")))
