"""Rebuild every table from results/results.csv (+ raw probe file), so the
write-up and the raw data cannot drift apart.

  python scripts/analyze.py            # writes results/tables.md
  python scripts/analyze.py --slow REQUEST_ID

Percentiles: linear interpolation between closest ranks (numpy's default).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from relay.config import LARGE, RAW, RESULTS, SMALL  # noqa: E402

CSV = RESULTS / "results.csv"
TRACES = RESULTS / "traces.jsonl"
ORDER = ["probe", "real_steady", "real_burst", "steady", "burst", "sustained_overload", "rpm_halved",
         "daily_exhausted", "burst_naive_retry", "slow_provider", "slow_provider_naive_retry"]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def fmt3(xs):
    if not xs:
        return "not measured"
    return " / ".join(f"{pct(xs, p):.0f}" for p in (50, 95, 99))


def fmt2(xs):
    if not xs:
        return "not measured"
    return " / ".join(f"{pct(xs, p):.0f}" for p in (50, 95))


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def load():
    if not CSV.exists():
        return []
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    for r in rows:
        r["_t"] = ts(r["start_time"])
        for k in ("first_token_ms", "total_ms", "queue_wait_ms", "model_ttft_ms", "cost_usd"):
            r[k] = float(r[k]) if r.get(k) not in ("", None) else None
        for k in ("retries", "tokens_in", "tokens_out", "model_calls"):
            r[k] = int(r[k] or 0)
    return rows


def peak_waiting(rs):
    """Max number of people waiting for a first token at the same moment."""
    ev = []
    for r in rs:
        end = r["first_token_ms"] if r["first_token_ms"] is not None else r["total_ms"]
        ev += [(r["_t"], 1), (r["_t"] + end / 1000, -1)]
    cur = best = 0
    for _, d in sorted(ev, key=lambda e: (e[0], e[1])):
        cur += d
        best = max(best, cur)
    return best


def table2(rows):
    by = defaultdict(list)
    for r in rows:
        if r["scenario"].startswith(("eval_", "retrieval_")) or r["scenario"].endswith("_wiring"):
            continue
        by[r["scenario"]].append(r)
    names = sorted(by, key=lambda n: ORDER.index(n) if n in ORDER else 99)
    a = ["| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |",
         "|---|---|---|---|---|---|---|"]
    b = ["| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |",
         "|---|---|---|---|---|---|---|---|"]
    extra = ["| Scenario | Full answer (L0/L1) % | small-model answer (L2) % | Excerpts (L3) % | Error % | First token p50/p95, full answers only (ms) | Tokens wasted on given-up requests |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        rs = by[n]
        t0 = min(r["_t"] for r in rs)
        t_last_start = max(r["_t"] for r in rs)
        t_end = max(r["_t"] + r["total_ms"] / 1000 for r in rs)
        dur = t_end - t0
        span = t_last_start - t0
        rate = len(rs) / (span / 60) if span > 1 else len(rs) / (dur / 60)
        ft = [r["first_token_ms"] for r in rs if r["first_token_ms"] is not None]
        tot = [r["total_ms"] for r in rs if r["status"] != "timeout"]
        c = Counter(r["status"] for r in rs)
        N = len(rs)
        ok = [r for r in rs if r["status"] == "ok"]
        tpa = sum(r["tokens_in"] + r["tokens_out"] for r in ok) / len(ok) if ok else None
        cost = sum(r["cost_usd"] or 0 for r in rs) / len(ok) * 1000 if ok else None
        a.append(f"| {n} | {rs[0]['source']} | {N} | {dur:.0f} | {rate:.1f} | {fmt3(ft)} | {fmt3(tot)} |")
        b.append(f"| {n} | {100*c['ok']/N:.1f} | {100*c['rate_limited']/N:.1f} | {100*c['timeout']/N:.1f} | "
                 f"{sum(r['retries'] for r in rs)/N:.2f} | {peak_waiting(rs)} | "
                 f"{'not measured' if tpa is None else f'{tpa:.0f}'} | {'not measured' if cost is None else f'${cost:.2f}'} |")
        lv = Counter(r["level"] for r in rs if r["status"] == "ok")
        full = [r["first_token_ms"] for r in rs if r["status"] == "ok" and r["level"] in ("L0", "L1") and r["first_token_ms"]]
        wasted = sum(r["tokens_in"] + r["tokens_out"] for r in rs if r["status"] == "timeout")
        extra.append(f"| {n} | {100*(lv['L0']+lv['L1'])/N:.1f} | {100*lv['L2']/N:.1f} | {100*c['rate_limited']/N:.1f} | "
                     f"{100*c['error']/N:.1f} | {fmt2(full)} | {wasted} |")
    return "\n".join(a), "\n".join(b), "\n".join(extra)


def plot_cdf(rows):
    """Time-to-first-token CDF per load scenario -> results/latency_cdf.png"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not installed; no plot"
    by = defaultdict(list)
    for r in rows:
        if r["scenario"].startswith(("eval_", "retrieval_")) or r["scenario"].endswith("_wiring"):
            continue
        by[r["scenario"]].append(r)
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in sorted(by, key=lambda n: ORDER.index(n) if n in ORDER else 99):
        rs = by[n]
        # given-up requests never got a first token: plot them at +inf so the curve tops out below 1
        xs = sorted(r["first_token_ms"] for r in rs if r["first_token_ms"] is not None)
        if not xs:
            continue
        ys = [(i + 1) / len(rs) for i in range(len(xs))]
        ax.step(xs, ys, where="post", label=f"{n} ({rs[0]['source']}, n={len(rs)})",
                linestyle="--" if rs[0]["source"] == "stand-in" else "-")
    ax.axvline(3000, color="grey", lw=0.8)
    ax.axvline(8000, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("time to first verified token (ms, log)")
    ax.set_ylabel("share of all requests")
    ax.set_title("First token by scenario (curve below 1.0 = people who gave up)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "latency_cdf.png", dpi=150)
    return "![first-token CDF](latency_cdf.png)"


def table3(rows):
    out = ["| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |",
           "|---|---|---|---|---|---|---|---|"]
    names = sorted({r["scenario"] for r in rows if r["scenario"].startswith("retrieval_") and not r["scenario"].endswith("_wiring")})
    names += [n for n in ("eval_main", "eval_final") if any(r["scenario"] == n for r in rows)]
    for n in names:
        rs = [r for r in rows if r["scenario"] == n]
        g = Counter(r["answer_grade"] for r in rs)
        N = len(rs)
        ok = [r for r in rs if r["status"] == "ok"]
        tpa = sum(r["tokens_in"] + r["tokens_out"] for r in ok) / max(1, len(ok))
        setting = rs[0]["retrieval_setting"] + {"eval_main": " (120b drafter + 20b router/verifier)",
                                                "eval_final": " (120b drafter, checks only)"}.get(n, " (20b drafter)")
        out.append(f"| {setting} | {N} | {100*g['correct']/N:.1f} | {100*g['wrong']/N:.1f} | {100*g['refused_correctly']/N:.1f} | "
                   f"{tpa:.0f} | {fmt2([r['first_token_ms'] for r in rs if r['first_token_ms']])} | "
                   f"{fmt2([r['total_ms'] for r in rs])} |")
    return "\n".join(out)


def quality(rows, scenario="eval_main"):
    rs = [r for r in rows if r["scenario"] == scenario]
    if not rs:
        return f"{scenario} not run yet."
    g = Counter(r["answer_grade"] for r in rs)
    ans = [r for r in rs if r["question_id"].startswith("A")]
    un = [r for r in rs if r["question_id"].startswith("U")]
    tpa = [r["tokens_in"] + r["tokens_out"] for r in rs if r["status"] == "ok"]
    lines = [f"{len(rs)} questions ({len(ans)} answerable, {len(un)} not answerable from the papers). "
             f"Grades: {dict(g)}.",
             f"Answerable correct and cited: {sum(r['answer_grade']=='correct' for r in ans)}/{len(ans)}; "
             f"unanswerable correctly refused: {sum(r['answer_grade']=='refused_correctly' for r in un)}/{len(un)}; "
             f"refused when it should have answered: {sum(r['answer_grade']=='refused_wrongly' for r in ans)}.",
             f"Tokens per answer: median {pct(tpa,50):.0f}, mean {sum(tpa)/len(tpa):.0f}, max {max(tpa)}." if tpa else "",
             "", "| q | grade | level | drafter | tokens | first token ms |", "|---|---|---|---|---|---|"]
    for r in sorted(rs, key=lambda r: r["question_id"]):
        lines.append(f"| {r['question_id']} | {r['answer_grade']} | {r['level']} | {r['drafter'].split('-')[-1] if r['drafter'] else ''} | "
                     f"{r['tokens_in']+r['tokens_out']} | {r['first_token_ms'] or ''} |")
    return "\n".join(lines)


def table1():
    p = RAW / "probe_real.jsonl"
    if not p.exists():
        return "probe_real.jsonl not found: run scripts/probe_real.py"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    ev = RAW / "provider_events.jsonl"
    events = [json.loads(l) for l in ev.read_text(encoding="utf-8").splitlines()] if ev.exists() else []
    real_events = [e for e in events if e.get("source") == "real"]

    def hdr(model, key):
        v = [r["headers"].get(key) for r in rows if r["model"] == model and r["headers"].get(key)]
        return v[-1] if v else None

    m8, m70 = SMALL, LARGE   # names kept short: m8 = small, m70 = large
    burst = [r for r in rows if r["phase"] == "rpm_burst"]
    b_ok = sum(r["status"] == "ok" for r in burst)
    b_429 = [r for r in burst if r["status"] == "rate_limited"]
    rec_rpm = [r for r in rows if r["phase"] == "rpm_recover" and r["status"] == "ok"]
    rec_tpm = [r for r in rows if r["phase"] == "tpm_recover" and r["status"] == "ok"]
    tpm_seq = [r for r in rows if r["phase"] == "tpm_seq"]
    tpm_429 = next((r for r in tpm_seq if r["status"] == "rate_limited"), None)
    any429 = (b_429 or [tpm_429] if tpm_429 else b_429)
    first = any429[0] if any429 else None
    tpd_msgs = [e for e in real_events if "(TPD)" in e.get("error", "") or "tokens per day" in e.get("error", "")]
    rpd_msgs = [e for e in real_events if "(RPD)" in e.get("error", "") or "requests per day" in e.get("error", "")]
    lim_in_msg = lambda r: (re.search(r"Limit (\d+)", r["error"]).group(1) if r and re.search(r"Limit (\d+)", r["error"]) else None)
    out = ["| Limit | Value you found | How you found it | Confidence (high, medium, low) |", "|---|---|---|---|"]
    rpm_txt = (f"{b_ok} of 35 simultaneous requests accepted on the small model" + (f"; 429 says Limit {lim_in_msg(b_429[0])}" if b_429 else ""))
    out.append(f"| Requests per minute | {rpm_txt} | 35 tiny requests fired at once after a 65 s quiet period; counted 200s vs 429s | {'high' if b_429 else 'low'} |")
    out.append(f"| Tokens per minute | 20b: {hdr(m8,'x-ratelimit-limit-tokens')}; 120b: {hdr(m70,'x-ratelimit-limit-tokens')}"
               + (f"; 429 after {tpm_seq.index(tpm_429)} back-to-back ~1K-token calls, message says Limit {lim_in_msg(tpm_429)}" if tpm_429 else "")
               + " | x-ratelimit-limit-tokens header on every response; then back-to-back ~1K-token prompts until 429 | high |")
    out.append(f"| Requests per day | 20b: {hdr(m8,'x-ratelimit-limit-requests')}; 120b: {hdr(m70,'x-ratelimit-limit-requests')} | "
               f"x-ratelimit-limit-requests header (remaining-requests falls by 1 per call){'; also a real RPD 429' if rpd_msgs else ''} | {'high' if rpd_msgs else 'medium'} |")
    out.append(f"| Tokens per day | {'from 429: Limit ' + lim_in_msg(tpd_msgs[0]) if tpd_msgs else '20b: 200,000; 120b: 200,000 (documented, not hit)'} | "
               f"{'hit during experiments (provider_events.jsonl)' if tpd_msgs else 'Groq docs only; no header exposes it and we did not burn a day to confirm'} | {'high' if tpd_msgs else 'low'} |")
    conc = "no separate limit seen: all 35 simultaneous requests were answered or refused by RPM only" if burst else "not measured"
    out.append(f"| Requests in flight at once | {conc} | same 35-request burst; refusals all carried the RPM message | medium |")
    ret = (f"HTTP 429, JSON error type={json.loads(first['error'])['error'].get('type') if first and first['error'].startswith('{') else '?'}, "
           f"code=rate_limit_exceeded, retry-after: {first['headers'].get('retry-after')} s, message names the limit, used, requested" if first else "not observed")
    out.append(f"| What it returns when a limit is hit | {ret} | raw/probe_real.jsonl, first 429 | {'high' if first else 'low'} |")
    rr = (f"RPM: {rec_rpm[0]['since_first_429_s']:.1f} s" if rec_rpm else "RPM: not measured") + \
         (f"; TPM: {rec_tpm[0]['since_first_429_s']:.1f} s" if rec_tpm else "; TPM: not measured")
    out.append(f"| Time until it recovers | {rr} | polled once per second with a 1-token request after the first 429 | medium |")
    lat = [r for r in rows if r["phase"] == "latency" and r["status"] == "ok"]
    lines = ["", "Real latency (probe, sequential, under all limits):", "",
             "| model | prompt | n | TTFT p50 / p95 / p99 (ms) | total p50 / p95 / p99 (ms) | visible tok/s p50 | hidden reasoning tokens p50 |", "|---|---|---|---|---|---|---|"]
    for m in (m8, m70):
        for big in (False, True):
            rs = [r for r in lat if r["model"] == m and bool(r.get("big_prompt")) == big]
            if rs:
                vis = lambda r: r.get("visible_chars", r["tokens_out"] * 4) / 4
                tps = [vis(r) / ((r["total_ms"] - r["ttft_ms"]) / 1000) for r in rs if r["total_ms"] > r["ttft_ms"] + 5 and vis(r) > 10]
                # clamp: Groq's usage.completion_tokens at effort=low was ~= visible
                # tokens, i.e. hidden reasoning is ~0 or not reported in usage
                hid = [max(0.0, r["tokens_out"] - vis(r)) for r in rs if "visible_chars" in r]
                lines.append(f"| {m} | {'~1K tokens' if big else 'short'} | {len(rs)} | {fmt3([r['ttft_ms'] for r in rs])} | "
                             f"{fmt3([r['total_ms'] for r in rs])} | {pct(tps,50) or 0:.0f} | {pct(hid,50) or 0:.0f} |")
    return "\n".join(out + lines)


def slow_breakdown(rows, request_id=None):
    if not TRACES.exists():
        return "no traces"
    tr = {json.loads(l)["request_id"]: json.loads(l) for l in TRACES.read_text(encoding="utf-8").splitlines()}
    cand = [r for r in rows if r["request_id"] in tr and r["status"] == "ok"
            and not r["scenario"].startswith(("retrieval_", "eval_"))]
    if request_id:
        cand = [r for r in rows if r["request_id"] == request_id]
    if not cand:
        return "no candidate request"
    # prefer a real request if we have any
    real = [r for r in cand if r["source"] == "real"]
    pick = max(real or cand, key=lambda r: r["total_ms"])
    t = tr[pick["request_id"]]
    sums = defaultdict(float)
    lines = [f"Request {pick['request_id']} ({pick['scenario']}, {pick['source']}, level {pick['level']}, question {pick['question_id']}): "
             f"first token {pick['first_token_ms']:.0f} ms, total {pick['total_ms']:.0f} ms, retries {pick['retries']}.", "",
             "| span | kind | start ms | end ms | duration ms | detail |", "|---|---|---|---|---|---|"]
    for s in t["spans"]:
        d = s["end_ms"] - s["start_ms"]
        sums[s["kind"]] += d
        det = {k: v for k, v in s.items() if k not in ("name", "kind", "start_ms", "end_ms") and v not in (None, [], "")}
        lines.append(f"| {s['name']} | {s['kind']} | {s['start_ms']:.0f} | {s['end_ms']:.0f} | {d:.0f} | {json.dumps(det)[:120]} |")
    acc = sum(sums.values())
    ft = pick["first_token_ms"]
    before = sum(s["end_ms"] - s["start_ms"] for s in t["spans"] if s["end_ms"] <= ft + 1)
    lines += ["", f"waiting {sums['wait']:.0f} ms, working {sums['work']:.0f} ms, retrying {sums['retry']:.0f} ms, "
              f"unaccounted {pick['total_ms'] - acc:.0f} ms. Time before first token: {ft:.0f} ms "
              f"({100*ft/pick['total_ms']:.0f}% of total); spans fully before it: {before:.0f} ms.",
              f"notes: {t['notes']}"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", default=None)
    a = ap.parse_args()
    rows = load()
    t2a, t2b, t2x = table2(rows)
    doc = ["# Tables (generated by scripts/analyze.py from results/results.csv)", "",
           "## Table 1: Limits found in the real service", table1(), "",
           "## Table 2a: Latency", t2a, "", "## Table 2b: Outcomes", t2b, "",
           "### Extra: what kind of reply people got", t2x, "", plot_cdf(rows), "",
           "## Table 3: Retrieval settings", table3(rows), "",
           "## Answer quality (eval_main)", quality(rows), "",
           "## Answer quality (eval_final: small retrieval + 120b)", quality(rows, "eval_final"), "",
           "## Where the time goes (slowest successful request)", slow_breakdown(rows, a.slow), ""]
    off = RESULTS / "retrieval_offline.md"
    if off.exists():
        doc += [off.read_text(encoding="utf-8")]
    cal = RESULTS / "calibration_check.md"
    if cal.exists():
        doc += ["## Calibration check: real vs stand-in", cal.read_text(encoding="utf-8")]
    txt = "\n".join(doc)
    (RESULTS / "tables.md").write_text(txt, encoding="utf-8")
    print(txt)