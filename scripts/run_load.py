"""Load shapes.

  python scripts/run_load.py --scenario all                # stand-in, ~30 min
  python scripts/run_load.py --scenario real_steady --real # real Groq, small
  python scripts/run_load.py --scenario burst --scale 0.25 # quick smoke test

Patience: a person waits for the first token for U(4, 14) s, then gives up.
We chose that range because incident-time questions come from people who are
already blocked; ~10 s is roughly where people start re-asking or leaving.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.config import DATA, DRAFT_MODEL, FALLBACK_DRAFT_MODEL, PipelineConfig  # noqa: E402
from relay.runner import StandInProcess, make_relay, run_schedule  # noqa: E402

QUESTIONS = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))


def patience(rng):
    return rng.uniform(4.0, 14.0)


def poisson(rate_per_min, duration_s, start=0.0, seed=0):
    rng, t, out = random.Random(seed), start, []
    while True:
        t += rng.expovariate(rate_per_min / 60)
        if t >= start + duration_s:
            return out
        out.append(t)


def burst(n, over_s, at):
    return [at + i * over_s / n for i in range(n)]


def scenarios(scale=1.0):
    d = lambda s: s * scale
    return {
        # name: (arrivals, setup, cfg overrides)
        "steady": (poisson(4, d(480), seed=1), {}, {}),
        "burst": (sorted(poisson(2, d(60), seed=2) + burst(max(8, int(40 * scale)), 10, d(60))
                         + poisson(2, d(90), start=d(70), seed=3)), {}, {}),
        "sustained_overload": (poisson(20, d(300), seed=4), {}, {}),
        "rpm_halved": (poisson(4, d(360), seed=5), {"halve": True}, {}),
        "daily_exhausted": (poisson(4, d(240), seed=6), {"exhaust": True}, {}),
        # the retry experiment, part 2: the large model 'succeeds slowly' on 30% of
        # calls (as during a provider incident). Same arrivals, two retry policies.
        "slow_provider": (poisson(8, d(180), seed=8), {"slow": True}, {}),
        "slow_provider_naive_retry": (poisson(8, d(180), seed=8), {"slow": True}, {"retry_policy": "naive"}),
        "burst_naive_retry": (sorted(poisson(2, d(60), seed=2) + burst(max(8, int(40 * scale)), 10, d(60))
                                     + poisson(2, d(90), start=d(70), seed=3)), {}, {"retry_policy": "naive"}),
    }


async def run_standin(names, scale):
    sc = scenarios(scale)
    async with StandInProcess() as si:
        for name in names:
            arrivals, setup, over = sc[name]
            await si.admin("reset", {"_config_reload": True})
            if setup.get("halve"):
                cfg = (await si.admin("config", {}))
                await si.admin("config", {m: {"rpm": v["rpm"] / 2, "tpm": v["tpm"] / 2}
                                          for m, v in cfg.items() if not m.startswith("_")})
            if setup.get("slow"):
                await si.admin("config", {DRAFT_MODEL: {"p_slow": 0.30, "slow_mult": [6.0, 15.0]}})
            if setup.get("exhaust"):
                cfg = (await si.admin("config", {}))
                b70, b8 = cfg[DRAFT_MODEL], cfg[FALLBACK_DRAFT_MODEL]
                # large model's daily budget fully spent; small has ~25 requests' worth left
                await si.admin("reset", {DRAFT_MODEL: {"day_requests": b70["rpd"], "day_tokens": b70["tpd"]},
                                         FALLBACK_DRAFT_MODEL: {"day_requests": b8["rpd"] - 25,
                                                                "day_tokens": b8["tpd"] - 40000}})
            relay = make_relay("standin", PipelineConfig(**over), limits=_standin_limits(await si.admin("config", {})))
            if setup.get("exhaust"):
                # the governor learns this from headers on the first call; we
                # also start it with what yesterday's counters would say
                relay.gov[DRAFT_MODEL].day_tokens = relay.gov[DRAFT_MODEL].tpd
                relay.gov[FALLBACK_DRAFT_MODEL].day_tokens = relay.gov[FALLBACK_DRAFT_MODEL].tpd - 40000
                relay.gov[FALLBACK_DRAFT_MODEL].day_remaining_requests = 25
            print(f"[{name}] {len(arrivals)} requests over {arrivals[-1]:.0f}s", flush=True)
            recs = await run_schedule(relay, name, arrivals, QUESTIONS, patience=patience, seed=hash(name) % 1000)
            await relay.client.close()
            st = {}
            for r in recs:
                st[r.status] = st.get(r.status, 0) + 1
            print(f"[{name}] done: {st}  peak_queue={relay.peak_queue}  429s={(await si.stats())['_429s']}", flush=True)


def _standin_limits(cfg):
    return {m: dict(rpm=int(v["rpm"]), tpm=int(v["tpm"]), rpd=int(v["rpd"]), tpd=int(v["tpd"]))
            for m, v in cfg.items() if not m.startswith("_")}


async def run_real(name):
    if name == "real_steady":
        arrivals = poisson(3, 300, seed=11)              # ~15 requests over 5 min
    elif name == "real_burst":
        arrivals = burst(12, 2, 0)                       # 12 people at once
    else:
        raise SystemExit(name)
    relay = make_relay("real")
    print(f"[{name}] {len(arrivals)} real requests", flush=True)
    recs = await run_schedule(relay, name, arrivals, QUESTIONS, patience=patience, grade_answers=True, seed=5)
    await relay.client.close()
    st = {}
    for r in recs:
        st[r.status] = st.get(r.status, 0) + 1
    print(f"[{name}] done: {st}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink durations for smoke tests")
    a = ap.parse_args()
    if a.real:
        asyncio.run(run_real(a.scenario))
    else:
        names = list(scenarios()) if a.scenario == "all" else a.scenario.split(",")
        asyncio.run(run_standin(names, a.scale))
