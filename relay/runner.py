"""Drives the pipeline with an arrival schedule and impatient people."""
from __future__ import annotations

import asyncio
import random
import subprocess
import sys
import time

import aiohttp

from .config import ROOT, PipelineConfig, learned_limits
from .governor import Governor
from .grading import grade, load_overrides
from .llm import LLMClient
from .pipeline import Relay
from .results import row, write


async def one(relay: Relay, q: dict, scenario: str, patience_s: float | None, hard_cap_s: float = 30.0):
    first = asyncio.Event()
    holder: dict = {}
    task = asyncio.create_task(relay.run(q["question"], scenario=scenario, qid=q["id"],
                                         sink=lambda _p: first.set(), out=holder))
    gave_up = False
    try:
        if patience_s is not None:
            done, _ = await asyncio.wait({task, asyncio.create_task(first.wait())},
                                         timeout=patience_s, return_when=asyncio.FIRST_COMPLETED)
            if not first.is_set() and not task.done():
                gave_up = True
                task.cancel()
        await asyncio.wait_for(asyncio.shield(task), timeout=hard_cap_s) if not task.cancelled() else None
    except (asyncio.CancelledError, asyncio.TimeoutError):
        if not task.done():
            task.cancel()
            gave_up = True
    try:
        await task
    except asyncio.CancelledError:
        pass
    rec = holder["rec"]
    if gave_up:
        rec.status = "timeout"
        rec.total_ms = rec.ms()
    return rec, gave_up


async def run_schedule(relay: Relay, scenario: str, arrivals: list[float], questions: list[dict],
                       patience=None, grade_answers: bool = False, seed: int = 1, on_tick=None,
                       sequential: bool = False):
    """arrivals: seconds from start. patience: callable(rng)->seconds or None."""
    rng = random.Random(seed)
    overrides = load_overrides()
    t0 = time.monotonic()
    tasks, pats = [], []
    for i, at in enumerate(arrivals):
        delay = at - (time.monotonic() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        if on_tick:
            await on_tick(at)
        q = questions[i % len(questions)]
        p = patience(rng) if patience else None
        pats.append(p)
        tasks.append(asyncio.create_task(one(relay, q, scenario, p, hard_cap_s=30.0 if patience else 180.0)))
        if sequential:
            await asyncio.wait({tasks[-1]})     # quality runs: one question at a time, no queueing
    outs = await asyncio.gather(*tasks)
    rows, recs = [], []
    qmap = {q["id"]: q for q in questions}
    for (rec, gave_up), p in zip(outs, pats):
        g = ""
        if grade_answers and rec.status in ("ok", "rate_limited"):
            g = overrides.get(f"{scenario}:{rec.qid}") or grade(qmap[rec.qid], rec.answer, rec.refused, rec.level, rec.cited_papers)
        rows.append(row(rec, g, None if p is None else p * 1000, gave_up))
        recs.append(rec)
    write(rows, recs, scenario)
    return recs


def make_relay(source: str, cfg: PipelineConfig | None = None, limits: dict | None = None) -> Relay:
    cfg = cfg or PipelineConfig()
    client = LLMClient(source)
    gov = Governor(limits or learned_limits(), cfg.max_inflight_per_model)
    return Relay(client, gov, cfg)


class StandInProcess:
    """Start/stop the stand-in server for a run."""
    def __init__(self, port=8088, seed=7):
        self.port, self.seed, self.p = port, seed, None
        self.base = f"http://127.0.0.1:{port}"

    async def __aenter__(self):
        import socket
        with socket.socket() as sk:
            if sk.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"port {self.port} is already in use: is another run still going?")
        self.p = subprocess.Popen([sys.executable, "-m", "relay.standin", "--port", str(self.port),
                                   "--seed", str(self.seed)], cwd=ROOT)
        for _ in range(100):
            try:
                async with aiohttp.ClientSession() as s, s.get(f"{self.base}/admin/stats") as r:
                    if r.status == 200:
                        return self
            except aiohttp.ClientError:
                await asyncio.sleep(0.1)
        raise RuntimeError("stand-in did not start")

    async def admin(self, path, payload):
        async with aiohttp.ClientSession() as s, s.post(f"{self.base}/admin/{path}", json=payload) as r:
            return await r.json()

    async def stats(self):
        async with aiohttp.ClientSession() as s, s.get(f"{self.base}/admin/stats") as r:
            return await r.json()

    async def __aexit__(self, *a):
        self.p.terminate()
        self.p.wait(timeout=5)
