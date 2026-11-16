"""Stand-in model service: an OpenAI/Groq-compatible chat endpoint we control.

It can be slow, fail, refuse with a Groq-shaped 429 when a limit is reached,
get slower under load, and stream its reply a piece at a time.
Numbers come from results/standin_config.json, written by scripts/calibrate.py
from the real probe. Without it, it uses DEFAULTS (marked uncalibrated).

Run:  python -m relay.standin --port 8088
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
from collections import deque

from aiohttp import web

from .config import LARGE, RESULTS, SMALL

DEFAULT_MODEL = dict(
    rpm=30, tpm=6000, rpd=14400, tpd=500000,
    ttft_mu_ms=300.0, ttft_sigma=0.45,       # lognormal: median, shape
    prefill_ms_per_1k=25.0,                  # extra TTFT per 1k prompt tokens
    decode_tok_s=500.0,                      # streaming speed when idle
    capacity=6,                              # concurrent requests before queueing
    load_alpha=0.8,                          # decode slowdown per (inflight/capacity)
    p_slow=0.03, slow_mult=(4.0, 12.0),      # 'succeeds slowly'
    p_error=0.01,                            # 5xx
    p_stall=0.01, stall_s=(3.0, 8.0),        # stream pauses mid-way
    chunk_tokens=4,
    reasoning_tokens=120.0,                  # hidden 'thinking' before the first visible token
)
DEFAULTS = {
    "_meta": {"calibrated": False, "note": "defaults; run scripts/calibrate.py after probe_real.py"},
    SMALL: dict(DEFAULT_MODEL, tpm=8000, rpd=1000, tpd=200000, ttft_mu_ms=250.0, decode_tok_s=1000.0),
    LARGE: dict(DEFAULT_MODEL, tpm=8000, rpd=1000, tpd=200000, ttft_mu_ms=350.0, decode_tok_s=500.0),
}


def load_config() -> dict:
    p = RESULTS / "standin_config.json"
    cfg = json.loads(json.dumps(DEFAULTS))
    if p.exists():
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            if isinstance(v, dict):
                cfg.setdefault(k, dict(DEFAULT_MODEL)).update(v)
    return cfg


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ModelState:
    def __init__(self, name, c):
        self.name, self.c = name, c
        self.win = deque()            # (t, tokens)
        self.day_req = 0
        self.day_tok = 0
        self.inflight = 0
        self.sem = asyncio.Semaphore(int(c["capacity"]))

    def trim(self, now):
        while self.win and now - self.win[0][0] >= 60:
            self.win.popleft()

    def headers(self, now):
        self.trim(now)
        used_tok = sum(t for _, t in self.win)
        rem_tok = max(0, int(self.c["tpm"]) - used_tok)
        reset_tok = 0.0 if not self.win else max(0.0, 60 - (now - self.win[0][0])) * (used_tok / max(1, self.c["tpm"]))
        return {
            "x-ratelimit-limit-requests": str(int(self.c["rpd"])),
            "x-ratelimit-remaining-requests": str(max(0, int(self.c["rpd"]) - self.day_req)),
            "x-ratelimit-limit-tokens": str(int(self.c["tpm"])),
            "x-ratelimit-remaining-tokens": str(rem_tok),
            "x-ratelimit-reset-requests": f"{86400 / max(1, self.c['rpd']):.2f}s",
            "x-ratelimit-reset-tokens": f"{reset_tok:.2f}s",
        }

    def check(self, now, req_tokens):
        """Return None if admitted, else (limit_name, used, limit, retry_after_s)."""
        self.trim(now)
        c = self.c
        if self.day_req + 1 > c["rpd"]:
            return ("requests per day (RPD)", self.day_req, c["rpd"], 3600.0)
        if self.day_tok + req_tokens > c["tpd"]:
            return ("tokens per day (TPD)", self.day_tok, c["tpd"], 1800.0)
        if len(self.win) + 1 > c["rpm"]:
            return ("requests per minute (RPM)", len(self.win), c["rpm"], 60 - (now - self.win[0][0]))
        used = sum(t for _, t in self.win)
        if used + req_tokens > c["tpm"]:
            need, acc, ra = used + req_tokens - c["tpm"], 0, 60.0
            for t0, tok in self.win:
                acc += tok
                if acc >= need:
                    ra = 60 - (now - t0)
                    break
            return ("tokens per minute (TPM)", used, c["tpm"], ra)
        return None


# ---- fake content -------------------------------------------------------------
_W = re.compile(r"[a-z0-9.]+")


def _answer_like(system: str, user: str) -> str:
    if "ROLE: router" in system:
        q = user.lower()
        paper = "moshi" if any(w in q for w in ("moshi", "mimi", "helium", "speech", "audio")) else \
            "hnet" if any(w in q for w in ("h-net", "hnet", "chunk", "byte", "tokeniz")) else \
            "attention" if any(w in q for w in ("transformer", "bleu", "attention", "wmt")) else "none"
        return json.dumps({"in_scope": True, "paper": paper, "query": user[:200]})
    if "ROLE: verifier" in system:
        return "SUPPORTED"
    if "ROLE: drafter" in system:
        m = re.search(r"Question:(.*?)\n\nPassages:\n(.*)", user, re.S)
        if not m:
            return "NOT IN PAPERS: the passages do not cover this."
        q = set(_W.findall(m.group(1).lower()))
        passages = re.split(r"\n\n(?=\[\d+\] \()", m.group(2))
        best = (0, None, None)
        for p in passages:
            mm = re.match(r"\[(\d+)\] \([^)]*\)\n(.*)", p, re.S)
            if not mm:
                continue
            for s in re.split(r"(?<=[.!?])\s+", mm.group(2)):
                sc = len(q & set(_W.findall(s.lower())))
                if sc > best[0] and 6 < len(s.split()) < 60:
                    best = (sc, s.strip(), mm.group(1))
        if best[0] < 2:
            return "NOT IN PAPERS: the passages discuss related material but not this."
        s = best[1].rstrip(".")
        return f"{s} [{best[2]}]. This is what the cited passage states [{best[2]}]."
    # anything else (the probe's prompts): a one-sentence-length reply
    return " ".join(["token"] * 36) + "."


class StandIn:
    def __init__(self, cfg: dict, seed: int = 7):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.models = {m: ModelState(m, c) for m, c in cfg.items() if not m.startswith("_")}
        self.log = []

    def model(self, name):
        if name not in self.models:
            self.models[name] = ModelState(name, dict(DEFAULT_MODEL))
        return self.models[name]

    async def chat(self, request: web.Request):
        t_arr = time.monotonic()
        body = await request.json()
        ms = self.model(body.get("model", ""))
        c = ms.c
        msgs = body.get("messages", [])
        system = " ".join(m["content"] for m in msgs if m["role"] == "system")
        user = "\n".join(m["content"] for m in msgs if m["role"] != "system")
        p_tok = est_tokens(system + user)
        max_t = int(body.get("max_completion_tokens") or body.get("max_tokens") or 256)
        lim = ms.check(t_arr, p_tok + max_t)
        if lim:
            name, used, limit, ra = lim
            ra = max(0.5, ra)
            msg = (f"Rate limit reached for model `{ms.name}` in organization `org_standin` service tier `on_demand` "
                   f"on {name}: Limit {int(limit)}, Used {int(used)}, Requested {p_tok + max_t if 'token' in name else 1}. "
                   f"Please try again in {ra:.2f}s.")
            h = ms.headers(t_arr) | {"retry-after": str(int(math.ceil(ra)))}
            self.log.append(("429", ms.name, name))
            return web.json_response({"error": {"message": msg, "type": "tokens" if "token" in name else "requests",
                                                "code": "rate_limit_exceeded"}}, status=429, headers=h)
        ms.win.append((t_arr, p_tok + max_t))
        ms.day_req += 1
        ms.day_tok += p_tok + max_t
        if self.rng.random() < c["p_error"]:
            await asyncio.sleep(self.rng.uniform(0.05, 0.4))
            return web.json_response({"error": {"message": "Service Unavailable", "type": "internal_server_error"}},
                                     status=503, headers=ms.headers(time.monotonic()))
        text = _answer_like(system, user)
        reasoning = 0
        if "gpt-oss" in ms.name or c.get("reasoning_tokens", 0) > 0:
            reasoning = int(self.rng.lognormvariate(math.log(max(1.0, c.get("reasoning_tokens", 0))), 0.5)) if c.get("reasoning_tokens", 0) else 0
        if reasoning >= max_t:          # thought for the whole budget: empty answer, like the real thing
            reasoning, text = max_t, ""
        out_tokens = min(max_t, reasoning + est_tokens(text))
        # --- latency model ---
        ms.inflight += 1
        try:
            async with ms.sem:                       # server-side queue
                q_ms = (time.monotonic() - t_arr) * 1000
                load = ms.inflight / max(1, c["capacity"])
                ttft = self.rng.lognormvariate(math.log(c["ttft_mu_ms"]), c["ttft_sigma"])
                ttft += c["prefill_ms_per_1k"] * p_tok / 1000
                ttft *= 1 + 0.5 * c["load_alpha"] * max(0.0, load - 0.5)
                if self.rng.random() < c["p_slow"]:
                    ttft *= self.rng.uniform(*c["slow_mult"])
                rate0 = c["decode_tok_s"] / (1 + c["load_alpha"] * load)
                ttft += reasoning / rate0 * 1000          # hidden reasoning happens before visible tokens
                stall_at = self.rng.randint(1, max(1, out_tokens - reasoning)) if self.rng.random() < c["p_stall"] else -1
                rate = c["decode_tok_s"] / (1 + c["load_alpha"] * load)
                resp = web.StreamResponse(status=200, headers=ms.headers(time.monotonic()) |
                                          {"content-type": "text/event-stream", "x-standin-queue-ms": f"{q_ms:.1f}"})
                await resp.prepare(request)
                await asyncio.sleep(ttft / 1000)
                step = int(c["chunk_tokens"]) * 4
                sent_tok = 0
                for i in range(0, len(text), step):
                    piece = text[i:i + step]
                    ev = {"id": "standin", "object": "chat.completion.chunk", "model": ms.name,
                          "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                    await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
                    sent_tok += max(1, len(piece) // 4)
                    if stall_at >= 0 and sent_tok >= stall_at:
                        await asyncio.sleep(self.rng.uniform(*c["stall_s"]))
                        stall_at = -1
                    await asyncio.sleep(max(1, len(piece) // 4) / rate)
                usage = {"prompt_tokens": p_tok, "completion_tokens": out_tokens, "total_tokens": p_tok + out_tokens}
                fin = {"id": "standin", "object": "chat.completion.chunk", "model": ms.name,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "x_groq": {"usage": usage}}
                await resp.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())
                # correct the minute window: Groq counts actual tokens once known
                for j in range(len(ms.win) - 1, -1, -1):
                    if ms.win[j][0] == t_arr:
                        ms.win[j] = (t_arr, p_tok + out_tokens)
                        break
                ms.day_tok += out_tokens - max_t
                await resp.write_eof()
                return resp
        except ConnectionResetError:
            return web.Response(status=499)     # client gave up mid-stream
        finally:
            ms.inflight -= 1

    # ---- admin: change behaviour mid-run ----------------------------------------
    async def admin_config(self, request):
        upd = await request.json()
        for m, v in upd.items():
            st = self.model(m)
            st.c.update(v)
            if "capacity" in v:
                st.sem = asyncio.Semaphore(int(v["capacity"]))
        return web.json_response({m: s.c for m, s in self.models.items()})

    async def admin_reset(self, request):
        upd = await request.json() if request.can_read_body else {}
        for m, s in self.models.items():
            s.win.clear()
            s.day_req = int(upd.get(m, {}).get("day_requests", 0))
            s.day_tok = int(upd.get(m, {}).get("day_tokens", 0))
        if upd.get("_config_reload"):
            cfg = load_config()
            for m, s in self.models.items():
                if m in cfg:
                    s.c = dict(cfg[m])
                    s.sem = asyncio.Semaphore(int(s.c["capacity"]))
        self.log.clear()
        return web.json_response({"ok": True})

    async def admin_stats(self, request):
        now = time.monotonic()
        return web.json_response({m: dict(inflight=s.inflight, window=len(s.win), day_req=s.day_req,
                                          day_tok=s.day_tok, headers=s.headers(now)) for m, s in self.models.items()}
                                 | {"_429s": len([x for x in self.log if x[0] == "429"])})


def make_app(seed=7) -> web.Application:
    s = StandIn(load_config(), seed)
    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.add_routes([web.post("/openai/v1/chat/completions", s.chat),
                    web.post("/admin/config", s.admin_config),
                    web.post("/admin/reset", s.admin_reset),
                    web.get("/admin/stats", s.admin_stats)])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    web.run_app(make_app(a.seed), host="127.0.0.1", port=a.port, access_log=None, print=lambda *_: None)
