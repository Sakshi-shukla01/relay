"""Streaming OpenAI-compatible chat client (Groq or the stand-in).

One call = one HTTP request. It never retries by itself: retrying is a
policy decision that belongs to the pipeline, which knows the deadline.
It does enforce three timeouts, because 'succeeds slowly' is the failure
we saw most and a slow success holds a concurrency slot and a user.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field

import aiohttp

from .config import GROQ_BASE, RAW, REASONING_EFFORT, STANDIN_BASE

_EVENTS = RAW / "provider_events.jsonl"


def parse_duration(s: str | None) -> float | None:
    """Groq reset headers look like '2m59.56s', '7.66s', '120ms'."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    total, ok = 0.0, False
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", s):
        ok = True
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if ok else None


@dataclass
class CallResult:
    model: str
    status: str                     # ok | rate_limited | timeout | error
    http_status: int = 0
    text: str = ""
    started: float = 0.0            # time.monotonic() at send
    ttft_ms: float | None = None    # first *visible* content token, relative to send
    first_reasoning_ms: float | None = None  # first hidden-reasoning token (gpt-oss), if streamed
    total_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    retry_after_s: float | None = None
    headers: dict = field(default_factory=dict)
    error: str = ""
    server_queue_ms: float | None = None   # stand-in only (x-standin-queue-ms)


class LLMClient:
    def __init__(self, source: str = "standin", api_key: str | None = None, log_events: bool = True):
        self.source = source
        self.base = GROQ_BASE if source == "real" else STANDIN_BASE
        self.key = api_key or (os.environ.get("GROQ_API_KEY") if source == "real" else "standin")
        if source == "real" and not self.key:
            raise SystemExit("Set GROQ_API_KEY in your environment (never commit it).")
        self.log_events = log_events
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=256))
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()

    def _log(self, kind: str, res: CallResult, extra: dict | None = None):
        if not self.log_events:
            return
        rec = dict(ts=time.time(), source=self.source, kind=kind, model=res.model,
                   http_status=res.http_status, status=res.status,
                   headers={k: v for k, v in res.headers.items() if "ratelimit" in k or k in ("retry-after", "x-groq-region")},
                   error=res.error[:600])
        if extra:
            rec.update(extra)
        with _EVENTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    async def chat(self, model: str, messages: list[dict], max_tokens: int = 256,
                   temperature: float = 0.0, ttft_timeout: float = 8.0,
                   stall_timeout: float = 4.0, total_timeout: float = 30.0,
                   on_piece=None, json_mode: bool = False) -> CallResult:
        body = dict(model=model, messages=messages, max_completion_tokens=max_tokens,
                    temperature=temperature, stream=True)
        if "gpt-oss" in model:
            body["reasoning_effort"] = REASONING_EFFORT
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        res = CallResult(model=model, status="error", started=time.monotonic())
        s = await self.session()
        t0 = res.started
        try:
            async with asyncio.timeout(total_timeout):
                async with s.post(f"{self.base}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {self.key}"}) as r:
                    res.http_status = r.status
                    res.headers = {k.lower(): v for k, v in r.headers.items()}
                    if "x-standin-queue-ms" in res.headers:
                        res.server_queue_ms = float(res.headers["x-standin-queue-ms"])
                    if r.status != 200:
                        res.error = await r.text()
                        res.retry_after_s = parse_duration(res.headers.get("retry-after"))
                        if r.status == 429:
                            res.status = "rate_limited"
                            m = re.search(r"try again in ([\d.]+m?s?[\d.]*s?)", res.error)
                            if res.retry_after_s is None and m:
                                res.retry_after_s = parse_duration(m.group(1))
                        else:
                            res.status = "error"
                        res.total_ms = (time.monotonic() - t0) * 1000
                        self._log("http_error", res)
                        return res
                    parts = []
                    first_deadline = t0 + ttft_timeout
                    buf = b""
                    while True:
                        now = time.monotonic()
                        wait = (first_deadline - now) if res.ttft_ms is None else stall_timeout
                        if wait <= 0:
                            raise TimeoutError("ttft")
                        try:
                            data = await asyncio.wait_for(r.content.readany(), timeout=wait)
                        except asyncio.TimeoutError:
                            raise TimeoutError("ttft" if res.ttft_ms is None else "stall")
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line.startswith(b"data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == b"[DONE]":
                                continue
                            try:
                                ev = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            usage = (ev.get("x_groq") or {}).get("usage") or ev.get("usage")
                            if usage:
                                res.tokens_in = usage.get("prompt_tokens", 0)
                                res.tokens_out = usage.get("completion_tokens", 0)
                            if "error" in ev:
                                res.error = json.dumps(ev["error"])
                                raise RuntimeError("stream error")
                            for ch in ev.get("choices", []):
                                if (ch.get("delta") or {}).get("reasoning") and res.first_reasoning_ms is None:
                                    res.first_reasoning_ms = (time.monotonic() - t0) * 1000
                                piece = (ch.get("delta") or {}).get("content")
                                if piece:
                                    if res.ttft_ms is None:
                                        res.ttft_ms = (time.monotonic() - t0) * 1000
                                    parts.append(piece)
                                    if on_piece:
                                        on_piece(piece)
                    res.text = "".join(parts)
                    res.status = "ok"
                    if not res.text.strip() and max_tokens > 8:
                        # reasoning models can spend the whole budget thinking
                        res.status = "error"
                        res.error = "empty answer: hidden reasoning used the whole max_completion_tokens"
        except (TimeoutError, asyncio.TimeoutError) as e:
            res.status = "timeout"
            res.error = res.error or f"timeout:{e}"
        except asyncio.CancelledError:
            res.status = "timeout"
            res.error = "cancelled (user gave up)"
            res.total_ms = (time.monotonic() - t0) * 1000
            raise
        except (aiohttp.ClientError, RuntimeError) as e:
            res.status = "error"
            res.error = res.error or repr(e)
        res.total_ms = (time.monotonic() - t0) * 1000
        if not res.tokens_in:
            res.tokens_in = sum(len(m["content"]) for m in messages) // 4
        if not res.tokens_out and res.text:
            res.tokens_out = max(1, len(res.text) // 4)
        if res.status != "ok":
            self._log("call_failed", res)
        return res
