"""Client-side rate governor.

The provider's limits are enforced *after* we have spent a round trip and a
user's patience. So we keep our own model of every per-model budget, and ask
it before each call: 'can this call start now, and if not, how long until it
can?'. The pipeline uses the answer to decide whether to wait, degrade, or
refuse. The model starts from documented limits and is corrected by the
x-ratelimit-* headers on every response, because the limits 'can change
without notice'.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from .llm import CallResult, parse_duration


@dataclass
class ModelBudget:
    model: str
    rpm: int
    tpm: int
    rpd: int
    tpd: int
    safety: float = 0.9
    window: deque = field(default_factory=deque)     # (t, tokens) sent in last 60 s
    day_requests: int = 0
    day_tokens: int = 0
    day_remaining_requests: int | None = None        # from headers
    blocked_until: float = 0.0                       # after a 429
    day_exhausted_until: float = 0.0
    minute_tokens_remaining: int | None = None
    minute_tokens_reset_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    n_429: int = 0

    def _trim(self, now):
        while self.window and now - self.window[0][0] >= 60:
            self.window.popleft()

    def day_left_frac(self) -> float:
        req_left = self.rpd - self.day_requests if self.day_remaining_requests is None else self.day_remaining_requests
        tok_left = self.tpd - self.day_tokens
        return max(0.0, min(req_left / max(1, self.rpd), tok_left / max(1, self.tpd)))

    def day_exhausted(self, reserve_frac: float = 0.0) -> bool:
        if time.monotonic() < self.day_exhausted_until:
            return True
        return self.day_left_frac() <= reserve_frac

    def wait_needed(self, est_tokens: int, now: float | None = None) -> float:
        """Seconds until a call of est_tokens may start (inf = not today)."""
        now = now or time.monotonic()
        if now < self.day_exhausted_until:
            return float("inf")
        if self.day_tokens + est_tokens > self.tpd or (self.day_remaining_requests is not None and self.day_remaining_requests <= 0):
            return float("inf")
        self._trim(now)
        waits = [max(0.0, self.blocked_until - now)]
        rpm_cap = max(1, int(self.rpm * self.safety))
        if len(self.window) >= rpm_cap:
            waits.append(60 - (now - self.window[len(self.window) - rpm_cap][0]))
        tpm_cap = self.tpm * self.safety
        if est_tokens > tpm_cap:
            return float("inf")             # can never fit in one minute
        used = sum(t for _, t in self.window)
        if used + est_tokens > tpm_cap:
            # time until enough old tokens fall out of the window
            need, acc = used + est_tokens - tpm_cap, 0
            for t0, tok in self.window:
                acc += tok
                if acc >= need:
                    waits.append(60 - (now - t0))
                    break
        if self.minute_tokens_remaining is not None and now < self.minute_tokens_reset_at \
                and self.minute_tokens_remaining < est_tokens:
            waits.append(self.minute_tokens_reset_at - now)
        return max(waits)

    async def acquire(self, est_tokens: int, max_wait_s: float) -> float | None:
        """Wait for a slot. Returns seconds waited, or None if the wait would
        exceed max_wait_s (caller should degrade instead of queueing)."""
        t_start = time.monotonic()
        if self.wait_needed(est_tokens) > max_wait_s:
            return None                # fail fast: do not even join the line
        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=max(0.01, max_wait_s))
        except asyncio.TimeoutError:
            return None
        try:                           # FIFO admission per model
            while True:
                now = time.monotonic()
                w = self.wait_needed(est_tokens, now)
                if w <= 0:
                    self.window.append((now, est_tokens))
                    self.day_requests += 1
                    self.day_tokens += est_tokens
                    return now - t_start
                if (now - t_start) + w > max_wait_s:
                    return None
                await asyncio.sleep(min(w, 0.5))
        finally:
            self.lock.release()

    def settle(self, est_tokens: int, res: CallResult):
        """Correct our estimate with the real usage and the provider's headers."""
        actual = res.tokens_in + res.tokens_out
        if actual and self.window:
            # replace the estimate with actual for the most recent matching entry
            for i in range(len(self.window) - 1, -1, -1):
                if self.window[i][1] == est_tokens:
                    self.window[i] = (self.window[i][0], actual)
                    break
            self.day_tokens += actual - est_tokens
        h = res.headers
        now = time.monotonic()
        try:
            if "x-ratelimit-limit-requests" in h:
                self.rpd = int(float(h["x-ratelimit-limit-requests"]))
            if "x-ratelimit-remaining-requests" in h:
                self.day_remaining_requests = int(float(h["x-ratelimit-remaining-requests"]))
            if "x-ratelimit-limit-tokens" in h:
                self.tpm = int(float(h["x-ratelimit-limit-tokens"]))
            if "x-ratelimit-remaining-tokens" in h:
                self.minute_tokens_remaining = int(float(h["x-ratelimit-remaining-tokens"]))
                self.minute_tokens_reset_at = now + (parse_duration(h.get("x-ratelimit-reset-tokens")) or 0)
        except ValueError:
            pass
        if res.status == "rate_limited":
            self.n_429 += 1
            ra = res.retry_after_s if res.retry_after_s is not None else 2.0
            msg = res.error.lower()
            if "per day" in msg or "(rpd)" in msg or "(tpd)" in msg:
                self.day_exhausted_until = now + ra
            self.blocked_until = max(self.blocked_until, now + ra)


class Governor:
    def __init__(self, limits: dict, max_inflight: int = 8):
        self.b = {m: ModelBudget(m, **v) for m, v in limits.items()}
        self.sem = {m: asyncio.Semaphore(max_inflight) for m in limits}

    def __getitem__(self, m) -> ModelBudget:
        return self.b[m]

    def snapshot(self) -> dict:
        now = time.monotonic()
        out = {}
        for m, b in self.b.items():
            b._trim(now)
            out[m] = dict(window_reqs=len(b.window), window_tokens=sum(t for _, t in b.window),
                          day_left=round(b.day_left_frac(), 4), n_429=b.n_429)
        return out
