"""The Relay pipeline.

    understand (small router) -> retrieve (tool) -> decide (branch / re-retrieve loop)
      -> draft (large, cited) -> check (deterministic) -> verify (small) -> [revise loop] -> send

    small = openai/gpt-oss-20b, large = openai/gpt-oss-120b (see config.py)

Shaping decision: the answer is *held until it has been checked* and only
then streamed to the person. We never stream unverified tokens and then
retract them. That costs the draft's generation time on time-to-first-token;
on Groq that time is small (it generates hundreds of tokens per second), and
we measured what it costs (model_ttft_ms vs first_token_ms in results.csv).

Every request picks a *level* when it starts and can drop a level at any step
if the budget or the deadline says so:

    L0 full      router + large draft + checks + small verifier
    L1 no-small  large draft + deterministic checks          (small model's minute budget tight)
    L2 small     small draft on fewer passages + checks      (large model's minute/day budget gone)
    L3 excerpts  no model call: the top passages, cited    (all budgets gone / deadline)

Nobody at any level gets an answer that failed the deterministic checks.
Excerpt replies say why: "busy" (minute budget), "limit" (daily budget) or "unverified"
(checks rejected the draft; counted as a refusal with status ok, not as rate-limited).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field

from .config import (DRAFT_MODEL, FALLBACK_DRAFT_MODEL, PRESSURE_RETRIEVAL, PRICES,
                     RETRIEVAL_SETTINGS, ROUTER_MODEL, VERIFY_MODEL, PipelineConfig)
from .corpus import PAPER_BY_KEY
from .governor import Governor
from .llm import CallResult, LLMClient
from .retrieval import Hit, coverage, retrieve
from .verify import REFUSAL_MARK, check_answer, cited_ids, is_refusal


@dataclass
class Record:
    request_id: str
    scenario: str
    source: str
    question: str
    qid: str = ""
    start_wall: float = 0.0
    t0: float = 0.0
    status: str = "error"
    level: str = ""
    drafter: str = ""
    retrieval_setting: str = ""
    first_token_ms: float | None = None
    total_ms: float = 0.0
    queue_wait_ms: float = 0.0
    model_ttft_ms: float | None = None   # provider's first token of the draft, from request start
    retries: int = 0
    model_calls: int = 0
    tokens: dict = field(default_factory=dict)   # model -> [in, out]
    spans: list = field(default_factory=list)
    answer: str = ""
    refused: bool = False
    cited_papers: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def ms(self) -> float:
        return (time.monotonic() - self.t0) * 1000

    def span(self, name, kind, start_ms, **kw):
        self.spans.append(dict(name=name, kind=kind, start_ms=round(start_ms, 1), end_ms=round(self.ms(), 1), **kw))

    @property
    def tokens_in(self):
        return sum(v[0] for v in self.tokens.values())

    @property
    def tokens_out(self):
        return sum(v[1] for v in self.tokens.values())

    @property
    def cost_usd(self):
        c = 0.0
        for m, (ti, to) in self.tokens.items():
            pi, po = PRICES.get(m, (0, 0))
            c += ti * pi / 1e6 + to * po / 1e6
        return c


ROUTER_SYS = """ROLE: router
You route engineering questions for an assistant that may ONLY answer from three papers:
- attention: "Attention Is All You Need" (Transformer, machine translation, 2017)
- moshi: "Moshi: a speech-text foundation model for real-time dialogue" (Mimi codec, Helium, Kyutai)
- hnet: "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (H-Net, byte-level, tokenizer-free)
Return JSON only: {"in_scope": true|false, "paper": "attention"|"moshi"|"hnet"|"none", "query": "<keyword search query>"}
in_scope=false only if the question clearly cannot be about these papers. The query should keep every
technical term and number from the question and add likely synonyms used in the papers."""

DRAFT_SYS = """ROLE: drafter
You answer engineers' questions using ONLY the numbered passages from research papers.
Rules:
1. Every sentence that states a fact ends with the passage number(s) it came from, like [2] or [1, 3].
2. Copy numbers exactly as written in the passages. Never compute or recall numbers not in the passages.
3. If the passages do not contain the answer, reply with exactly: NOT IN PAPERS: <one sentence on what the passages do cover>
4. If the papers contradict each other or themselves, say so and cite both.
5. Be brief: 1-4 sentences. No preamble."""

VERIFY_SYS = """ROLE: verifier
You check a drafted answer against the passages it cites. Reply with one line:
SUPPORTED
or
UNSUPPORTED: <the claim that the passages do not support>"""


# The Transformer paper cites its references as [31, 2, 8], the same shape as our passage
# labels. Strip them from passage text so a copied "[36]" can't be read as passage 36
# (false "does not exist") or a copied "[2]" as passage 2 (false pass).
REF_MARK = re.compile(r"\[\d{1,3}(?:\s*[,\u2013-]\s*\d{1,3})*\]")


def _clean(text: str) -> str:
    return REF_MARK.sub("", text)


def _passages_block(hits: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        lines.append(f"[{i}] ({PAPER_BY_KEY[c.paper]['title']}, {c.section}, p.{c.page_start})\n{_clean(c.text)}")
    return "\n\n".join(lines)


def _sources_block(hits: list[Hit], ids: list[int]) -> str:
    used = sorted(set(i for i in ids if 1 <= i <= len(hits)))
    return "\n".join(f"[{i}] {hits[i - 1].chunk.cite()}" for i in used)


def _est_tokens(messages, max_tokens) -> int:
    return sum(len(m["content"]) for m in messages) // 4 + max_tokens + 8


class Relay:
    def __init__(self, client: LLMClient, gov: Governor, cfg: PipelineConfig | None = None):
        self.client, self.gov = client, gov
        self.cfg = cfg or PipelineConfig()
        self.workers = asyncio.Semaphore(32)    # requests being worked on at once
        self.queue_len = 0
        self.peak_queue = 0
        # running estimates of tokens per call, used to predict budget waits
        self.est_draft = 3000.0
        self.est_small = 1800.0

    # ---- one model call, with the retry policy -------------------------------
    async def _call(self, rec: Record, step: str, model: str, messages, max_tokens: int,
                    deadline_ms: float, ttft_timeout: float, json_mode=False) -> CallResult | None:
        """Returns a successful CallResult, or None if this step should give up
        (caller decides the fallback). Never waits past deadline_ms."""
        attempts = 0
        max_attempts = 4 if self.cfg.retry_policy == "naive" else 2
        b = self.gov[model]
        est = _est_tokens(messages, max_tokens)
        while attempts < max_attempts:
            if rec.model_calls >= self.cfg.slo.max_model_calls:
                rec.notes.append("model-call budget used up")
                return None
            left_s = (deadline_ms - rec.ms()) / 1000
            if left_s <= 0.3:
                return None
            t_wait = rec.ms()
            # the governor is on for both policies, so the retry experiment
            # changes only what happens *after* a failure
            waited = await b.acquire(est, max_wait_s=max(0.0, left_s - 1.0))
            if waited is None:
                rec.notes.append(f"{step}: {model} budget would exceed deadline")
                return None
            if waited > 0.01:
                rec.span(f"{step}:budget_wait", "wait", t_wait, model=model)
            t_sem = rec.ms()
            async with self.gov.sem[model]:
                if rec.ms() - t_sem > 10:
                    rec.span(f"{step}:inflight_wait", "wait", t_sem, model=model)
                left_s = (deadline_ms - rec.ms()) / 1000
                if left_s <= 0.3:
                    return None
                t_call = rec.ms()
                rec.model_calls += 1
                res = await self.client.chat(model, messages, max_tokens=max_tokens,
                                             ttft_timeout=min(ttft_timeout, left_s),
                                             stall_timeout=3.0, total_timeout=left_s,
                                             json_mode=json_mode)
            b.settle(est, res)
            ti, to = rec.tokens.get(model, [0, 0])
            if res.status == "ok" or res.tokens_out:
                rec.tokens[model] = [ti + res.tokens_in, to + res.tokens_out]
            elif res.status == "timeout":
                # tokens were reserved and probably partly spent; count the prompt
                rec.tokens[model] = [ti + res.tokens_in, to]
            attempts += 1
            if res.status == "ok":
                tok = res.tokens_in + max(res.tokens_out, max_tokens // 2)
                if step in ("draft", "revise") and model == DRAFT_MODEL:
                    self.est_draft = 0.8 * self.est_draft + 0.2 * tok
                elif model == ROUTER_MODEL:
                    self.est_small = 0.8 * self.est_small + 0.2 * tok * (2 if step == "route" else 1)
                rec.span(step, "work", t_call, model=model, provider_ttft_ms=res.ttft_ms,
                         first_reasoning_ms=res.first_reasoning_ms,
                         server_queue_ms=res.server_queue_ms, tokens_out=res.tokens_out)
                return res
            rec.span(f"{step}:failed_{res.status}", "retry", t_call, model=model, http=res.http_status)
            if attempts >= max_attempts:
                break
            # --- retry policy ---
            if self.cfg.retry_policy == "naive":
                # the common client default: same model, fixed short delay,
                # ignore retry-after, retry timeouts too
                rec.retries += 1
                t_b = rec.ms()
                await asyncio.sleep(0.2)
                rec.span(f"{step}:backoff", "retry", t_b, model=model)
                continue
            if res.status == "rate_limited":
                ra = res.retry_after_s if res.retry_after_s is not None else 2.0
                if b.day_exhausted() or ra > (deadline_ms - rec.ms()) / 1000 - 1.5:
                    rec.notes.append(f"{step}: 429 retry-after {ra:.1f}s does not fit deadline")
                    return None
                rec.retries += 1
                t_b = rec.ms()
                await asyncio.sleep(ra + 0.1)
                rec.span(f"{step}:backoff", "retry", t_b, model=model)
                continue
            if res.status == "timeout":
                # slow-success guard fired. Retrying the same overloaded model is
                # what makes bursts worse; the caller falls back instead.
                return None
            # 5xx / connection error: one quick retry with jitter
            rec.retries += 1
            t_b = rec.ms()
            await asyncio.sleep(0.25 + 0.25 * (hash(rec.request_id) % 4) / 4)
            rec.span(f"{step}:backoff", "retry", t_b, model=model)
        return None

    # ---- level selection ---------------------------------------------------------
    def _pick_level(self, rec: Record) -> str:
        r = self.cfg.day_reserve_frac
        b70, b8 = self.gov[DRAFT_MODEL], self.gov[ROUTER_MODEL]
        left_ttft = (self.cfg.slo.ttft_deadline_ms - rec.ms()) / 1000
        w70 = b70.wait_needed(int(self.est_draft))
        w8 = b8.wait_needed(int(self.est_small))
        if not b70.day_exhausted(r) and w70 < left_ttft - 2.5:
            if w8 < 0.5 and not b8.day_exhausted(r) and self.cfg.llm_verify:
                return "L0"
            return "L1"
        if not b8.day_exhausted() and w8 < left_ttft - 2.0:
            return "L2"
        return "L3"

    # ---- the request -------------------------------------------------------------
    async def run(self, question: str, scenario: str = "", qid: str = "", sink=None,
                  request_id: str | None = None, out: dict | None = None) -> Record:
        rec = Record(request_id=request_id or uuid.uuid4().hex[:10], scenario=scenario,
                     source=self.client.source, question=question, qid=qid,
                     start_wall=time.time(), t0=time.monotonic())
        if out is not None:
            out["rec"] = rec        # so the caller still has it if we are cancelled
        sent = []

        def emit(piece: str):
            if rec.first_token_ms is None:
                rec.first_token_ms = rec.ms()
            sent.append(piece)
            if sink:
                sink(piece)

        self.queue_len += 1
        self.peak_queue = max(self.peak_queue, self.queue_len)
        got_worker, dequeued = False, False
        try:
            if self.queue_len > self.cfg.max_queue:
                self.queue_len -= 1
                dequeued = True
                rec.notes.append("queue full")
                await self._excerpts(rec, question, emit, reason="busy")
                return rec
            t_q = rec.ms()
            try:
                await asyncio.wait_for(self.workers.acquire(),
                                       timeout=max(0.1, (self.cfg.slo.ttft_deadline_ms - 2500) / 1000))
                got_worker = True
            except asyncio.TimeoutError:
                rec.notes.append("no worker before deadline")
            rec.queue_wait_ms = rec.ms() - t_q
            self.queue_len -= 1
            dequeued = True
            rec.span("queue", "wait", t_q)
            if not got_worker:
                await self._excerpts(rec, question, emit, reason="busy")
                return rec
            await self._work(rec, question, emit)
        except asyncio.CancelledError:
            rec.status = "timeout"
            rec.notes.append("user gave up")
            raise
        except Exception as e:  # never crash the whole run on one request
            rec.status = "error"
            rec.notes.append(f"exception: {e!r}")
        finally:
            if not dequeued:
                self.queue_len -= 1
            if got_worker:
                self.workers.release()
            rec.total_ms = rec.ms()
            rec.answer = "".join(sent)
        return rec

    async def _work(self, rec: Record, question: str, emit):
        slo = self.cfg.slo
        level = self.cfg.force_level or self._pick_level(rec)
        rec.level = level
        setting = RETRIEVAL_SETTINGS[self.cfg.retrieval if level in ("L0", "L1") else PRESSURE_RETRIEVAL]
        rec.retrieval_setting = setting.label()

        # 1. understand
        query, paper = question, None
        if level == "L0":
            t = rec.ms()
            msgs = [{"role": "system", "content": ROUTER_SYS}, {"role": "user", "content": question}]
            res = await self._call(rec, "route", ROUTER_MODEL, msgs, 250, deadline_ms=min(slo.ttft_deadline_ms, rec.ms() + 3500),
                                   ttft_timeout=3.0)
            if res:
                try:
                    j = json.loads(re.search(r"\{.*\}", res.text, re.S).group(0))
                    if j.get("paper") in PAPER_BY_KEY:
                        paper = j["paper"]
                    if j.get("query"):
                        query = f"{question} {j['query']}"
                    rec.notes.append(f"router: in_scope={j.get('in_scope')} paper={j.get('paper')}")
                    if j.get("in_scope") is False:
                        hits = retrieve(question, setting)
                        if coverage(question, hits) < 0.5:
                            rec.refused = True
                            await self._send(rec, emit, f"{REFUSAL_MARK}: this question is outside the three papers in the library.", hits, [])
                            rec.status = "ok"
                            return
                except (AttributeError, ValueError):
                    rec.notes.append("router: unparseable, using raw question")
            else:
                rec.notes.append("router skipped (failed or no budget)")

        # 2. retrieve  (+ 3. decide, with one re-retrieve loop)
        t = rec.ms()
        hits = retrieve(query, setting, paper)
        cov = coverage(question, hits)
        rec.span("retrieve", "work", t, coverage=round(cov, 2))
        if cov < 0.45 and query == question:
            t = rec.ms()
            # loop: widen the search (more candidates, ignore paper hint)
            wider = RETRIEVAL_SETTINGS["medium_rerank"]
            hits2 = retrieve(question, wider)
            if coverage(question, hits2) > cov:
                hits, cov = hits2[: setting.k], coverage(question, hits2)
            rec.span("re-retrieve", "work", t, coverage=round(cov, 2))
        if not hits:
            rec.refused = True
            await self._send(rec, emit, f"{REFUSAL_MARK}: nothing in the three papers matches this question.", [], [])
            rec.status = "ok"
            return

        if level == "L3":
            await self._excerpts(rec, question, emit, hits=hits, reason=self._limit_reason())
            return

        # 4. draft (+5/6 check, verify, one revise loop)
        drafters = [DRAFT_MODEL, FALLBACK_DRAFT_MODEL] if level in ("L0", "L1") else [FALLBACK_DRAFT_MODEL]
        if self.cfg.draft_models:
            drafters = list(self.cfg.draft_models)
        user = f"Question: {question}\n\nPassages:\n{_passages_block(hits)}"
        msgs = [{"role": "system", "content": DRAFT_SYS}, {"role": "user", "content": user}]
        passages = [_clean(h.chunk.text) for h in hits]
        answer, revised = None, False
        for model in drafters:
            if model == FALLBACK_DRAFT_MODEL and model != drafters[0]:
                # fell back: shrink the prompt so the small model's 6K TPM fits
                hits = hits[:RETRIEVAL_SETTINGS[PRESSURE_RETRIEVAL].k]
                passages = [_clean(h.chunk.text) for h in hits]
                user = f"Question: {question}\n\nPassages:\n{_passages_block(hits)}"
                msgs = [{"role": "system", "content": DRAFT_SYS}, {"role": "user", "content": user}]
                rec.notes.append("fell back to small drafter")
                rec.retries += 1
            ttft_guard = 4.0 if model == DRAFT_MODEL else 3.0
            res = await self._call(rec, "draft", model, msgs, self.cfg.draft_max_tokens,
                                   deadline_ms=slo.total_deadline_ms - 1500, ttft_timeout=ttft_guard)
            if not res:
                continue
            rec.drafter = model
            if rec.model_ttft_ms is None and res.ttft_ms is not None:
                rec.model_ttft_ms = (res.started - rec.t0) * 1000 + res.ttft_ms
            text = res.text.strip()
            chk = check_answer(text, passages, question, [h.chunk.section for h in hits])
            t = rec.ms()
            rec.span("check", "work", t, ok=chk.ok, problems=chk.problems, draft=text[:600])
            if not chk.ok and not revised and rec.ms() < slo.ttft_deadline_ms:
                revised = True
                fix = msgs + [{"role": "assistant", "content": text},
                              {"role": "user", "content": "Your answer failed checks: " + "; ".join(chk.problems)
                               + ". Rewrite it following the rules exactly."}]
                res2 = await self._call(rec, "revise", model, fix, self.cfg.draft_max_tokens,
                                        deadline_ms=slo.total_deadline_ms - 1500, ttft_timeout=ttft_guard)
                if res2:
                    text = res2.text.strip()
                    chk = check_answer(text, passages, question, [h.chunk.section for h in hits])
                    rec.span("check2", "work", rec.ms(), ok=chk.ok, problems=chk.problems, draft=text[:600])
            if not chk.ok:
                rec.notes.append(f"draft failed checks: {chk.problems}")
                break
            if level == "L0" and not chk.refusal and self.cfg.llm_verify:
                ids = sorted(set(i for i in cited_ids(text) if 1 <= i <= len(hits)))
                vuser = "Passages:\n" + "\n\n".join(f"[{i}] {_clean(hits[i-1].chunk.text)}" for i in ids) + f"\n\nAnswer:\n{text}"
                vres = await self._call(rec, "verify", VERIFY_MODEL,
                                        [{"role": "system", "content": VERIFY_SYS}, {"role": "user", "content": vuser}],
                                        200, deadline_ms=min(slo.total_deadline_ms - 1000, rec.ms() + 4500), ttft_timeout=3.0)
                if vres and vres.text.strip().upper().startswith("UNSUPPORTED"):
                    rec.notes.append(f"verifier: {vres.text.strip()[:160]}")
                    if revised or rec.ms() > slo.ttft_deadline_ms:
                        break          # no time for another loop: send excerpts, not a doubtful answer
                    revised = True
                    fix = msgs + [{"role": "assistant", "content": text},
                                  {"role": "user", "content": "A reviewer says: " + vres.text.strip()[:300]
                                   + ". Rewrite the answer so every claim is directly supported by a cited passage,"
                                   " or reply NOT IN PAPERS if they do not support an answer."}]
                    res2 = await self._call(rec, "revise", model, fix, self.cfg.draft_max_tokens,
                                            deadline_ms=slo.total_deadline_ms - 1500, ttft_timeout=ttft_guard)
                    if not res2:
                        break
                    text = res2.text.strip()
                    chk = check_answer(text, passages, question, [h.chunk.section for h in hits])
                    rec.span("check2", "work", rec.ms(), ok=chk.ok, problems=chk.problems, draft=text[:600])
                    if not chk.ok:
                        break
                elif not vres:
                    rec.notes.append("verifier unavailable; deterministic checks only")
            answer = text
            break

        if answer is None:
            # budget/deadline failure vs. an answer we could not verify
            failed_checks = any("failed checks" in n or n.startswith("verifier:") for n in rec.notes)
            await self._excerpts(rec, question, emit, hits=hits, reason="unverified" if failed_checks else self._limit_reason())
            return
        rec.refused = is_refusal(answer)
        await self._send(rec, emit, answer, hits, cited_ids(answer))
        rec.status = "ok"

    async def _send(self, rec: Record, emit, answer: str, hits: list[Hit], ids: list[int]):
        """Stream the verified answer to the person, a piece at a time."""
        t = rec.ms()
        if is_refusal(answer):
            # fixed text: the model's explanation after "NOT IN PAPERS:" is not checked by
            # anything, and once invented a fact ("Helium was pretrained on audio data")
            near = "; ".join(h.chunk.cite() for h in hits[:2])
            answer = "The papers in this library do not say." + (
                f" The closest passages I found: {near}." if near else "")
            ids = []
        src = _sources_block(hits, ids) if ids else ""
        full = answer + (f"\n\nSources:\n{src}" if src else "")
        if rec.level == "L2" or (rec.drafter and rec.drafter != DRAFT_MODEL and not self.cfg.draft_models):
            full += ("\n\n(Answered by our smaller model because the main model is at its usage limit right now. "
                     "It was checked against the cited passages, but it may be less complete.)")
        rec.cited_papers = sorted({hits[i - 1].chunk.paper for i in ids if 1 <= i <= len(hits)})
        words = full.split(" ")
        for i in range(0, len(words), 6):
            emit(" ".join(words[i:i + 6]) + " ")
            await asyncio.sleep(0)
        rec.span("send", "work", t)

    async def _excerpts(self, rec: Record, question: str, emit, hits=None, reason="limit"):
        """L3: no model. Honest about why, still cited, never wrong."""
        t = rec.ms()
        rec.level = "L3"
        hits = hits if hits is not None else retrieve(question, RETRIEVAL_SETTINGS[PRESSURE_RETRIEVAL])
        hits = hits[:3]
        when = self._recovery_hint()
        head = {"busy": "We're answering a lot of questions right now, so I can't write a full answer for you this minute.",
                "unverified": "I couldn't write an answer I could check against the papers, so instead of guessing,"
                              " here are the passages that look most relevant.",
                }.get(reason, "I can't write a full answer right now: the assistant has used its model budget.")
        if reason == "unverified":
            when = ""
        lines = [f"{head} {when} In the meantime, these passages from the papers look most relevant:"
                 if reason != "unverified" else head]
        for i, h in enumerate(hits, 1):
            snippet = " ".join(h.chunk.text.split()[:45])
            lines.append(f"[{i}] \"{snippet} ...\" — {h.chunk.cite()}")
        if not hits:
            lines.append("(No passage in the three papers matches this question; they may not cover it.)")
        rec.cited_papers = sorted({h.chunk.paper for h in hits})
        words = "\n".join(lines).split(" ")
        for i in range(0, len(words), 8):
            emit(" ".join(words[i:i + 8]) + " ")
            await asyncio.sleep(0)
        rec.refused = True
        # a check refusal is the system choosing caution, not a limit
        rec.status = "ok" if reason == "unverified" else "rate_limited"
        rec.notes.append(f"excerpts: {reason}")
        rec.span("excerpts", "work", t)

    def _limit_reason(self) -> str:
        """'limit' (daily budget gone) only when the small model's day is gone, since it is
        the last model that could answer; anything else is a per-minute squeeze ('busy')."""
        return "limit" if self.gov[FALLBACK_DRAFT_MODEL].day_exhausted() else "busy"

    def _recovery_hint(self) -> str:
        b70 = self.gov[DRAFT_MODEL]
        now = time.monotonic()
        if b70.day_exhausted():
            if b70.day_exhausted_until > now:
                mins = int((b70.day_exhausted_until - now) / 60) + 1
                return f"Full answers should be back in about {mins} min."
            return "Full answers come back when the daily budget resets (00:00 UTC, 05:30 IST)."
        w = max(b70.wait_needed(int(self.est_draft)), 0)
        return f"Ask again in about {max(5, int(w) + 5)} seconds for a full answer." if w < 300 else ""