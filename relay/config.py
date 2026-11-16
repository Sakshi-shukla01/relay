"""Central configuration.

Everything that is a *choice* lives here, with the reason next to it, so the
live review can change a number in one place and re-run.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = ROOT / "raw"
RESULTS = ROOT / "results"
for _d in (DATA, RAW, RESULTS):
    _d.mkdir(exist_ok=True)

# RELAY_REAL_BASE exists only so the probe/analysis code can be tested against a local server.
GROQ_BASE = os.environ.get("RELAY_REAL_BASE", "https://api.groq.com/openai/v1")
STANDIN_BASE = os.environ.get("STANDIN_BASE", "http://127.0.0.1:8088/openai/v1")

# --- Models -----------------------------------------------------------------
# Llama 3.1 8B / 3.3 70B were shut down for free-tier accounts on 16 Aug 2026;
# Groq's recommended replacements are the gpt-oss reasoning models. They think
# (hidden reasoning tokens) before the first visible token, and those
# reasoning tokens count against max_completion_tokens AND the token limits.
# Override with RELAY_SMALL_MODEL / RELAY_LARGE_MODEL if Groq changes again.
SMALL = os.environ.get("RELAY_SMALL_MODEL", "openai/gpt-oss-20b")
LARGE = os.environ.get("RELAY_LARGE_MODEL", "openai/gpt-oss-120b")

# Free-tier limits as *documented* (Groq rate-limit page, as read Nov 2026).
# Only the starting guess: the governor overwrites them with the
# x-ratelimit-* headers, and scripts/probe_real.py measures them.
DOCUMENTED_LIMITS = {
    LARGE: dict(rpm=30, rpd=1000, tpm=8000, tpd=200000),
    SMALL: dict(rpm=30, rpd=1000, tpm=8000, tpd=200000),
}
# Groq on-demand list prices, USD per 1M tokens (in, out), used only to price
# "cost per 1,000 answers" as if on the paid tier.
PRICES = {
    LARGE: (0.15, 0.60),
    SMALL: (0.075, 0.30),
}

ROUTER_MODEL = SMALL         # cheap: classify + rewrite query
DRAFT_MODEL = LARGE          # quality: the cited answer
FALLBACK_DRAFT_MODEL = SMALL
VERIFY_MODEL = SMALL
# reasoning effort per call: 'low' everywhere. Each extra level multiplies
# hidden tokens, and tokens-per-minute (8K) is the binding limit.
REASONING_EFFORT = os.environ.get("RELAY_REASONING", "low")


def learned_limits() -> dict:
    """Limits measured by probe_real.py override the documented ones."""
    p = RESULTS / "limits_learned.json"
    lim = {m: dict(v) for m, v in DOCUMENTED_LIMITS.items()}
    if p.exists():
        for m, v in json.loads(p.read_text(encoding="utf-8")).items():
            lim.setdefault(m, {}).update({k: x for k, x in v.items() if x})
    return lim


# --- Service-level targets ("what fast means") -------------------------------
@dataclass
class SLO:
    # Time to first *verified* answer token. People asking an engineering
    # question after an incident will wait a few seconds for a cited answer,
    # but not a spinner of unknown length.
    ttft_target_ms: int = 3000
    # Past this we stop trying for the best answer and degrade.
    ttft_deadline_ms: int = 8000
    # Hard cap on the whole request. Beyond it we send what we have or an
    # honest "busy" reply.
    total_deadline_ms: int = 20000
    # Max model calls a single request may make (stops loops).
    max_model_calls: int = 6


@dataclass
class RetrievalSetting:
    name: str
    chunk_words: int
    overlap_words: int
    k: int
    rerank: bool
    candidates: int = 0          # how many BM25 hits go into the reranker
    drop_references: bool = True

    def label(self) -> str:
        r = f"rerank{self.candidates}->{self.k}" if self.rerank else "norerank"
        return f"{self.name}:{self.chunk_words}w/k{self.k}/{r}"


RETRIEVAL_SETTINGS = {
    "small": RetrievalSetting("small", 120, 30, 8, False),
    "medium": RetrievalSetting("medium", 250, 50, 4, False),
    "medium_rerank": RetrievalSetting("medium_rerank", 250, 50, 4, True, candidates=16),
    "large": RetrievalSetting("large", 600, 100, 3, False),
    # used under pressure: same index as medium_rerank, fewer passages
    "lean": RetrievalSetting("lean", 250, 50, 2, True, candidates=16),
    # small pieces, fewer of them: the pressure version of 'small'
    "small_lean": RetrievalSetting("small_lean", 120, 30, 4, False),
    # naive baseline for the "how the library misleads" section
    "naive": RetrievalSetting("naive", 250, 50, 4, False, drop_references=False),
}
# measured: small (120w x 8) had the best recall offline (89.5 % vs 84.2 %)
# and the most correct answers in the real sweep (61.5 % vs ...). eval_main
# was run with medium_rerank before that result existed.
DEFAULT_RETRIEVAL = os.environ.get("RELAY_RETRIEVAL", "medium_rerank")
PRESSURE_RETRIEVAL = "lean"


@dataclass
class PipelineConfig:
    slo: SLO = field(default_factory=SLO)
    retrieval: str = DEFAULT_RETRIEVAL
    retry_policy: str = "budgeted"   # or "naive" (for the retry experiment)
    max_inflight_per_model: int = 8
    max_queue: int = 200
    # keep this share of each daily budget back for the small-model fallback path
    day_reserve_frac: float = 0.03
    # visible answer (~150) + hidden reasoning; too low => empty answer
    # probe: gpt-oss at effort=low used ~55-75 completion tokens in total on
    # short answers, so 600 leaves room while not over-reserving minute tokens
    draft_max_tokens: int = 600
    llm_verify: bool = True
    force_level: str | None = None          # experiments only
    draft_models: tuple | None = None       # experiments only (e.g. retrieval sweep on the small model)