"""Retrieval tool: BM25 over chunks + an optional cheap lexical re-ranker.

No embedding model on purpose: it would be another hosted call with its own
rate limit (or a local model the stand-in cannot imitate). BM25 runs in
~1 ms here and costs zero tokens, which matters when tokens per minute is the
binding limit. The price is weaker recall on paraphrased questions; the
router step's query rewrite is how we pay that back.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .config import RETRIEVAL_SETTINGS, RetrievalSetting
from .corpus import Chunk, build_chunks

_STOP = set("""a an the of to in on for and or is are was were be been by with as at from that this these those it its
what which who whom how why when where does do did can could should would will may might than then there their they
we our you your i me my about into over under between also such using used use paper papers model models""".split())
_TOK = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOK.findall(text.lower()):
        if t in _STOP or (len(t) == 1 and not t.isdigit()):
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.docs = [tokenize(c.text + " " + c.section) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, n: int, paper: str | None = None) -> list[Hit]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                if paper and self.chunks[i].paper == paper:
                    s *= 1.3   # router said which paper; nudge, do not filter
                scores.append((s, i))
        scores.sort(reverse=True)
        return [Hit(self.chunks[i], s) for s, i in scores[:n]]


def _bigrams(toks):
    return set(zip(toks, toks[1:]))


def rerank(query: str, hits: list[Hit], k: int) -> list[Hit]:
    """Cheap, deterministic re-ranker. Rewards chunks that cover *all* the
    question's terms, keep its word order (bigrams), and contain the numbers
    the question mentions. BM25 alone loves long chunks that repeat one rare
    word (e.g. a references list that says 'Transformer' 30 times)."""
    q = tokenize(query)
    if not hits or not q:
        return hits[:k]
    qset, qbi = set(q), _bigrams(q)
    top = max(h.score for h in hits)
    out = []
    for h in hits:
        d = tokenize(h.chunk.text)
        dset = set(d)
        coverage = len(qset & dset) / len(qset)
        bigram = len(qbi & _bigrams(d)) / max(1, len(qbi))
        sec = len(qset & set(tokenize(h.chunk.section))) / len(qset)
        s = 0.45 * (h.score / top) + 0.35 * coverage + 0.12 * bigram + 0.08 * sec
        if h.chunk.is_reference:
            s *= 0.5
        out.append(Hit(h.chunk, s))
    out.sort(key=lambda h: -h.score)
    return out[:k]


@lru_cache(maxsize=8)
def _index(chunk_words: int, overlap: int, drop_refs: bool) -> BM25Index:
    chunks = build_chunks(chunk_words, overlap)
    if drop_refs:
        chunks = [c for c in chunks if not c.is_reference]
    return BM25Index(chunks)


def retrieve(query: str, setting: RetrievalSetting | str, paper: str | None = None) -> list[Hit]:
    if isinstance(setting, str):
        setting = RETRIEVAL_SETTINGS[setting]
    idx = _index(setting.chunk_words, setting.overlap_words, setting.drop_references)
    if setting.rerank:
        return rerank(query, idx.search(query, setting.candidates, paper), setting.k)
    return idx.search(query, setting.k, paper)


def coverage(query: str, hits: list[Hit]) -> float:
    """Share of the query's content words found in the retrieved text: the
    'decide' step uses it to tell 'not in the papers' from 'bad query'."""
    q = set(tokenize(query))
    if not q or not hits:
        return 0.0
    got = set()
    for h in hits:
        got |= set(tokenize(h.chunk.text))
    return len(q & got) / len(q)
