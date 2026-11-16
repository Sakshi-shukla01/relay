"""Deterministic answer checks. These run on every answer at every level,
including when we are shedding load: they cost ~0 ms and 0 tokens, and they
catch the embarrassing class of error (a number the papers never said, a
citation to a passage that does not exist)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE = re.compile(r"\[(\d{1,2})(?:\s*,\s*(\d{1,2}))*\]")
# a number may be glued to a unit ('2.1T tokens', '100B', '12.5Hz'): no
# lookahead on letters. Bug found on the real papers: the paper said '2.1T',
# the model said '2.1 trillion', and the check called 2.1 invented.
NUM = re.compile(r"(?<![\d.])\d+(?:[.,]\d+)*")
REFUSAL_MARK = "NOT IN PAPERS"


def cited_ids(text: str) -> list[int]:
    ids = []
    for m in re.finditer(r"\[([\d,\s]+)\]", text):
        for x in m.group(1).split(","):
            x = x.strip()
            if x.isdigit():
                ids.append(int(x))
    return ids


def _nums(text: str) -> set[str]:
    text = re.sub(r"\[[\d,\s]+\]", " ", text)
    text = text.replace("\u202f", "").replace("\u00a0", " ")   # '4\u202f000' -> '4000'
    out = set()
    for n in NUM.findall(text):
        n = n.replace(",", "")
        if n.endswith(".0"):
            n = n[:-2]
        out.add(n)
    return out


def is_refusal(text: str) -> bool:
    t = text.strip().upper()
    return t.startswith(REFUSAL_MARK) or t.startswith("THE PAPERS DO NOT SAY") or "NOT IN PAPERS" in t[:80]


@dataclass
class Check:
    ok: bool
    problems: list[str] = field(default_factory=list)
    refusal: bool = False


def check_answer(answer: str, passages: list[str], question: str, labels: list[str] | None = None) -> Check:
    if is_refusal(answer):
        return Check(True, refusal=True)
    ids = cited_ids(answer)
    probs = []
    if not ids:
        probs.append("no citations")
    bad = sorted({i for i in ids if i < 1 or i > len(passages)})
    if bad:
        probs.append(f"cites passages that do not exist: {bad}")
    valid = [i for i in ids if 1 <= i <= len(passages)]
    support = " ".join(passages[i - 1] for i in set(valid))
    # section numbers the model may repeat ('§5.4') come from the passage labels
    label_txt = " ".join(labels[i - 1] for i in set(valid)) if labels else ""
    support_nums = _nums(support) | _nums(question) | _nums(label_txt)
    # numbers 1-3 are usually list markers / 'two papers'; ignore them
    missing = sorted(n for n in _nums(answer) if n not in support_nums and not (n.isdigit() and int(n) <= 3))
    if missing:
        probs.append(f"numbers not found in cited passages: {missing[:6]}")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(s.split()) > 4]
    if sentences:
        cited_share = sum(1 for s in sentences if CITE.search(s)) / len(sentences)
        # allow a citation at the end of a short paragraph to cover it
        if cited_share < 0.5 and len(sentences) > 2:
            probs.append(f"only {cited_share:.0%} of sentences carry a citation")
    return Check(not probs, probs)
