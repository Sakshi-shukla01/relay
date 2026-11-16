"""Deterministic grader, so reviewers can recompute every grade.

correct            answerable, contains every key fact of one accepted variant,
                   and cites the paper the fact is in
wrong              answerable but missing the fact / citing the wrong paper,
                   or unanswerable and the assistant answered anyway
refused_correctly  unanswerable and the assistant said the papers do not say
refused_wrongly    answerable and the assistant said the papers do not say
                   (an excerpts-only reply counts as a refusal)

Automatic grading of free text is imperfect; data/grade_overrides.json lets a
human overrule a grade, and every override is listed in the write-up.
"""
from __future__ import annotations

import json
import re

from .config import DATA


def norm(s: str) -> str:
    s = s.lower().replace("×", "x").replace("−", "-").replace("\u2011", "-")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    # thousands written with a (thin/non-breaking) space: '37 000' -> '37000'
    s = re.sub(r"(?<=\d)[ \u202f\u00a0\u2009](?=\d{3}\b)", "", s)
    # '100 K steps' / '100k' -> '100k'
    s = re.sub(r"(\d)\s?k\b", r"\1k", s)
    return re.sub(r"\s+", " ", s)


def load_overrides() -> dict:
    p = DATA / "grade_overrides.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def has(body: str, phrase: str) -> bool:
    p = norm(phrase)
    if re.fullmatch(r"[\d.]+", p):      # numbers: whole-number match, so '6' does not match '16'
        return re.search(rf"(?<![\d.]){re.escape(p)}(?![\d])", body) is not None
    return p in body


def grade(q: dict, answer: str, refused: bool, level: str, cited_papers: list[str]) -> str:
    body = norm(re.sub(r"\[[\d,\s]+\]", " ", answer.split("Sources:")[0]))
    is_refusal = refused or level == "L3"
    if not q["answerable"]:
        return "refused_correctly" if is_refusal else "wrong"
    if is_refusal:
        return "refused_wrongly"
    for variant in q["expect"]:
        if all(has(body, p) for p in variant):
            return "correct" if q["paper"] in cited_papers else "wrong"
    return "wrong"