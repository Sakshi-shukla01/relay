"""The library: fetch the three papers at exact versions, extract text per
page, find section headings, and cut into citable pieces.

Every chunk carries (paper, section, page range) so every answer can say
where it came from.
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import DATA

PAPERS = [
    dict(key="attention", arxiv="1706.03762", version="v7",
         title="Attention Is All You Need", short="Vaswani et al. 2017"),
    dict(key="moshi", arxiv="2410.00037", version="v1",
         title="Moshi: a speech-text foundation model for real-time dialogue",
         short="Défossez et al. 2024"),
    dict(key="hnet", arxiv="2507.07955", version="v2",
         title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
         short="Hwang et al. 2025"),
]
PAPER_BY_KEY = {p["key"]: p for p in PAPERS}

import os

# RELAY_FIXTURE=1 points everything at a small synthetic corpus used only to
# test the code where arXiv is unreachable. Never used for reported results.
_FIX = os.environ.get("RELAY_FIXTURE") == "1"
PDF_DIR = DATA / ("fixture_papers" if _FIX else "papers")
PAGES_JSON = DATA / ("fixture_pages.json" if _FIX else "pages.json")


def fetch_papers(force: bool = False) -> None:
    """Download the exact versions and check the version stamp in the text."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for p in PAPERS:
        dst = PDF_DIR / f"{p['arxiv']}{p['version']}.pdf"
        if dst.exists() and not force:
            continue
        url = f"https://arxiv.org/pdf/{p['arxiv']}{p['version']}"
        print(f"fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "relay-assignment/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dst.write_bytes(r.read())


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")
    # re-join words hyphenated across line ends: "frame-\nwork" -> "framework"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # the Moshi PDF's kerning splits a capital from the rest of its word:
    # "T okenization", "W atermarking", "V oice". 'A' and 'I' are real words,
    # so they are left alone.
    text = re.sub(r"\b([B-HJ-Z]) (?=[a-z]{2,})", r"\1", text)
    return text


def extract_pages() -> dict:
    """{paper_key: [page_text, ...]} using pypdf. Cached in data/pages.json."""
    from pypdf import PdfReader

    out = {}
    for p in PAPERS:
        pdf = PDF_DIR / f"{p['arxiv']}{p['version']}.pdf"
        reader = PdfReader(str(pdf))
        pages = [_clean(pg.extract_text() or "") for pg in reader.pages]
        stamp = f"{p['arxiv']}{p['version']}"
        joined = " ".join(pages[:2]).replace(" ", "").replace("\n", "")
        p["version_verified"] = stamp in joined
        if not p["version_verified"]:
            print(f"WARNING: could not find version stamp {stamp} in {pdf.name}")
        out[p["key"]] = pages
    PAGES_JSON.write_text(json.dumps(out), encoding="utf-8")
    if not _FIX:
        (DATA / "papers_meta.json").write_text(json.dumps(PAPERS, indent=1), encoding="utf-8")
    return out


def load_pages() -> dict:
    if not PAGES_JSON.exists():
        fetch_papers()
        return extract_pages()
    return json.loads(PAGES_JSON.read_text(encoding="utf-8"))


# --- sections ---------------------------------------------------------------
_HEAD = re.compile(r"^(?P<num>(?:[1-9]|[A-H])(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][A-Za-z0-9 ,:;()\-–'/&]{2,70})$")
_UNNUMBERED = re.compile(r"^(Abstract|References|Bibliography|Acknowledg(e)?ments?|Appendix.*|Attention Visualizations)$", re.I)


def _is_heading(line: str):
    line = line.strip()
    if len(line) > 80 or len(line) < 4:
        return None
    m = _HEAD.match(line)
    if m:
        title = m.group("title").strip()
        # reject table rows / equations: too many digits or ends with a number
        if sum(c.isdigit() for c in title) > 3 or re.search(r"\d$", title):
            return None
        if len(title.split()) > 10:
            return None
        return f"§{m.group('num')} {title}"
    if _UNNUMBERED.match(line):
        return line
    return None


@dataclass
class Chunk:
    id: str
    paper: str
    section: str
    page_start: int   # 1-based, as printed in the PDF viewer
    page_end: int
    text: str
    is_reference: bool

    def cite(self) -> str:
        p = PAPER_BY_KEY[self.paper]
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        return f"{p['title']} (arXiv:{p['arxiv']}{p['version']}), {self.section}, {pages}"


def _words_with_meta(pages: list[str]):
    """Yield (word, page_no, section, in_refs) for one paper."""
    section, in_refs = "front matter", False
    for pno, text in enumerate(pages, start=1):
        for line in text.split("\n"):
            h = _is_heading(line)
            if h:
                section = h
                low = h.lower()
                if low.startswith(("references", "bibliography")):
                    in_refs = True
                elif h.startswith("§") and not re.match(r"§[A-H]\b", h):
                    in_refs = False
                elif low.startswith(("appendix", "attention visualizations")) or re.match(r"§[A-H]\b", h):
                    in_refs = False
            for w in line.split():
                yield w, pno, section, in_refs


def build_chunks(chunk_words: int, overlap_words: int, pages_by_paper: dict | None = None) -> list[Chunk]:
    pages_by_paper = pages_by_paper or load_pages()
    chunks: list[Chunk] = []
    step = max(1, chunk_words - overlap_words)
    for key, pages in pages_by_paper.items():
        # group words by (section, in_refs) runs so a chunk never spans two sections
        runs, cur, cur_key = [], [], None
        for w, pno, sec, refs in _words_with_meta(pages):
            k = (sec, refs)
            if k != cur_key and cur:
                runs.append((cur_key, cur))
                cur = []
            cur_key = k
            cur.append((w, pno))
        if cur:
            runs.append((cur_key, cur))
        n = 0
        for (sec, refs), words in runs:
            for start in range(0, max(1, len(words) - overlap_words), step):
                piece = words[start:start + chunk_words]
                if len(piece) < 12:
                    continue
                chunks.append(Chunk(
                    id=f"{key}-{n}", paper=key, section=sec,
                    page_start=piece[0][1], page_end=piece[-1][1],
                    text=" ".join(w for w, _ in piece), is_reference=refs))
                n += 1
    return chunks


def paper_text(key: str) -> str:
    return "\n".join(load_pages()[key])


if __name__ == "__main__":
    fetch_papers()
    pages = extract_pages()
    for k, v in pages.items():
        print(k, len(v), "pages", sum(len(p.split()) for p in v), "words")
    ch = build_chunks(250, 50, pages)
    secs = sorted({(c.paper, c.section) for c in ch})
    print(len(ch), "chunks;", len(secs), "sections")
    for s in secs:
        print("  ", s)
