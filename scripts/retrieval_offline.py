"""Zero-token retrieval checks: recall@k per setting, prompt size, and the
three ways this library misleads a naive retriever.

  python scripts/retrieval_offline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.config import DATA, RESULTS, RETRIEVAL_SETTINGS  # noqa: E402
from relay.grading import has, norm  # noqa: E402
from relay.retrieval import retrieve  # noqa: E402

QUESTIONS = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))

MISLEAD = [
    ("cross-paper vocabulary", "What BLEU score does H-Net get on WMT 2014 English-German translation?",
     "H-Net has no translation results; 'BLEU' and 'WMT' only occur in the Transformer paper, so lexical search returns "
     "the Transformer's numbers and a model happily attributes them to H-Net."),
    ("references and related-work lists", "Which paper introduced WaveNet and what does Moshi say about it?",
     "Bibliography chunks are dense with model names; BM25 ranks them first and the answer cites a reference entry, not a claim."),
    ("the paper disagrees with itself", "What BLEU score does the big Transformer get on WMT 2014 English-to-French?",
     "v7 says 41.8 in the abstract and Table 2 but 41.0 in Section 6.1; whichever chunk wins decides the answer."),
    ("tables lose their structure in PDF text", "What is the maximum path length for a restricted self-attention layer?",
     "Table 1 comes out of the PDF as a run of symbols; the right row is retrieved but its columns are scrambled."),
]


# Questions about *other* work. Their names fill the bibliographies, so a
# retriever that keeps reference lists finds the list, not a claim.
REF_PROBES = [
    "Who proposed the Adam optimizer?", "Which paper introduced byte-pair encoding?",
    "What is WaveNet?", "Which work introduced layer normalization?", "Who introduced the Mamba architecture?",
    "What is SoundStream?", "Which paper proposed the RQ-Transformer?", "What is MambaByte?",
    "What is the Byte Latent Transformer?", "Which paper introduced dropout?", "What is EnCodec?",
    "What is SpaceByte?", "Who wrote the paper on residual connections?", "What is AudioLM?",
]


def ref_pollution(setting, k=4):
    out = []
    for q in REF_PROBES:
        hits = retrieve(q, RETRIEVAL_SETTINGS[setting])[:k]
        out.append((q, sum(h.chunk.is_reference for h in hits)))
    return out


def recall(setting):
    s = RETRIEVAL_SETTINGS[setting]
    hit, toks, n, ms = 0, 0, 0, 0.0
    for q in QUESTIONS:
        if not q["answerable"]:
            continue
        t = time.perf_counter()
        hits = retrieve(q["question"], s)
        ms += (time.perf_counter() - t) * 1000
        n += 1
        toks += sum(len(h.chunk.text) for h in hits) // 4
        text = norm(" ".join(h.chunk.text for h in hits if h.chunk.paper == q["paper"]))
        if any(all(has(text, p) for p in v) for v in q["expect"]):
            hit += 1
    return dict(setting=setting, label=s.label(), recall_at_k=round(100 * hit / n, 1), n=n,
                passage_tokens=round(toks / n), retrieve_ms=round(ms / n, 2))


def show(q, setting):
    out = []
    for h in retrieve(q, RETRIEVAL_SETTINGS[setting])[:3]:
        out.append(f"   - {h.chunk.paper:9s} {h.chunk.section[:40]:40s} p.{h.chunk.page_start:<3} ref={int(h.chunk.is_reference)} "
                   f"| {h.chunk.text[:90]!r}")
    return "\n".join(out)


if __name__ == "__main__":
    res = [recall(s) for s in ("small", "small_lean", "medium", "medium_rerank", "large", "lean", "naive")]
    lines = ["| setting | recall@k (answerable, n) | passage tokens / question | retrieve ms |", "|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['label']} | {r['recall_at_k']}% ({r['n']}) | {r['passage_tokens']} | {r['retrieve_ms']} |")
    rep = ["## Offline retrieval (0 tokens)", *lines, "", "## How this library misleads a naive retriever"]
    naive_ref, ours_ref = ref_pollution("naive"), ref_pollution("medium_rerank")
    n_naive = sum(1 for _, c in naive_ref if c)
    n_ours = sum(1 for _, c in ours_ref if c)
    worst = max(naive_ref, key=lambda x: x[1])
    rep += [f"Reference-list pollution: for {len(REF_PROBES)} questions about other work, a references chunk was in the "
            f"top-4 for {n_naive} with the naive retriever and {n_ours} with ours "
            f"(references dropped from the index). Worst case: '{worst[0]}' ({worst[1]} of 4 hits were bibliography).",
            "naive top-3 for the worst case:", show(worst[0], "naive"), "ours:", show(worst[0], "medium_rerank"), ""]
    MISLEAD[:] = [m for m in MISLEAD if not m[0].startswith("references")]
    for name, q, why in MISLEAD:
        rep += [f"### {name}", f"Q: {q}", f"Why: {why}", "naive (BM25, references kept):", show(q, "naive"),
                "ours (references dropped, re-ranked):", show(q, "medium_rerank"), ""]
    txt = "\n".join(rep)
    (RESULTS / "retrieval_offline.md").write_text(txt, encoding="utf-8")
    print(txt)