"""Builds a tiny synthetic stand-in for the three PDFs so the code can be
tested where arXiv is unreachable. NOT the papers. Results are never
reported from it (scenario rows produced under RELAY_FIXTURE=1 are deleted
by run_all.py before the real runs)."""
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "data" / "fixture_papers"
OUT.mkdir(parents=True, exist_ok=True)

DOCS = {
    "1706.03762v7": ["arXiv:1706.03762v7 [cs.CL] 2 Aug 2023", "Attention Is All You Need", "Abstract",
        "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task. On English-to-French our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs.",
        "5.1 Training Data and Batching", "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens.",
        "5.2 Hardware and Schedule", "We trained our models on one machine with 8 NVIDIA P100 GPUs. We trained the base models for a total of 100,000 steps or 12 hours. The big models were trained for 300,000 steps (3.5 days).",
        "5.3 Optimizer", "We used the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 10-9. We used warmup_steps = 4000.",
        "5.4 Regularization", "During training, we employed label smoothing of value 0.1. For the base model, we use a rate of Pdrop = 0.1.",
        "6.1 Machine Translation", "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0. We used beam search with a beam size of 4 and length penalty alpha = 0.6.",
        "6.3 English Constituency Parsing", "The 4-layer transformer trained on the Wall Street Journal portion only reached 91.3 F1 on section 23, and 92.7 in the semi-supervised setting.",
        "References", "[1] Jimmy Lei Ba. Layer normalization. [2] Dzmitry Bahdanau. Neural machine translation by jointly learning to align and translate. Transformer Transformer BLEU WMT translation."],
    "2410.00037v1": ["arXiv:2410.00037v1 [eess.AS] 17 Sep 2024", "Moshi: a speech-text foundation model for real-time dialogue", "Abstract",
        "The resulting model is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice.",
        "3.3 Audio Tokenization", "Mimi operates at a frame rate of 12.5 Hz. With Q = 8 codebooks of 2048 entries, Mimi reaches a bitrate of 1.1 kbps. We distill WavLM into the first level, a semantic VQ.",
        "3.2 The Helium Text Language Model", "Helium is a 7B parameter model pretrained on 2.1T tokens of public English data with a context of 4096 tokens.",
        "Table 1", "The Temporal Transformer has 32 layers and the Depth Transformer has 6 layers.",
        "4.2 Audio data", "We use the Fisher dataset, about 2000 hours of phone conversations with each speaker on a separate channel, to fine-tune full-duplex interaction.",
        "References", "WaveNet: a generative model for raw audio. SoundStream. EnCodec. AudioLM."],
    "2507.07955v2": ["arXiv:2507.07955v2 [cs.LG] 15 Jul 2025", "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", "Abstract",
        "An H-Net with one stage of hierarchy operating at the byte level outperforms a strong Transformer over BPE tokens.",
        "2.2 Dynamic Chunking", "The routing module predicts boundaries from the cosine similarity between adjacent encoder representations. The encoder and decoder networks use Mamba-2 layers.",
        "2.2.3 Ratio Loss", "We use alpha = 0.03 in all experiments.",
        "3 Experiments", "We train on a 100B-token subset of FineWeb-Edu. On DNA, H-Net shows 3.6x better data efficiency than isotropic models.",
        "References", "Byte Latent Transformer. MambaByte. SpaceByte. BPE."],
}

for name, lines in DOCS.items():
    c = canvas.Canvas(str(OUT / f"{name}.pdf"), pagesize=letter)
    y = 750
    for i, line in enumerate(lines):
        words, buf = line.split(), ""
        for w in words:
            if len(buf) + len(w) > 95:
                c.drawString(50, y, buf); y -= 14; buf = ""
            buf += w + " "
        c.drawString(50, y, buf.strip()); y -= 20
        if y < 100:
            c.showPage(); y = 750
    c.save()
print("fixture written to", OUT)
