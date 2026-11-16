"""Draw results/latency_cdf.png from results/results.csv.

Two panels (real runs | stand-in load scenarios). Each curve shows the share of ALL
requests in a scenario that got a checked model reply (L0/L1/L2, status ok) within
t milliseconds. Excerpt-only replies and people who gave up never count, so the
height where a curve ends = the share of people who got a real reply.

    python scripts/plot_cdf.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "results.csv"
OUT = ROOT / "results" / "latency_cdf.png"

PANELS = [
    ("Real Groq", [("real_steady", "steady (3.8/min)"), ("real_burst", "burst (12 at once)")]),
    ("Stand-in", [("steady", "steady"), ("burst", "burst"), ("sustained_overload", "overload"),
                  ("rpm_halved", "RPM halved"), ("daily_exhausted", "daily budget gone")]),
]
# validated categorical order (fixed, never cycled); dashes are the secondary encoding
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
GOALS = [(3000, "goal p50 3 s"), (8000, "goal p95 8 s")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"


def load():
    with CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def curve(rows):
    n = len(rows)
    got = sorted(float(r["first_token_ms"]) for r in rows
                 if r["status"] == "ok" and r.get("level") in ("L0", "L1", "L2") and r["first_token_ms"])
    xs, ys = [], []
    for i, x in enumerate(got, 1):
        xs += [x, x]
        ys += [(i - 1) / n, i / n]
    return xs, ys, len(got), n


def main():
    rows = load()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, dpi=150)
    for ax, (title, scenarios) in zip(axes, PANELS):
        for i, (sc, label) in enumerate(scenarios):
            sub = [r for r in rows if r["scenario"] == sc]
            if not sub:
                continue
            xs, ys, got, n = curve(sub)
            if not xs:
                continue
            ax.plot(xs, ys, color=COLORS[i], linestyle=DASHES[i], linewidth=2,
                    label=f"{label}: {got}/{n} replied")
            # direct label at the end of each line
            ax.annotate(f"{label} {100 * got / n:.0f}%", (xs[-1], ys[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9))
        for x, text in GOALS:
            ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
            ax.text(x * 1.04, 0.02, text, ha="left", va="bottom", fontsize=7.5, color=MUTED,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_xscale("log")
        ax.set_xlim(100, 30000)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
        ax.set_xlabel("time to first token of a checked reply (ms, log scale)")
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    axes[0].set_ylabel("share of all requests that got a checked reply")
    fig.suptitle("Share of people who got a checked reply within t "
                 "(excerpt-only replies and people who gave up never count)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())