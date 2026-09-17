"""Generate course diagrams into docs/ with matplotlib (no external tools needed).
    python scripts/make_diagrams.py
Outputs: architecture.png, usecase.png, sequence.png, class-diagram.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BLUE, GREEN, AMBER, GRAY = "#DBEAFE", "#DCFCE7", "#FEF3C7", "#F3F4F6"


def box(ax, xy, w, h, text, fc=BLUE, fs=9):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=fc, ec="#111827")
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                 color="#111827", linewidth=1.2, shrinkA=2, shrinkB=4))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DOCS / name, dpi=150)
    plt.close(fig)
    print("wrote", DOCS / name)


def architecture():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("BoardScan — System Architecture", fontsize=13, weight="bold")
    box(ax, (0.2, 2.0), 1.5, 1.0, "Photo\n(RGB)")
    box(ax, (2.3, 2.0), 1.7, 1.0, "1. Detection\nCanny→Hough→\nhomography")
    box(ax, (4.6, 2.0), 1.7, 1.0, "2. Squares\nslice 8×8\n+ CNN(13)")
    box(ax, (6.9, 2.0), 1.7, 1.0, "3. Notation\nFEN + Stockfish")
    box(ax, (4.6, 0.4), 1.7, 0.9, "4. Debug layer\noverlays + dumps", fc=GRAY)
    box(ax, (4.6, 3.6), 1.7, 0.9, "config.py\nall thresholds", fc=AMBER)
    box(ax, (8.1, 2.0), 0.5, 1.0, "", fc="white")
    ax.text(8.35, 2.5, "FEN\n+ eval", fontsize=9, va="center")
    for a, b in [((1.7, 2.5), (2.3, 2.5)), ((4.0, 2.5), (4.6, 2.5)),
                 ((6.3, 2.5), (6.9, 2.5)), ((8.6, 2.5), (8.9, 2.5))]:
        arrow(ax, a, b)
    arrow(ax, (5.4, 2.0), (5.4, 1.3))
    arrow(ax, (5.4, 3.6), (5.4, 3.0))
    save(fig, "architecture.png")


def usecase():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("BoardScan — Use Case Diagram", fontsize=13, weight="bold")
    actors = [(1.0, 4.5, "Club\nplayer"), (1.0, 2.5, "Coach"), (1.0, 0.8, "Arbiter")]
    for x, y, name in actors:
        ax.plot([x, x], [y - 0.45, y + 0.15], color="#111827", lw=2)
        ax.add_patch(plt.Circle((x, y + 0.32), 0.17, fc="white", ec="#111827", lw=2))
        ax.plot([x - 0.3, x + 0.3], [y - 0.05, y - 0.05], color="#111827", lw=2)
        ax.plot([x, x - 0.25], [y - 0.45, y - 0.95], color="#111827", lw=2)
        ax.plot([x, x + 0.25], [y - 0.45, y - 0.95], color="#111827", lw=2)
        ax.text(x, y - 1.25, name, ha="center", fontsize=9)
    ax.add_patch(plt.Rectangle((2.6, 0.2), 6.8, 5.1, fc="none", ec="#111827", lw=1.2))
    ax.text(6.0, 5.0, "BoardScan system", fontsize=10, style="italic")
    uses = [(6.0, 4.2, "Upload board\nphoto"), (6.0, 3.2, "Review pipeline\nstages"),
            (6.0, 2.2, "Correct square\nlabels"), (6.0, 1.2, "Copy FEN\n+ eval")]
    for x, y, name in uses:
        ax.add_patch(Ellipse((x, y), 2.6, 0.8, fc=BLUE, ec="#111827"))
        ax.text(x, y, name, ha="center", va="center", fontsize=9)
    for _, y, _ in uses:
        for ax_, ay, _ in actors:
            ax.plot([ax_ + 0.35, 4.7], [ay - 0.3, y], color="#6B7280", lw=0.8)
    save(fig, "usecase.png")


def sequence():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("BoardScan — Sequence Diagram (upload → FEN)", fontsize=13, weight="bold")
    parts = ["User", "Streamlit\nUI", "Detection", "Classifier", "Notation/\nEngine"]
    xs = [1.0, 3.0, 5.0, 7.0, 9.0]
    for x, p in zip(xs, parts):
        box(ax, (x - 0.7, 5.4), 1.4, 0.7, p, fc=GREEN, fs=8)
        ax.plot([x, x], [0.4, 5.4], color="#111827", lw=1, ls="--")
    msgs = [(1.0, 3.0, 5.0, "upload photo"), (3.0, 5.0, 4.5, "detect_and_rectify()"),
            (5.0, 3.0, 4.1, "warped 800×800"), (3.0, 7.0, 3.7, "slice + predict_grid()"),
            (7.0, 3.0, 3.3, "8×8 labels"), (3.0, 3.0, 2.9, "show editable grid"),
            (1.0, 3.0, 2.5, "confirm / fix squares"), (3.0, 9.0, 2.1, "grid_to_fen + analyze()"),
            (9.0, 3.0, 1.7, "FEN + eval"), (3.0, 1.0, 1.3, "display + copy"),
            (1.0, 3.0, 0.9, "copy FEN")]
    for x1, x2, y, label in msgs:
        arrow(ax, (x1, y), (x2, y))
        ax.text((x1 + x2) / 2, y + 0.12, label, ha="center", fontsize=7)
    save(fig, "sequence.png")


def class_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("BoardScan — Class Diagram", fontsize=13, weight="bold")

    def cls(x, y, name, attrs, methods):
        h = 0.5 + 0.28 * (len(attrs) + len(methods))
        w = 2.9
        ax.add_patch(plt.Rectangle((x, y - h), w, h, fc="white", ec="#111827", lw=1.2))
        ax.text(x + w / 2, y - 0.25, name, ha="center", fontsize=9, weight="bold")
        ax.plot([x, x + w], [y - 0.5, y - 0.5], color="#111827", lw=1)
        t = y - 0.7
        for a in attrs:
            ax.text(x + 0.1, t, f"• {a}", fontsize=7, va="top")
            t -= 0.28
        ax.plot([x, x + w], [t - 0.05, t - 0.05], color="#111827", lw=1)
        t -= 0.3
        for m in methods:
            ax.text(x + 0.1, t, f"+ {m}", fontsize=7, va="top")
            t -= 0.28

    cls(0.3, 6.0, "PipelineResult", ["ok: bool", "warped: image", "corners: 4×2", "method: str"],
        ["detect_and_rectify()"])
    cls(3.6, 6.0, "PieceCNN", ["features: 3×Conv", "gap: AvgPool", "fc: 13"],
        ["forward(x)->logits"])
    cls(6.9, 6.0, "PieceClassifier", ["net: PieceCNN", "labels[13]"],
        ["predict(crop)", "predict_grid()"])
    cls(0.3, 3.1, "FenResult", ["fen: str", "flipped/ambiguous", "warnings[]"],
        ["grid_to_fen()", "maybe_flip()"])
    cls(3.6, 3.1, "EngineResult", ["score_cp / mate_in", "best_move, pv[]", "degraded: bool"],
        ["analyze(fen)"])
    cls(6.9, 3.1, "SquareDataset", ["rows: manifest", "augment cfg"],
        ["split_by_photo()", "__getitem__()"])
    arrow(ax, (3.6, 4.9), (3.2, 4.9))
    ax.text(3.4, 5.05, "uses", fontsize=7, ha="center")
    arrow(ax, (0.9, 3.1), (0.9, 2.75))
    ax.text(1.15, 2.9, "feeds", fontsize=7)
    save(fig, "class-diagram.png")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    architecture()
    usecase()
    sequence()
    class_diagram()
