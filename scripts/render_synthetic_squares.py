"""Bootstrap training data without a physical board: render stylised piece
silhouettes on square backgrounds. Real photos (Phase 0) replace this via
scripts/build_squares_dataset.py — the manifest format is identical.

Each class gets a distinct silhouette; white/black differ by fill colour.
Squares are grouped into pseudo-photos of 64 so the by-photo split (dataset.py)
works unchanged.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

SIZE = config.SQUARE_CROP_SIZE
WHITE_FILL, WHITE_EDGE = (242, 242, 242), (25, 25, 25)
BLACK_FILL, BLACK_EDGE = (28, 28, 28), (225, 225, 225)
LIGHT_SQ, DARK_SQ = (238, 222, 190), (148, 108, 78)


def _poly(img, pts, fill, edge):
    p = np.array(pts, np.int32)
    cv2.fillPoly(img, [p], fill)
    cv2.polylines(img, [p], True, edge, 4, cv2.LINE_AA)


def draw_piece(img, piece: str):
    """piece: FEN char. Draws centred silhouette standing on the square."""
    white = piece.isupper()
    fill, edge = (WHITE_FILL, WHITE_EDGE) if white else (BLACK_FILL, BLACK_EDGE)
    t = piece.upper()
    if t == "P":
        cv2.circle(img, (50, 34), 15, fill, -1, cv2.LINE_AA)
        cv2.circle(img, (50, 34), 15, edge, 4, cv2.LINE_AA)
        _poly(img, [(34, 48), (66, 48), (72, 86), (28, 86)], fill, edge)
    elif t == "N":
        _poly(img, [(34, 86), (34, 56), (44, 40), (60, 28), (56, 48),
                    (70, 54), (70, 86)], fill, edge)
        cv2.circle(img, (52, 44), 4, edge, -1, cv2.LINE_AA)
    elif t == "B":
        cv2.circle(img, (50, 16), 8, fill, -1, cv2.LINE_AA)
        cv2.circle(img, (50, 16), 8, edge, 3, cv2.LINE_AA)
        _poly(img, [(50, 26), (63, 46), (50, 66), (37, 46)], fill, edge)
        cv2.line(img, (50, 34), (50, 58), edge, 3, cv2.LINE_AA)
        _poly(img, [(40, 66), (60, 66), (68, 86), (32, 86)], fill, edge)
    elif t == "R":
        for x in (32, 44, 56):
            cv2.rectangle(img, (x, 32), (x + 8, 46), fill, -1)
            cv2.rectangle(img, (x, 32), (x + 8, 46), edge, 3)
        _poly(img, [(30, 46), (70, 46), (70, 86), (30, 86)], fill, edge)
        cv2.line(img, (30, 58), (70, 58), edge, 3, cv2.LINE_AA)
    elif t == "Q":
        for x in (30, 44, 58):
            cv2.circle(img, (x + 3, 24), 5, fill, -1, cv2.LINE_AA)
            cv2.circle(img, (x + 3, 24), 5, edge, 3, cv2.LINE_AA)
        _poly(img, [(30, 34), (37, 52), (46, 34), (53, 52), (62, 34),
                    (70, 56), (30, 56)], fill, edge)
        _poly(img, [(36, 56), (64, 56), (70, 86), (30, 86)], fill, edge)
    elif t == "K":
        cv2.line(img, (50, 10), (50, 30), edge, 6, cv2.LINE_AA)
        cv2.line(img, (41, 20), (59, 20), edge, 6, cv2.LINE_AA)
        _poly(img, [(34, 36), (66, 36), (62, 58), (38, 58)], fill, edge)
        _poly(img, [(38, 58), (62, 58), (70, 86), (30, 86)], fill, edge)


def render_square(label: str, rng: np.random.Generator, dark: bool) -> np.ndarray:
    base = np.array(DARK_SQ if dark else LIGHT_SQ, np.float32)
    img = np.ones((SIZE, SIZE, 3), np.float32) * base
    img += rng.normal(0, 6, img.shape)                      # wood grain
    gain = float(rng.uniform(0.8, 1.2))                     # lighting jitter
    img = np.clip(img * gain, 0, 255)
    canvas = img.astype(np.uint8)
    if label != "empty":
        dx, dy = int(rng.integers(-4, 5)), int(rng.integers(-3, 4))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        tmp = np.zeros_like(canvas)
        draw_piece(tmp, label)
        tmp = cv2.warpAffine(tmp, M, (SIZE, SIZE), borderValue=(0, 0, 0))
        m = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY) > 10
        canvas[m] = tmp[m]
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=config.SYNTH_SQUARES_PER_CLASS)
    ap.add_argument("--seed", type=int, default=config.SYNTH_SEED)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synth_squares")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out, imgdir = args.out, args.out / "img"
    imgdir.mkdir(parents=True, exist_ok=True)
    # Build a shuffled multiset (per_class copies of each class) and deal it
    # into 64-square pseudo-photos, so every photo holds a realistic MIX of
    # classes like a real board. Class-pure photos would let the by-photo
    # split hide whole classes from train or test and inflate accuracy.
    multiset = [c for c in config.CLASS_LABELS for _ in range(args.per_class)]
    rng.shuffle(multiset)
    rows = []
    for idx, label in enumerate(multiset):
        dark = bool(rng.integers(0, 2))
        img = render_square(label, rng, dark)
        name = f"sq_{idx:05d}_{label}.png"
        cv2.imwrite(str(imgdir / name), img)
        rows.append((f"data/synth_squares/img/{name}", label, f"synth_{idx // 64:03d}"))
    # shuffle rows (photo grouping preserved in photo_id column)
    rng.shuffle(rows)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "photo_id"])
        w.writerows(rows)
    print(f"wrote {len(rows)} squares, {idx // 64 + 1} pseudo-photos -> {out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
