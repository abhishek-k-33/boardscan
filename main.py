"""BoardScan CLI — fully headless entry point (no GUI required).

Usage:
    python main.py --input data/raw/001.jpg --output debug/warped.png --debug
    python main.py --synthetic --output debug/synth_warped.png --debug
    python main.py --input data/raw/001.jpg --classify [--side-to-move b]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.detection.pipeline import detect_and_rectify_array
from scripts.synthetic_board import make_board, warped_photo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BoardScan: photo -> rectified 800x800 board")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="path to board photo")
    src.add_argument("--synthetic", action="store_true", help="use a generated test board")
    p.add_argument("--output", type=str, default="debug/warped.png", help="where to save warped board")
    p.add_argument("--debug", action="store_true", help="dump every stage image to debug/")
    p.add_argument("--seed", type=int, default=0, help="synthetic board seed")
    p.add_argument("--classify", action="store_true",
                   help="run slicer + CNN + FEN + engine on the warped board")
    p.add_argument("--side-to-move", type=str, default="w", choices=["w", "b"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic:
        photo, _ = warped_photo(make_board(), seed=args.seed, perturb=60)
    else:
        photo = cv2.imread(args.input)
        if photo is None:
            print(f"ERROR: cannot read image: {args.input}", file=sys.stderr)
            return 1
    t0 = time.time()
    res = detect_and_rectify_array(photo, debug=args.debug)
    dt = time.time() - t0
    if not res.ok:
        print(f"FAILED ({dt:.2f}s): {res.error}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), res.warped)
    print(f"OK ({dt:.2f}s) method={res.method} corners={res.corners.reshape(-1).round(1).tolist()}")
    print(f"warped board -> {out}")
    if args.classify:
        from src.squares.slicer import slice_board
        from src.squares.classifier import PieceClassifier
        from src.notation.fen import grid_to_fen
        from src.notation.engine import analyze
        try:
            grid = PieceClassifier().predict_grid(slice_board(res.warped))
        except Exception as e:
            print(f"CLASSIFY FAILED: {e}", file=sys.stderr)
            return 3
        fr = grid_to_fen(grid, side_to_move=args.side_to_move)
        if not fr.ok:
            print(f"FEN FAILED: {fr.error}", file=sys.stderr)
            return 4
        print(f"FEN: {fr.fen}"
              + (f" [flipped]" if fr.flipped else "")
              + (f" [ambiguous orientation — verify]" if fr.ambiguous else ""))
        for w in fr.warnings:
            print(f"warn: {w}")
        er = analyze(fr.fen)
        if er.degraded:
            print(f"engine: {er.error}")
        elif not er.ok:
            print(f"engine FAILED: {er.error}", file=sys.stderr)
            return 5
        else:
            mate = f" mate {er.mate_in}" if er.mate_in else ""
            print(f"eval: {er.score_cp}cp{mate} best={er.best_move} pv={' '.join(er.pv[:5])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
