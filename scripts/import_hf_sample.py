"""Import honi05/chess-positions-cv sample boards into data/raw/NNN.jpg + NNN.fen.

Source: https://huggingface.co/datasets/honi05/chess-positions-cv (synthetic
400x400 renders, FEN placement in the filename). Used as Phase-0 bootstrap
data; disclosed in README/report. Real photos can replace these 1:1 later.
Usage: python scripts/import_hf_sample.py --zip /tmp/chessdl/sample.zip --count 75
"""
import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import chess


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--count", type=int, default=75)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.zip)
    imgs = sorted(n for n in z.namelist() if n.startswith("images/") and n.endswith(".jpeg"))
    done = 0
    for i, name in enumerate(imgs):
        if done >= args.count:
            break
        placement = Path(name).stem.replace("-", "/")  # ranks are '-' in filenames
        fen = f"{placement} w KQkq - 0 1"
        try:
            chess.Board(fen)
        except ValueError:
            continue  # skip unparseable (e.g. bad pawn rows)
        data = z.read(name)
        (args.out / f"{done:03d}.jpg").write_bytes(data)
        (args.out / f"{done:03d}.fen").write_text(fen + "\n")
        done += 1
    print(f"imported {done} image/FEN pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
