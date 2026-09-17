"""Phase-0 acceptance helper: every data/raw/*.fen must parse under python-chess."""
import sys
from pathlib import Path
import chess

raw = Path(__file__).resolve().parent.parent / "data" / "raw"
fens = sorted(raw.glob("*.fen"))
if not fens:
    print("no .fen files found — capture Phase-0 data first")
    sys.exit(1)
bad = 0
for f in fens:
    try:
        chess.Board(f.read_text().strip().splitlines()[0].strip())
    except Exception as e:
        print(f"INVALID {f.name}: {e}")
        bad += 1
print(f"{len(fens)-bad}/{len(fens)} FENs valid")
sys.exit(1 if bad else 0)
