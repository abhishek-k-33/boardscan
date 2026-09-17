"""End-to-end smoke test (Modules 1-3 on synthetic data).

Composes a full 800x800 board from rendered piece squares, then runs the real
pipeline: slice (bias 0 to match classifier training) -> classify -> FEN.
Also covers engine degraded mode (no Stockfish binary in CI).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from render_synthetic_squares import render_square
from src.squares.slicer import slice_board
from src.squares.classifier import PieceClassifier
from src.squares.labels import fen_to_grid
from src.notation.fen import grid_to_fen
from src.notation.engine import analyze

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",  # startpos
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",  # open game
]


def compose_board(placement: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = fen_to_grid(placement)
    board = np.zeros((800, 800, 3), np.uint8)
    for r in range(8):
        for c in range(8):
            dark = (r + c) % 2 == 1
            sq = render_square(grid[r][c], rng, dark)
            board[r * 100:(r + 1) * 100, c * 100:(c + 1) * 100] = sq
    return board


def pipe(placement: str, seed: int) -> str:
    board = compose_board(placement, seed)
    clf = PieceClassifier()
    crops = slice_board(board, vertical_bias_px=0)
    grid = clf.predict_grid(crops)
    res = grid_to_fen(grid)
    assert res.ok, res.error
    return res.placement


def test_end_to_end_exact_fen():
    for i, placement in enumerate(POSITIONS):
        assert pipe(placement, seed=i) == placement


def test_engine_degraded_without_binary():
    res = analyze("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                  path="/nonexistent/stockfish")
    assert res.ok and res.degraded and res.score_cp is None
    assert "FEN-only" in res.error


def test_engine_live_if_binary_present():
    import shutil
    if not (shutil.which("stockfish") or Path(config.STOCKFISH_PATH).is_file()):
        return  # degraded mode already covered above
    res = analyze("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert res.ok and not res.degraded and res.best_move
