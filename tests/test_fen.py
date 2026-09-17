"""Tests for Module 3a: grid -> FEN, orientation, validation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.squares.labels import fen_to_grid
from src.notation.fen import grid_to_fen, maybe_flip

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_startpos_roundtrip():
    res = grid_to_fen(fen_to_grid(STARTPOS))
    assert res.ok, res.error
    assert res.placement == STARTPOS
    assert res.fen == f"{STARTPOS} w KQkq - 0 1"
    assert not res.flipped and not res.ambiguous


def test_empty_board_warns_but_ok():
    res = grid_to_fen([["empty"] * 8 for _ in range(8)])
    assert res.ok and res.placement == "8/8/8/8/8/8/8/8"
    assert any("kings" in w for w in res.warnings)


def test_two_white_kings_is_error_not_crash():
    grid = fen_to_grid("K6K/8/8/8/8/8/8/k7")
    res = grid_to_fen(grid)
    assert not res.ok and "kings" in res.error


def test_orientation_flips_rotated_board():
    grid = fen_to_grid(STARTPOS)
    upside_down = [row[:] for row in reversed(grid)]  # white back rank on row 0
    fixed, flipped, ambiguous = maybe_flip(upside_down)
    assert flipped and not ambiguous
    assert fixed == grid


def test_orientation_tie_is_ambiguous():
    grid = [["empty"] * 8 for _ in range(8)]
    grid[0][0] = "K"
    grid[0][1] = "k"  # symmetric mass either way up -> tie
    _, flipped, ambiguous = maybe_flip(grid)
    assert ambiguous and not flipped


def test_bad_inputs_rejected():
    assert not grid_to_fen([["empty"] * 8] * 7).ok
    assert not grid_to_fen(fen_to_grid(STARTPOS), side_to_move="x").ok
