"""Tests for Module 2 pieces that need no trained weights: slicer, labels, split."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.squares.slicer import slice_board, square_name
from src.squares.labels import fen_to_grid, grid_to_placement, CLASS_TO_IDX


def test_slice_names_shapes_determinism():
    board = np.zeros((800, 800, 3), np.uint8)
    out = slice_board(board)
    assert len(out) == 64
    names = [n for n, _ in out]
    assert names[0] == "a8" and names[7] == "h8" and names[56] == "a1" and names[63] == "h1"
    assert len(set(names)) == 64
    for _, crop in out:
        assert crop.shape == (100, 100, 3)
    out2 = slice_board(board)  # deterministic
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(out, out2))


def test_vertical_bias_shifts_window_up():
    img = np.zeros((800, 800, 3), np.uint8)
    img[100:200, :] = 255  # white band exactly on row-1 square
    no_bias = dict(slice_board(img, vertical_bias_px=0))
    biased = dict(slice_board(img, vertical_bias_px=25))
    # row-1 square (a7) without bias == rows 100..200 -> all white
    assert float(no_bias["a7"].mean()) == 255.0
    # with bias the window moves up 25px -> includes black band above
    assert float(biased["a7"].mean()) < 255.0
    # top row clamps at image edge, stays valid
    assert biased["a8"].shape == (100, 100, 3)


def test_fen_labels_startpos():
    grid = fen_to_grid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    assert grid[0] == list("rnbqkbnr")      # rank 8
    assert grid[7] == list("RNBQKBNR")      # rank 1
    assert grid[4] == ["empty"] * 8         # middle empty
    assert grid_to_placement(grid) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert len(CLASS_TO_IDX) == 13 and CLASS_TO_IDX["empty"] == 0


def test_split_by_photo_no_leakage(tmp_path):
    from src.model.dataset import split_by_photo
    photos = [f"p{i:02d}" for i in range(20)]
    rows = [(f"s{q}.png", "empty", p) for p in photos for q in range(64)]
    tr, va, te = split_by_photo(rows, seed=0)
    ptr = {r[2] for r in tr}
    pva = {r[2] for r in va}
    pte = {r[2] for r in te}
    assert not (ptr & pva) and not (ptr & pte) and not (pva & pte)  # no shared photo
    assert len(tr) + len(va) + len(te) == len(rows)
    assert len(tr) > len(va) and len(tr) > len(te)


def test_cnn_forward_shape():
    torch = __import__("torch")
    from src.model.architecture import PieceCNN, count_parameters
    net = PieceCNN()
    out = net(torch.zeros(2, 3, config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE))
    assert out.shape == (2, config.MODEL_NUM_CLASSES)
    assert count_parameters(net) * 4 < 5 * 1024 * 1024  # NFR: model under 5MB
