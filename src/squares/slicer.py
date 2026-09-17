"""Phase 2a: deterministic 8x8 slicing of the rectified board.

Orientation assumption: warped row 0 = rank 8 (black back rank at top),
col 0 = file a. Board-flip handling lives in Phase 3 (fen.py).
"""
import cv2
import numpy as np

import config


def square_name(row: int, col: int) -> str:
    """row 0..7 top-to-bottom -> ranks 8..1; col 0..7 -> files a..h."""
    return f"{chr(ord('a') + col)}{8 - row}"


def slice_board(warped: np.ndarray,
                vertical_bias_px: int = config.SQUARE_VERTICAL_BIAS_PX
                ) -> list[tuple[str, np.ndarray]]:
    """Slice 800x800 board into 64 labelled (name, 100x100 BGR crop) pairs.

    The crop window is shifted upward by vertical_bias_px because pieces are
    taller than their square — without this the piece head (king/queen crown)
    is cut off. Window stays 100x100, clamped to the image (pure function).
    """
    h, w = warped.shape[:2]
    sq = config.SQUARE_CROP_SIZE
    out = []
    for r in range(config.NUM_SQUARES):
        for c in range(config.NUM_SQUARES):
            x0 = c * sq
            y0 = max(0, r * sq - vertical_bias_px)
            y0 = min(y0, h - sq)
            x0 = min(x0, w - sq)
            crop = warped[y0:y0 + sq, x0:x0 + sq].copy()
            if crop.shape[:2] != (sq, sq):  # tiny-input safety: pad, never crash
                crop = cv2.resize(crop, (sq, sq))
            out.append((square_name(r, c), crop))
    return out
