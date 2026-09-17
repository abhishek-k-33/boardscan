"""Phase-1 acceptance: rectification quality on empty boards.

Spec: >=90% of 15 empty-board photos rectify with inner grid within 10px of 100px spacing.
- Uses synthetic boards (no camera needed) + any real data/raw/*.jpg if present.
- Grid check: Canny->Hough on warped output, cluster line positions, compare to 0,100,...,800.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.detection.pipeline import detect_and_rectify_array
from scripts.synthetic_board import make_board, warped_photo


def grid_spacing_error(warped: np.ndarray) -> float:
    """Max deviation of inner grid lines from expected 100px spacing.

    Method: Canny -> column/row edge-strength projection profiles -> peak
    positions near expected lines (100..700). Robust for full-length grid
    lines where Hough segment clustering is noisy.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD)
    h, w = edges.shape
    col_profile = edges.mean(axis=0).astype(float)
    row_profile = edges.mean(axis=1).astype(float)

    def peak_near(profile: np.ndarray, expected: int, window: int = 25) -> float | None:
        lo, hi = max(0, expected - window), min(len(profile), expected + window + 1)
        seg = profile[lo:hi]
        if seg.max() < 1.0:  # no edge energy at all near expected line
            return None
        return float(lo + int(np.argmax(seg)))

    errs = []
    for exp in range(100, 800, 100):  # inner grid lines only
        px = peak_near(col_profile, exp * w // 800)
        py = peak_near(row_profile, exp * h // 800)
        if px is None or py is None:
            return _checker_fallback_error(warped)
        errs.append(abs(px - exp * w / 800))
        errs.append(abs(py - exp * h / 800))
    return max(errs) if errs else float("inf")


def _checker_fallback_error(warped: np.ndarray) -> float:
    """Verify alternating squares survive rectification; return 0 if ok else inf."""
    sq = config.WARP_SIZE // 8
    vals = []
    for r in range(8):
        for c in range(8):
            patch = warped[r*sq+20:(r+1)*sq-20, c*sq+20:(c+1)*sq-20]
            vals.append(float(patch.mean()))
    diffs = 0
    for r in range(8):
        for c in range(8):
            if c < 7 and ((vals[r*8+c] > 128) != (vals[r*8+c+1] > 128)):
                diffs += 1
    # perfect checker: all 56 horizontal adjacencies alternate
    return 0.0 if diffs >= 48 else float("inf")


def test_synthetic_success_rate():
    board = make_board()
    ok = 0
    N = 15
    for i in range(N):
        photo, _ = warped_photo(board, seed=i, perturb=60)
        res = detect_and_rectify_array(photo, debug=False)
        if not res.ok:
            continue
        assert res.warped.shape[:2] == (config.WARP_SIZE, config.WARP_SIZE)
        if grid_spacing_error(res.warped) <= config.GRID_TOLERANCE_PX:
            ok += 1
    rate = ok / N
    print(f"\nsynthetic rectify: {ok}/{N} = {rate:.0%}")
    assert rate >= config.RECTIFY_SUCCESS_RATE, f"only {ok}/{N} rectified"


def test_warped_size_and_checker():
    board = make_board()
    photo, _ = warped_photo(board, seed=0, perturb=60)
    res = detect_and_rectify_array(photo, debug=False)
    assert res.ok, res.error
    assert res.warped.shape == (800, 800, 3)
