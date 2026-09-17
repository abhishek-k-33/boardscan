"""Preprocessing ablation for the report: rectification success + corner error
under 4 settings (CLAHE on/off x bilateral/gaussian) on 15 synthetic boards.
Usage: python scripts/ablation.py
"""
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from scripts.synthetic_board import make_board, warped_photo
from src.detection.pipeline import detect_and_rectify_array

sys.path.insert(0, str(ROOT / "tests"))
from test_rectify import grid_spacing_error


def corner_error(det: np.ndarray, gt: np.ndarray) -> float:
    """Mean corner distance under best of 24 point assignments."""
    best = float("inf")
    for perm in itertools.permutations(range(4)):
        d = np.linalg.norm(det - gt[list(perm)], axis=1).mean()
        best = min(best, d)
    return float(best)


def run_setting(use_clahe: bool, use_bilateral: bool, n: int = 15):
    config.USE_CLAHE = use_clahe
    config.USE_BILATERAL = use_bilateral
    board = make_board()
    errs, oks = [], 0
    for i in range(n):
        photo, gt = warped_photo(board, seed=i, perturb=60)
        res = detect_and_rectify_array(photo)
        if not res.ok:
            errs.append(float("nan"))
            continue
        errs.append(corner_error(res.corners, gt))
        if grid_spacing_error(res.warped) <= config.GRID_TOLERANCE_PX:
            oks += 1
    errs = np.array(errs)
    return oks / n, float(np.nanmean(errs)), float(np.nanpercentile(errs, 95))


if __name__ == "__main__":
    print(f"{'setting':<28} {'success':>8} {'mean_px':>8} {'p95_px':>8}")
    for clahe, bilat in [(True, True), (False, True), (True, False), (False, False)]:
        rate, mean, p95 = run_setting(clahe, bilat)
        tag = f"CLAHE={'on' if clahe else 'off'} denoise={'bilat' if bilat else 'gauss'}"
        print(f"{tag:<28} {rate:>7.0%} {mean:>8.1f} {p95:>8.1f}")
    config.USE_CLAHE, config.USE_BILATERAL = True, True  # restore defaults
