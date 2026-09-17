"""Phase 1d: order corners -> getPerspectiveTransform -> warpPerspective 800x800."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

import config


@dataclass
class RectifyResult:
    ok: bool
    warped: np.ndarray | None = None  # WARP_SIZE x WARP_SIZE
    matrix: np.ndarray | None = None
    corners_fullres: np.ndarray | None = None  # 4x2 in original image coords
    error: str = ""


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL. Angle-around-centroid (robust to tilt),
    then rotate so top-left is first."""
    p = np.array(pts, dtype=np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    order = np.argsort(ang)  # CCW starting from +x axis
    # angles: ~-135=TL, ~-45=TR, ~45=BR, ~135=BL -> reorder to TL,TR,BR,BL
    ccw = p[order]
    # find TL (min x+y) index in ccw and rotate
    sums = ccw.sum(axis=1)
    start = int(np.argmin(sums))
    ccw = np.roll(ccw, -start, axis=0)
    # ensure clockwise TL->TR->BR->BL (centroid cross-product sign check)
    v0 = ccw[1] - ccw[0]
    v1 = ccw[2] - ccw[1]
    if v0[0] * v1[1] - v0[1] * v1[0] < 0:  # CCW -> reverse (keep TL fixed)
        ccw = np.array([ccw[0], ccw[3], ccw[2], ccw[1]])
    return ccw.astype(np.float32)


def rectify(full_bgr: np.ndarray, corners_small: np.ndarray, scale: float,
            debug: bool = False, debug_dir: Path | None = None) -> RectifyResult:
    try:
        ordered_small = order_corners(corners_small)
        if scale != 0 and scale != 1.0:
            ordered_full = ordered_small / float(scale)
        else:
            ordered_full = ordered_small
        dst = np.array([
            [0, 0],
            [config.WARP_SIZE - 1, 0],
            [config.WARP_SIZE - 1, config.WARP_SIZE - 1],
            [0, config.WARP_SIZE - 1],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered_full.astype(np.float32), dst)
        warped = cv2.warpPerspective(full_bgr, M, (config.WARP_SIZE, config.WARP_SIZE))
        if debug:
            out = Path(debug_dir or config.DEBUG_DIR)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "07_warped.png"), warped)
        return RectifyResult(ok=True, warped=warped, matrix=M, corners_fullres=ordered_full)
    except Exception as e:
        return RectifyResult(ok=False, error=f"rectify failed: {e}")
