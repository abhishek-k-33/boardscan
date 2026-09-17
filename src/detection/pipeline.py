"""Full Phase-1 pipeline: preprocess -> lines -> corners -> rectify. Pure + typed."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

import config
from src.detection.preprocess import preprocess_image
from src.detection.lines import detect_lines
from src.detection.corners import find_corners
from src.detection.rectify import rectify


@dataclass
class PipelineResult:
    ok: bool
    warped: np.ndarray | None = None
    corners: np.ndarray | None = None  # full-res ordered
    method: str = ""
    error: str = ""


def detect_and_rectify(image_path: str | Path, debug: bool = False,
                       debug_dir: Path | None = None) -> PipelineResult:
    out = Path(debug_dir or config.DEBUG_DIR)
    if debug:
        out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return PipelineResult(ok=False, error=f"cannot read image: {image_path}")
    return detect_and_rectify_array(bgr, debug=debug, debug_dir=out)


def detect_and_rectify_array(bgr: np.ndarray, debug: bool = False,
                             debug_dir: Path | None = None) -> PipelineResult:
    out = Path(debug_dir or config.DEBUG_DIR)
    pre = preprocess_image(bgr, debug=debug, debug_dir=out)
    if not pre.ok:
        return PipelineResult(ok=False, error=pre.error)
    lin = detect_lines(pre.gray, debug=debug, debug_dir=out)
    if not lin.ok:
        # still try contour fallback with empty line sets
        from src.detection.corners import find_corners as _fc
        cor = _fc(pre.gray, [], [], debug=debug, debug_dir=out)
    else:
        cor = find_corners(pre.gray, lin.horizontal, lin.vertical)
    if not cor.ok:
        return PipelineResult(ok=False, error=cor.error)
    rec = rectify(bgr, cor.corners, pre.scale, debug=debug, debug_dir=out)
    if not rec.ok:
        return PipelineResult(ok=False, error=rec.error)
    if debug:
        from src.debug.overlay import draw_overlay
        overlay = draw_overlay(bgr, pre.scale, lin.horizontal if lin.ok else [],
                               lin.vertical if lin.ok else [], cor.corners)
        cv2.imwrite(str(out / "08_overlay.png"), overlay)
    return PipelineResult(ok=True, warped=rec.warped, corners=rec.corners_fullres, method=cor.method)
