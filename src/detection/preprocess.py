"""Phase 1a: grayscale -> CLAHE -> bilateral filter. All params from config."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

import config


@dataclass
class PreprocessResult:
    ok: bool
    gray: np.ndarray | None = None      # downscaled gray, CLAHE + denoised
    full_gray: np.ndarray | None = None  # full-res gray (for warp scale mapping)
    scale: float = 1.0                  # downscaled_dim / full_dim (to map corners back)
    orig_shape: tuple = (0, 0)
    error: str = ""


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    if not config.USE_CLAHE:
        return gray
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Bilateral (edge-preserving, keeps board edges, smooths wood grain)
    vs Gaussian (blurs edges). Toggle via config.USE_BILATERAL for ablation."""
    if config.USE_BILATERAL:
        return cv2.bilateralFilter(
            gray,
            d=config.BILATERAL_D,
            sigmaColor=config.BILATERAL_SIGMA_COLOR,
            sigmaSpace=config.BILATERAL_SIGMA_SPACE,
        )
    k = config.GAUSSIAN_KERNEL_SIZE
    return cv2.GaussianBlur(gray, k, config.GAUSSIAN_SIGMA)


def downscale(gray: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= config.DOWNSCALE_MAX_DIM:
        return gray, 1.0
    scale = config.DOWNSCALE_MAX_DIM / float(m)
    resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_image(bgr: np.ndarray, debug: bool = False, debug_dir: Path | None = None) -> PreprocessResult:
    if bgr is None or bgr.size == 0:
        return PreprocessResult(ok=False, error="empty input image")
    try:
        full_gray = to_grayscale(bgr)
        small, scale = downscale(full_gray)
        enhanced = apply_clahe(small)
        clean = denoise(enhanced)
        if debug:
            out = Path(debug_dir or config.DEBUG_DIR)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "01_gray.png"), small)
            cv2.imwrite(str(out / "02_clahe.png"), enhanced)
            cv2.imwrite(str(out / "03_denoised.png"), clean)
        return PreprocessResult(ok=True, gray=clean, full_gray=full_gray,
                                scale=scale, orig_shape=bgr.shape[:2])
    except Exception as e:  # typed failure, never throws to UI
        return PreprocessResult(ok=False, error=f"preprocess failed: {e}")
