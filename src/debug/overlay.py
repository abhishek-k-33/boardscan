"""Debug visualisation: draw Hough lines + corners + 8x8 grid on original photo."""
import cv2
import numpy as np

import config


def draw_overlay(full_bgr: np.ndarray, scale: float, horizontal: list,
                 vertical: list, corners_small: np.ndarray) -> np.ndarray:
    vis = full_bgr.copy()
    inv = 1.0 / scale if scale not in (0, 1.0) else 1.0
    for (x1, y1, x2, y2) in horizontal:
        cv2.line(vis, (int(x1 * inv), int(y1 * inv)), (int(x2 * inv), int(y2 * inv)), (255, 0, 0), 2)
    for (x1, y1, x2, y2) in vertical:
        cv2.line(vis, (int(x1 * inv), int(y1 * inv)), (int(x2 * inv), int(y2 * inv)), (0, 255, 0), 2)
    if corners_small is not None:
        for (x, y) in np.array(corners_small).reshape(-1, 2):
            cv2.circle(vis, (int(x * inv), int(y * inv)), 8, (0, 0, 255), -1)
    return vis


def draw_grid(warped: np.ndarray) -> np.ndarray:
    vis = warped.copy()
    s = config.WARP_SIZE // config.NUM_SQUARES
    for i in range(config.NUM_SQUARES + 1):
        cv2.line(vis, (i * s, 0), (i * s, config.WARP_SIZE - 1), (0, 0, 255), 1)
        cv2.line(vis, (0, i * s), (config.WARP_SIZE - 1, i * s), (0, 0, 255), 1)
    return vis
