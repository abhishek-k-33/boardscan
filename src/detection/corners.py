"""Phase 1c: pairwise H/V intersections -> 9x9 lattice -> outer corners.
Fallback: largest quadrilateral contour (approxPolyDP). Robust to occluded corners."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

import config


@dataclass
class CornersResult:
    ok: bool
    corners: np.ndarray | None = None  # 4x2 float32 in downscaled coords, TL/TR/BR/BL (unordered OK)
    method: str = ""                   # "lattice" or "contour"
    error: str = ""


def segment_intersection(h: tuple, v: tuple) -> tuple[float, float] | None:
    """Intersection of two infinite lines defined by segments. None if parallel."""
    x1, y1, x2, y2 = h
    x3, y3, x4, y4 = v
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def all_intersections(horizontal: list, vertical: list, shape) -> list:
    h, w = shape[:2]
    pts = []
    for hl in horizontal:
        for vl in vertical:
            p = segment_intersection(hl, vl)
            if p is None:
                continue
            # keep points inside image (+ small margin)
            if -0.1 * w < p[0] < 1.1 * w and -0.1 * h < p[1] < 1.1 * h:
                pts.append(p)
    return pts


def cluster_points(pts: list, tol: float) -> list:
    """Greedy clustering: average nearby points (lattice nodes seen by multiple line pairs)."""
    clusters: list[list] = []
    for p in pts:
        placed = False
        for c in clusters:
            cx = sum(q[0] for q in c) / len(c)
            cy = sum(q[1] for q in c) / len(c)
            if ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 < tol:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return [(sum(q[0] for q in c) / len(c), sum(q[1] for q in c) / len(c)) for c in clusters]


def outer_corners_from_lattice(nodes: list) -> np.ndarray | None:
    if len(nodes) < config.MIN_INTERSECTIONS_FOR_LATTICE:
        return None
    pts = np.array(nodes, dtype=np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, config.APPROX_POLY_EPSILON_RATIO * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    # fallback inside lattice path: min-area rect corners
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    return np.array(box, dtype=np.float32)


def largest_quad_contour(gray: np.ndarray) -> np.ndarray | None:
    # Primary: Canny edges -> dilate -> external contour. Works for checker + gray bg.
    edges = cv2.Canny(gray, config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = gray.shape[0] * gray.shape[1] * config.CONTOUR_MIN_AREA_RATIO
    quad = _best_quad(contours, area_min)
    if quad is not None:
        return quad
    # Secondary: Otsu threshold path (clean backgrounds)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = _best_quad(contours, area_min)
    if quad is not None:
        return quad
    # Last resort: biggest contour's min-area box
    all_cnts = contours
    if not all_cnts:
        return None
    biggest = max(all_cnts, key=cv2.contourArea)
    if cv2.contourArea(biggest) < area_min:
        return None
    return np.array(cv2.boxPoints(cv2.minAreaRect(biggest)), dtype=np.float32)


def _best_quad(contours, area_min: float) -> np.ndarray | None:
    best = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min or area <= best_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, config.APPROX_POLY_EPSILON_RATIO * peri, True)
        if len(approx) == 4:
            best, best_area = cnt, area
    if best is None:
        return None
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, config.APPROX_POLY_EPSILON_RATIO * peri, True)
    return approx.reshape(4, 2).astype(np.float32)


def find_corners(gray: np.ndarray, horizontal: list, vertical: list,
                 debug: bool = False, debug_dir: Path | None = None) -> CornersResult:
    """Compute BOTH candidates, keep the larger-area quad (true board border is
    the largest quad; lattice hull is often inset to interior grid lines)."""
    lattice_quad = None
    try:
        pts = all_intersections(horizontal, vertical, gray.shape)
        nodes = cluster_points(pts, config.INTERSECTION_CLUSTER_TOL)
        lattice_quad = outer_corners_from_lattice(nodes) if nodes else None
    except Exception:
        lattice_quad = None
    contour_quad = None
    try:
        contour_quad = largest_quad_contour(gray)
    except Exception:
        contour_quad = None
    if lattice_quad is None and contour_quad is None:
        return CornersResult(ok=False, error="no quadrilateral found (lattice + contour failed)")
    if lattice_quad is None:
        return CornersResult(ok=True, corners=contour_quad.astype(np.float32), method="contour")
    if contour_quad is None:
        return CornersResult(ok=True, corners=lattice_quad.astype(np.float32), method="lattice")
    # both found -> larger area wins (outer border)
    try:
        la = abs(float(cv2.contourArea(lattice_quad.astype(np.float32))))
        ca = abs(float(cv2.contourArea(contour_quad.astype(np.float32))))
    except Exception:
        return CornersResult(ok=True, corners=contour_quad.astype(np.float32), method="contour")
    if ca >= la:
        return CornersResult(ok=True, corners=contour_quad.astype(np.float32), method="contour")
    return CornersResult(ok=True, corners=lattice_quad.astype(np.float32), method="lattice")
