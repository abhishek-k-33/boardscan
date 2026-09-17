"""Phase 1b: Canny -> HoughLinesP -> angular clustering -> merge duplicates."""
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np

import config


@dataclass
class LinesResult:
    ok: bool
    horizontal: list = field(default_factory=list)  # each (x1,y1,x2,y2)
    vertical: list = field(default_factory=list)
    all_lines: list = field(default_factory=list)
    edges: np.ndarray | None = None
    error: str = ""


def detect_edges(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(
        gray,
        config.CANNY_LOW_THRESHOLD,
        config.CANNY_HIGH_THRESHOLD,
        apertureSize=config.CANNY_APERTURE_SIZE,
    )


def detect_hough(edges: np.ndarray) -> list:
    import math
    theta = math.radians(config.HOUGH_THETA_DEG)
    raw = cv2.HoughLinesP(
        edges,
        rho=config.HOUGH_RHO,
        theta=theta,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP,
    )
    if raw is None:
        return []
    # cv2 4.x returns (N,1,4), cv2 5.x returns (N,4) — handle both
    arr = np.asarray(raw).reshape(-1, 4)
    return [tuple(map(int, row)) for row in arr]


def segment_angle(x1, y1, x2, y2) -> float:
    """Angle in [0,180) degrees of segment."""
    import math
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return ang


def cluster_lines(lines: list) -> tuple[list, list]:
    """Split into two angular clusters: near-horizontal (0/180) and near-vertical (90).
    Tolerance from config. Ambiguous mid-angles go to nearest group."""
    h, v = [], []
    tol = config.ANGLE_CLUSTER_TOL_DEG
    for ln in lines:
        a = segment_angle(*ln)
        dist_h = min(a, 180.0 - a)          # distance to 0 deg
        dist_v = abs(a - 90.0)              # distance to 90 deg
        # widen acceptance: assign to nearer axis; tol only used to drop diagonals if extreme
        if dist_h <= dist_v:
            h.append(ln)
        else:
            v.append(ln)
    return h, v


def _line_rho_theta(x1, y1, x2, y2) -> tuple[float, float]:
    """Infinite-line (rho, theta-normal) representation for merging."""
    import math
    dx, dy = x2 - x1, y2 - y1
    # normal angle
    theta = math.atan2(dy, dx) + math.pi / 2.0
    theta = theta % math.pi
    rho = x1 * math.cos(theta) + y1 * math.sin(theta)
    if rho < 0:
        rho = -rho
        theta = (theta + math.pi) % math.pi
    return rho, theta


def merge_duplicates(lines: list) -> list:
    """Greedy merge of near-duplicate segments with similar (rho, theta) and close midpoints."""
    import math
    if not lines:
        return []
    reps = [(_line_rho_theta(*ln), ln) for ln in lines]
    rho_tol = config.MERGE_RHO_TOL
    theta_tol = math.radians(config.MERGE_THETA_TOL_DEG)
    mid_tol = config.MERGE_MIDPOINT_TOL
    kept: list = []
    kept_reps: list = []
    for (rho, theta), ln in reps:
        mx, my = (ln[0] + ln[2]) / 2.0, (ln[1] + ln[3]) / 2.0
        dup = False
        for i, (krho, ktheta) in enumerate(kept_reps):
            dtheta = min(abs(theta - ktheta), math.pi - abs(theta - ktheta))
            kln = kept[i]
            kmx, kmy = (kln[0] + kln[2]) / 2.0, (kln[1] + kln[3]) / 2.0
            if abs(rho - krho) < rho_tol and dtheta < theta_tol:
                if abs(mx - kmx) < 200 and abs(my - kmy) < 200 or ((mx - kmx) ** 2 + (my - kmy) ** 2) ** 0.5 < mid_tol * 10:
                    # extend kept segment to cover both (keep longest endpoints)
                    pts = [(kln[0], kln[1]), (kln[2], kln[3]), (ln[0], ln[1]), (ln[2], ln[3])]
                    # farthest pair
                    best, bestd = (kln[0], kln[1], kln[2], kln[3]), -1
                    for a in range(4):
                        for b in range(a + 1, 4):
                            d = (pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2
                            if d > bestd:
                                bestd = d
                                best = (pts[a][0], pts[a][1], pts[b][0], pts[b][1])
                    kept[i] = best
                    dup = True
                    break
        if not dup:
            kept.append(ln)
            kept_reps.append((rho, theta))
    return kept


def detect_lines(gray: np.ndarray, debug: bool = False, debug_dir: Path | None = None) -> LinesResult:
    try:
        edges = detect_edges(gray)
        raw = detect_hough(edges)
        if not raw:
            return LinesResult(ok=False, edges=edges, error="no Hough lines found")
        h, v = cluster_lines(raw)
        h, v = merge_duplicates(h), merge_duplicates(v)
        if debug:
            out = Path(debug_dir or config.DEBUG_DIR)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "04_edges.png"), edges)
        return LinesResult(ok=True, horizontal=h, vertical=v, all_lines=h + v, edges=edges)
    except Exception as e:
        return LinesResult(ok=False, error=f"line detection failed: {e}")
