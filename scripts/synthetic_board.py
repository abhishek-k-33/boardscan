"""Synthetic empty chessboard generator — unblocks Phase-1 dev before real photos exist."""
import cv2
import numpy as np
from pathlib import Path


def make_board(size: int = 800, squares: int = 8) -> np.ndarray:
    sq = size // squares
    board = np.zeros((size, size, 3), dtype=np.uint8)
    light, dark = (240, 240, 240), (60, 60, 60)
    for r in range(squares):
        for c in range(squares):
            color = light if (r + c) % 2 == 0 else dark
            board[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = color
    return board


def warped_photo(board: np.ndarray, out_size=(1000, 800), perturb: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    margin = 60
    W, H = out_size
    dst = np.float32([
        [margin + rng.integers(-perturb, perturb), margin + rng.integers(-perturb, perturb)],
        [W - margin + rng.integers(-perturb, perturb), margin + rng.integers(-perturb, perturb)],
        [W - margin + rng.integers(-perturb, perturb), H - margin + rng.integers(-perturb, perturb)],
        [margin + rng.integers(-perturb, perturb), H - margin + rng.integers(-perturb, perturb)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    # textured background so contour fallback has something to find
    bg = np.full((H, W, 3), 120, dtype=np.uint8)
    warped = cv2.warpPerspective(board, M, (W, H), borderValue=(120, 120, 120))
    mask = cv2.warpPerspective(np.ones((h, w), np.uint8) * 255, M, (W, H))
    photo = np.where(mask[..., None] > 0, warped, bg)
    return photo, dst  # dst = ground-truth corners in photo coords


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    board = make_board()
    for i in range(15):
        photo, corners = warped_photo(board, seed=i, perturb=80 + (i % 3) * 30)
        cv2.imwrite(str(out / f"synth_empty_{i:03d}.jpg"), photo)
        np.savetxt(str(out / f"synth_empty_{i:03d}.corners.txt"), corners, fmt="%.1f")
    print("wrote 15 synthetic empty-board photos to", out)
