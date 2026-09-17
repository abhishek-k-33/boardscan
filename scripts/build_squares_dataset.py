"""Phase 0 -> 2 bridge: slice every data/raw photo, auto-label squares from the
hand-typed ground-truth FEN. 60 photos x 64 squares ~= 3,800 labelled squares
for free. Run after capturing real photos:
    python scripts/build_squares_dataset.py
"""
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from src.detection.pipeline import detect_and_rectify_array
from src.squares.slicer import slice_board
from src.squares.labels import fen_to_grid


def main() -> int:
    raw = config.DATA_RAW_DIR
    out = ROOT / "data" / "squares"
    photos = sorted(raw.glob("*.jpg")) + sorted(raw.glob("*.png"))
    if not photos:
        print("no photos in data/raw — capture Phase-0 data first (synthetic stand-in: "
              "python scripts/render_synthetic_squares.py)")
        return 1
    manifest = []
    ok, fail = 0, 0
    for photo in photos:
        fen_file = photo.with_suffix(".fen")
        if not fen_file.exists():
            print(f"SKIP {photo.name}: no {fen_file.name} sidecar")
            continue
        grid = fen_to_grid(fen_file.read_text().strip().splitlines()[0])
        res = detect_and_rectify_array(cv2.imread(str(photo)))
        if not res.ok:
            print(f"FAIL {photo.name}: {res.error}")
            fail += 1
            continue
        for (sq, crop), label_row in zip(slice_board(res.warped),
                                         [c for row in grid for c in row]):
            d = out / label_row
            d.mkdir(parents=True, exist_ok=True)
            name = f"{photo.stem}_{sq}.png"
            cv2.imwrite(str(d / name), crop)
            manifest.append((f"data/squares/{label_row}/{name}", label_row, photo.stem))
        ok += 1
    with open(config.SQUARES_MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "photo_id"])
        w.writerows(manifest)
    print(f"photos ok={ok} failed={fail}, squares={len(manifest)} -> {config.SQUARES_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
