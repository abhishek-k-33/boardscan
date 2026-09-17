# BoardScan — Chess Position Digitiser (Photo → FEN → Evaluation)

## Overview

BoardScan takes a single photograph of a physical chessboard from an arbitrary
angle and outputs the position in Forsyth–Edwards Notation (FEN), plus an engine
evaluation. It is a Computer Vision course project: board localisation uses
**classical CV only** (grayscale → CLAHE → bilateral filter → Canny →
probabilistic Hough → line clustering → corner estimation → homography). A small
CNN is used only for the 64-square piece-classification step.

## Features

- **Module 1 — Board Detection & Rectification**: arbitrary-perspective photo →
  800×800 top-down warped board (implemented)
- **Module 2 — Square Segmentation & Piece Classification**: rectified board →
  8×8 label grid, 13 classes (planned)
- **Module 3 — FEN Assembly & Engine Analysis**: label grid → FEN string,
  centipawn eval + best move via Stockfish, graceful degraded mode without it
  (planned)
- **Module 4 — Debug/Visualisation Layer**: every stage dumps annotated
  intermediates to `debug/` via `--debug` (implemented)
- Synthetic board generator for testing without a physical board

## Tech Stack

- python 3.11, opencv-python 4.10.*, numpy 1.26.*, torch 2.3.* (classifier only),
  python-chess 1.10.*, streamlit 1.35.* (demo UI), pytest 8.*
- Stockfish binary (optional — app emits FEN-only mode if absent)

## Install

```bash
git clone https://github.com/<your-username>/boardscan.git
cd boardscan
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Rectify one board photo (headless, no GUI required):

```bash
python main.py --input data/raw/001.jpg --output debug/warped.png --debug
```

Generate 15 synthetic empty-board photos for testing:

```bash
python scripts/synthetic_board.py
```

Validate hand-typed ground-truth FEN sidecars (`data/raw/NNN.fen`):

```bash
python scripts/validate_fen.py
```

All CV parameters (Canny thresholds, Hough votes, kernel sizes, warp size) live
in `config.py` — no magic numbers in the pipeline code.

## Testing

```bash
pytest tests/ -v
```

- `test_rectify.py` — Phase-1 acceptance: ≥90% of empty-board photos rectify so
  inner grid lines land within 10px of the expected 100px spacing.
- `test_fen.py`, `test_pipeline.py` — planned with Modules 2–3.

## Screenshots

Stage dumps from `python main.py --input <photo> --debug` land in `debug/`:

| File | Stage |
|---|---|
| `01_gray.png` | downscaled grayscale |
| `02_clahe.png` | contrast normalised (CLAHE) |
| `03_denoised.png` | edge-preserving denoise (bilateral) |
| `04_edges.png` | Canny edge map |
| `07_warped.png` | rectified 800×800 board |
| `08_overlay.png` | Hough lines + corners on original |

## Project Structure

```
boardscan/
├── README.md  statement.md  requirements.txt  config.py  main.py
├── src/
│   ├── detection/  preprocess.py lines.py corners.py rectify.py pipeline.py
│   ├── squares/    slicer.py classifier.py        # Module 2 (planned)
│   ├── model/      architecture.py train.py dataset.py  # Module 2 (planned)
│   ├── notation/   fen.py engine.py               # Module 3 (planned)
│   └── debug/      overlay.py
├── app/            streamlit_app.py               # Module 4 UI (planned)
├── tests/  scripts/  data/raw  data/squares  docs/  debug/
```

Retrain the classifier any time (regenerates weights + confusion matrix):

```bash
python scripts/render_synthetic_squares.py --per-class 300
python src/model/train.py --epochs 20
```

## Status / Roadmap

- [x] Phase 1 — detection & rectification (classical CV, tested: 15/15 synthetic boards)
- [x] Phase 2 — slicer + 13-class CNN (100% per-square accuracy on 585 held-out
  synthetic squares from unseen pseudo-photos; confusion matrix in
  `docs/confusion_matrix.png`; `models/piece_cnn.pt` is 391KB, CPU-only).
  Note: synthetic silhouettes are clean by design — re-measure on real Phase-0
  photos via `python scripts/build_squares_dataset.py` + retrain.
- [x] Phase 3 — FEN assembly + Stockfish wrapper (orientation inference with
  user-flip fallback, illegal-position errors via king counts, degraded
  FEN-only mode without binary; `main.py --classify` end-to-end; e2e exact-FEN
  smoke test on synthetic positions).
  Note: `--classify` on bare synthetic boards underperforms (train/serve
  background skew) — retraining on real sliced squares closes it.
- [ ] Phase 0 — 60 real photo/FEN pairs (synthetic stand-in provided)
- [ ] Phase 2 — slicer + 13-class CNN (≥95% per-square accuracy)
- [ ] Phase 3 — FEN assembly + Stockfish wrapper (≥8/10 exact FEN)
- [ ] Phase 4 — Streamlit UI, UML docs, report figures
