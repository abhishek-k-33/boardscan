# statement.md — BoardScan

## Problem Statement

Chess players analysing an over-the-board game must manually re-enter the
position into an engine, which is slow and error-prone. BoardScan takes a single
photograph of a physical chessboard from an arbitrary angle and outputs the
position in Forsyth–Edwards Notation (FEN), along with an engine evaluation.

## Scope

- Input: one RGB photo of a full physical chessboard (0–45° tilt, varied
  lighting and board orientation).
- Output: FEN string of the position, engine centipawn score + best move, and
  debug overlays of each pipeline stage.
- In scope: classical-CV board localisation, per-square piece classification,
  FEN validation, graceful no-engine mode, editable result grid.
- Out of scope: video/live tracking, partial-board photos, chess-clock or
  scoresheet OCR.

## Target Users

- Club players digitising their over-the-board games.
- Coaches reviewing student games.
- Tournament arbiters recording positions.

## High-Level Features

1. Board Detection & Rectification — photo → 800×800 top-down board.
2. Square Segmentation & Piece Classification — board → 8×8 label grid.
3. FEN Assembly & Engine Analysis — labels → validated FEN + eval/best move.
4. Debug/Visualisation Layer — annotated stage images + manual grid correction.
