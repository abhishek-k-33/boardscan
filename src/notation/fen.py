"""Phase 3a: 8x8 label grid -> validated FEN.

Orientation: the slicer assumes warped row 0 = rank 8. infer_orientation()
scores that assumption (white mass on row 7 + black mass on row 0) against the
flipped one; ties are reported ambiguous and left unflipped for the user to
correct (manual flip in the UI).
"""
from dataclasses import dataclass, field

import chess

from src.squares.labels import grid_to_placement


@dataclass
class FenResult:
    ok: bool
    fen: str = ""
    placement: str = ""
    flipped: bool = False          # True if orientation inference flipped the grid
    ambiguous: bool = False        # True if orientation was a tie (user should check)
    warnings: list = field(default_factory=list)  # non-fatal, e.g. missing kings
    error: str = ""


def _is_white(cell: str) -> bool:
    return cell != "empty" and cell.isupper()


def _is_black(cell: str) -> bool:
    return cell != "empty" and cell.islower()


def orientation_score(grid: list[list[str]]) -> int:
    """Plausibility of 'row 0 = rank 8': white on row 7, black on row 0."""
    return (sum(1 for c in grid[7] if _is_white(c))
            + sum(1 for c in grid[0] if _is_black(c)))


def maybe_flip(grid: list[list[str]]) -> tuple[list[list[str]], bool, bool]:
    """Return (grid, flipped, ambiguous) under the orientation heuristic."""
    straight = orientation_score(grid)
    flipped_grid = [row[:] for row in reversed(grid)]
    flipped = orientation_score(flipped_grid)
    if flipped > straight:
        return flipped_grid, True, False
    if flipped == straight:
        return grid, False, True
    return grid, False, False


def _king_errors(grid: list[list[str]]) -> tuple[str, list]:
    """Impossible king counts are errors; missing kings (e.g. empty-board
    calibration photos) are warnings only."""
    wk = sum(c == "K" for row in grid for c in row)
    bk = sum(c == "k" for row in grid for c in row)
    if wk > 1 or bk > 1:
        return f"illegal position: {wk} white kings, {bk} black kings", []
    warns = []
    if wk == 0 or bk == 0:
        warns.append(f"missing kings (white={wk}, black={bk}) — FEN still emitted")
    return "", warns


def grid_to_fen(grid: list[list[str]], side_to_move: str = "w",
                auto_orient: bool = True) -> FenResult:
    try:
        if len(grid) != 8 or any(len(r) != 8 for r in grid):
            return FenResult(ok=False, error="grid must be 8x8")
        if side_to_move not in ("w", "b"):
            return FenResult(ok=False, error="side_to_move must be 'w' or 'b'")
        g, flipped, ambiguous = maybe_flip(grid) if auto_orient else (grid, False, False)
        err, warns = _king_errors(g)
        if err:
            return FenResult(ok=False, error=err)
        placement = grid_to_placement(g)
        # Castling / en-passant / clocks are unknowable from one photo.
        fen = f"{placement} {side_to_move} KQkq - 0 1"
        try:
            board = chess.Board(fen)  # syntax validation; never throws out
        except ValueError as e:
            return FenResult(ok=False, error=f"python-chess rejected FEN: {e}")
        if not board.is_valid():
            warns.append("python-chess flags position as not fully legal")
        return FenResult(ok=True, fen=fen, placement=placement,
                         flipped=flipped, ambiguous=ambiguous, warnings=warns)
    except Exception as e:
        return FenResult(ok=False, error=f"FEN assembly failed: {e}")
