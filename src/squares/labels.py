"""Phase 2b: FEN placement <-> 8x8 label grid. Shared by dataset builder (Phase 2)
and FEN assembly (Phase 3) so auto-labelling and inference use one mapping."""
import config

CLASS_TO_IDX = {name: i for i, name in enumerate(config.CLASS_LABELS)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}


def fen_to_grid(placement: str) -> list[list[str]]:
    """'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR' -> 8x8 of class labels,
    row 0 = rank 8. Raises ValueError on malformed placement."""
    rows = placement.strip().split()[0].split("/")
    if len(rows) != 8:
        raise ValueError(f"placement must have 8 ranks, got {len(rows)}")
    grid: list[list[str]] = []
    for rank in rows:
        row: list[str] = []
        for ch in rank:
            if ch.isdigit():
                row.extend(["empty"] * int(ch))
            elif ch in CLASS_TO_IDX:
                row.append(ch)
            else:
                raise ValueError(f"bad FEN char: {ch!r}")
        if len(row) != 8:
            raise ValueError(f"rank {rank!r} expands to {len(row)} squares")
        grid.append(row)
    return grid


def grid_to_placement(grid: list[list[str]]) -> str:
    """Inverse of fen_to_grid (used by Phase 3 fen.py)."""
    ranks = []
    for row in grid:
        rank, run = "", 0
        for cell in row:
            if cell == "empty":
                run += 1
            else:
                if run:
                    rank += str(run)
                    run = 0
                rank += cell
        if run:
            rank += str(run)
        ranks.append(rank)
    return "/".join(ranks)
