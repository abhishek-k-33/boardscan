"""Phase 3b: Stockfish wrapper at fixed depth. Missing binary -> degraded
FEN-only mode (still ok, never raises to the UI)."""
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class EngineResult:
    ok: bool
    score_cp: int | None = None     # centipawns from White's perspective
    mate_in: int | None = None      # mate distance if forced, else None
    best_move: str | None = None    # UCI, first move of the PV
    pv: list = field(default_factory=list)
    degraded: bool = False          # True when Stockfish is absent
    error: str = ""


def _resolve_binary(path: str | Path = config.STOCKFISH_PATH) -> str | None:
    p = str(path)
    if Path(p).is_file():
        return p
    found = shutil.which(p) or shutil.which("stockfish")
    return found


def analyze(fen: str, depth: int = config.STOCKFISH_DEPTH,
            path: str | Path = config.STOCKFISH_PATH,
            timeout: float = 30.0) -> EngineResult:
    binary = _resolve_binary(path)
    if binary is None:
        return EngineResult(ok=True, degraded=True,
                            error=f"Stockfish not found at {path}; FEN-only mode")
    try:
        import chess
        import chess.engine
        board = chess.Board(fen)
        engine = chess.engine.SimpleEngine.popen_uci(binary)
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
        finally:
            engine.quit()
        score = info.get("score")
        pv = [m.uci() for m in info.get("pv", [])]
        if score is None:
            return EngineResult(ok=False, error="engine returned no score")
        white = score.white()
        return EngineResult(
            ok=True,
            score_cp=white.score(),
            mate_in=white.mate(),
            best_move=pv[0] if pv else None,
            pv=pv,
        )
    except Exception as e:
        return EngineResult(ok=False, error=f"engine analysis failed: {e}")
