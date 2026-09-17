"""Module 4 UI: upload -> pipeline stages side-by-side -> editable 8x8 grid ->
final FEN + eval. Run: streamlit run app/streamlit_app.py"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.debug.overlay import draw_grid
from src.detection.pipeline import detect_and_rectify_array
from src.notation.engine import analyze
from src.notation.fen import grid_to_fen
from src.squares.classifier import PieceClassifier
from src.squares.labels import CLASS_TO_IDX
from src.squares.slicer import slice_board

st.set_page_config(page_title="BoardScan", layout="wide")
st.title("BoardScan — Photo → FEN → Eval")

uploaded = st.file_uploader("Board photo", type=["jpg", "jpeg", "png"])
side = st.radio("Side to move", ["w", "b"], horizontal=True)
force_flip = st.checkbox("Flip board orientation (override auto-detect)")

@st.cache_resource
def get_classifier():
    try:
        return PieceClassifier(), ""
    except Exception as e:
        return None, str(e)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if uploaded is not None:
    data = np.frombuffer(uploaded.read(), np.uint8)
    photo = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if photo is None:
        st.error("Could not decode image.")
        st.stop()
    with st.spinner("Detecting board…"):
        res = detect_and_rectify_array(photo, debug=False)
    if not res.ok:
        st.error(f"Board detection failed: {res.error}")
        st.stop()
    st.success(f"Board found (method={res.method})")

    c1, c2, c3 = st.columns(3)
    c1.image(bgr_to_rgb(photo), caption="Original", use_container_width=True)
    c2.image(bgr_to_rgb(res.warped), caption="Rectified 800×800", use_container_width=True)
    c3.image(bgr_to_rgb(draw_grid(res.warped)), caption="Square grid", use_container_width=True)

    st.subheader("1. Predicted position (edit any square, then re-run FEN below)")
    clf, clf_err = get_classifier()
    if clf is None:
        st.error(f"Classifier unavailable: {clf_err}")
        st.stop()
    with st.spinner("Classifying squares…"):
        pred_grid = clf.predict_grid(slice_board(res.warped))
    files = [f"{chr(ord('a') + c)}" for c in range(8)]
    df = pd.DataFrame(pred_grid, columns=files)
    edited = st.data_editor(df, use_container_width=True)
    st.caption("Each cell: empty, P N B R Q K (white), p n b r q k (black)")
    # sanitise edits: unknown tokens revert to the prediction
    grid = []
    fixed = 0
    for r in range(8):
        row = []
        for c in range(8):
            cell = str(edited.iat[r, c]).strip()
            if cell not in CLASS_TO_IDX:
                cell = pred_grid[r][c]
                fixed += 1
            row.append(cell)
        grid.append(row)
    if fixed:
        st.warning(f"{fixed} invalid cell(s) reverted to predictions.")
    if force_flip:
        grid = [row[:] for row in reversed(grid)]

    st.subheader("2. FEN + evaluation")
    fr = grid_to_fen(grid, side_to_move=side, auto_orient=not force_flip)
    if not fr.ok:
        st.error(f"Illegal position: {fr.error}")
        st.stop()
    if fr.flipped:
        st.info("Orientation auto-flipped (white mass on rank 1).")
    if fr.ambiguous:
        st.warning("Orientation ambiguous — verify with the Flip checkbox.")
    for w in fr.warnings:
        st.warning(w)
    st.code(fr.fen, language=None)  # code block ships a copy button
    er = analyze(fr.fen)
    if er.degraded:
        st.info(er.error)
    elif not er.ok:
        st.error(er.error)
    else:
        mate = f" (mate in {er.mate_in})" if er.mate_in else ""
        st.metric("Eval (White POV)", f"{er.score_cp} cp{mate}")
        st.write(f"Best move: `{er.best_move}` — PV: `{' '.join(er.pv[:6])}`")
else:
    st.info("Upload a board photo to start. No photo? Generate a test one:\n"
            "`python scripts/synthetic_board.py` (empty boards for detection testing).")
