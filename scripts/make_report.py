"""Generate the 15-section course report PDF into docs/BoardScan_Report.pdf.
Usage: python scripts/make_report.py  (needs fpdf2: pip install fpdf)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fpdf import FPDF  # noqa: E402

DOCS = ROOT / "docs"
DEBUG = ROOT / "debug"
OUT = DOCS / "BoardScan_Report.pdf"
REPO = "https://github.com/abhishek-k-33/boardscan"


def clean(s: str) -> str:
    return (s.replace("\u2192", "->").replace("\u2013", "-").replace("\u2014", "-")
             .replace("\u201c", '"').replace("\u201d", '"').replace("\u2265", ">=")
             .replace("\u00d7", "x").replace("\u00b1", "+/-"))


class Report(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"BoardScan Project Report  |  p. {self.page_no()}/{{nb}}", align="C")

    def h1(self, t):
        if self.get_y() > 245:  # keep heading with following content
            self.add_page()
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 40, 120)
        self.multi_cell(0, 9, clean(t), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def h2(self, t):
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(0, 7, clean(t), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, t):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, clean(t), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullets(self, items):
        self.set_font("Helvetica", "", 10)
        for it in items:
            self.multi_cell(0, 6, clean(f"  - {it}"), new_x="LMARGIN", new_y="NEXT")
            self.ln(1)
        self.ln(1)

    def table(self, head, rows, widths=None):
        self.set_font("Helvetica", "B", 9)
        w = widths or [45, 145]
        for i, h in enumerate(head):
            self.cell(w[i], 7, clean(h), border=1)
        self.ln()
        self.set_font("Helvetica", "", 9)
        for row in rows:
            for i, c in enumerate(row):
                self.cell(w[i], 6, clean(str(c)), border=1)
            self.ln()
        self.ln(2)

    def img(self, path, width=170, caption=""):
        p = DOCS / path if not str(path).startswith("/") else Path(path)
        if not Path(p).exists():
            self.body(f"[missing figure: {path}]")
            return
        if self.get_y() > 200:
            self.add_page()
        self.image(str(p), x=(210 - width) / 2, w=width)
        if caption:
            self.set_font("Helvetica", "I", 9)
            self.multi_cell(0, 6, clean(caption), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


r = Report()
r.alias_nb_pages("{nb}")
r.set_auto_page_break(True, 20)

# 1. Cover
r.add_page()
r.ln(55)
r.set_font("Helvetica", "B", 26)
r.multi_cell(0, 12, "BoardScan", align="C", new_x="LMARGIN", new_y="NEXT")
r.set_font("Helvetica", "", 16)
r.multi_cell(0, 10, "Chess Position Digitiser (Photo -> FEN -> Evaluation)", align="C", new_x="LMARGIN", new_y="NEXT")
r.ln(6)
r.set_font("Helvetica", "", 12)
r.multi_cell(0, 8, "Computer Vision - Course Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
r.ln(10)
for line in ["Name: [Your Name]", "Roll No.: [Roll No.]", "Course / Faculty: [Course]",
             "Date: [Date]", f"Repository: {REPO}"]:
    r.multi_cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")

# 2. Introduction
r.add_page()
r.h1("2. Introduction")
r.body("Over-the-board chess players must re-enter positions into an engine by hand, "
       "which is slow and error-prone. BoardScan photographs a physical board from an "
       "arbitrary angle and returns the position in FEN plus an engine evaluation. "
       "Board localisation uses classical CV only (edge detection, Hough transform, "
       "homography); a small CNN is used solely for per-square piece classification. "
       "The report documents requirements, design, implementation, honest evaluation "
       "results, and lessons learned.")

# 3. Problem Statement
r.h1("3. Problem Statement")
r.body("Manual re-entry of over-the-board positions into analysis engines is slow and "
       "error-prone. Target users: club players digitising games, coaches reviewing "
       "student games, tournament arbiters recording positions. Scope: one RGB photo "
       "of a full board (0-45 deg tilt, varied lighting) -> validated FEN + eval. "
       "Out of scope: video tracking, partial boards, clock/scoresheet OCR.")

# 4. Functional Requirements
r.h1("4. Functional Requirements")
r.table(["Module", "Input -> Output"],
        [["1. Detection & Rectification", "RGB photo -> 800x800 top-down board"],
         ["2. Square Segmentation & Classification", "warped board -> 8x8 label grid (13 classes)"],
         ["3. FEN Assembly & Engine Analysis", "label grid + side-to-move -> FEN, cp eval, best move"],
         ["4. Debug / Visualisation Layer", "any stage -> annotated overlay images"]],
        widths=[70, 120])

# 5. NFRs
r.h1("5. Non-functional Requirements")
r.table(["NFR", "Target / evidence"],
        [["Performance", "Full CLI pipeline 0.07-0.08 s on CPU (budget 3 s); downscale to 1200 px"],
         ["Reliability", "Every stage returns a typed ok/data/error result; UI shows errors, never crashes"],
         ["Usability", "Editable 8x8 grid: one wrong square never forces a retake"],
         ["Maintainability", "All CV params in config.py; pure independently-testable functions"],
         ["Observability", "--debug dumps numbered stage images to debug/"],
         ["Resource efficiency", "212k-param CNN, 829 KB fp32 (budget 5 MB), CPU-only"]],
        widths=[40, 150])

# 6. Architecture
r.add_page()
r.h1("6. System Architecture")
r.img("architecture.png", width=175, caption="Fig. 1 - Module data flow: photo -> FEN + eval.")
r.body("Photo -> preprocess (gray/CLAHE/bilateral) -> Canny/Hough/lines -> corners "
       "(lattice + contour fallback, larger quad wins) -> homography warp (800x800) -> "
       "biased 8x8 slicing -> PieceCNN -> orientation-aware FEN assembly (king-count "
       "validation) -> Stockfish or degraded FEN-only mode. config.py feeds every stage; "
       "the debug layer taps each one. Storage is file-based: data/raw photos + .fen "
       "sidecars, data/squares crops + manifests (no database; ER diagram not applicable).")

# 7. Design diagrams
r.h1("7. Design Diagrams")
r.img("usecase.png", width=170, caption="Fig. 2 - Use cases: upload, review stages, correct labels, copy FEN.")
r.img("sequence.png", width=175, caption="Fig. 3 - Upload-to-FEN sequence across UI and modules.")
r.img("class-diagram.png", width=175, caption="Fig. 4 - Key result/class containers and their relations.")
r.body("Workflow (pipeline order): upload -> detect_and_rectify -> slice_board -> "
       "predict_grid -> grid_to_fen -> analyze -> display. Each arrow carries a typed "
       "result; any failure short-circuits to a user-facing message.")

# 8. Design decisions
r.add_page()
r.h1("8. Design Decisions & Rationale")
r.bullets([
    "Bilateral over Gaussian: preserves board edges while smoothing wood grain; one-flag ablation toggle.",
    "CLAHE over global equalisation: handles uneven lamp light across the board.",
    "Probabilistic Hough over standard Hough: segment endpoints enable angle clustering; minLineLength tied to square size.",
    "Lattice + contour fallback (larger quad wins): lattice extrapolates clipped corners; contour traces occluded borders.",
    "Per-square CNN over whole-board detector: 64 easy 13-class problems beat "
    "one hard 32-object detector on only about eight thousand training squares.",
    "Split by photo, not by square: same-photo squares share lighting; naive split inflates accuracy (we reproduced a fake 100%).",
    "King-count validation: >1 king per side is an error; missing kings (empty calibration boards) only warn.",
    "Mixed synthetic + realistic training: single-style training collapses on unseen art (train/serve skew).",
])

# 9. Implementation
r.h1("9. Implementation Details")
r.bullets([
    "Stack: Python, OpenCV primitives, NumPy, PyTorch (classifier only), python-chess, Streamlit, pytest.",
    "detection/: preprocess.py, lines.py, corners.py, rectify.py, pipeline.py - pure functions, config-driven.",
    "squares/: slicer.py (100x100 crops, +25 px upward bias for piece heads), labels.py (FEN <-> grid), classifier.py.",
    "model/: PieceCNN 48/96/192ch + GAP + FC(13); SquareDataset with brightness/contrast, +/-3 deg rot, +/-2 px shift; no hflip.",
    "notation/: fen.py (orientation score, auto-flip, ambiguous flag), engine.py (fixed-depth Stockfish, degraded mode).",
    "Training: 8572 squares (3900 synthetic + 4672 rendered-board), 70/15/15 by photo, Adam 1e-3 + StepLR, 96px input.",
    "Data: honi05/chess-positions-cv sample (FEN in filename) imported to data/raw/NNN.jpg + NNN.fen; disclosed openly.",
])

# 10. Results
r.add_page()
r.h1("10. Screenshots / Results")
r.h2("Pipeline stages (debug dumps for data/raw/000.jpg)")
stages = [("01_gray.png", "Fig. 5 - grayscale + downscale"),
          ("04_edges.png", "Fig. 6 - Canny edge map"),
          ("07_warped.png", "Fig. 7 - rectified 800x800 board"),
          ("08_overlay.png", "Fig. 8 - Hough lines + corners overlay")]
x0, top, cw = 15, r.get_y(), 80
import cv2 as _cv2
row_h = 0
cells = []
for name, cap in stages:
    ih, iw = _cv2.imread(str(DEBUG / name)).shape[:2]
    cells.append((name, cap, cw * ih / iw))
row_h = max(h for _, _, h in cells) + 10
for i, (name, cap, h) in enumerate(cells):
    col, row = i % 2, i // 2
    x, y = x0 + col * 92, top + row * row_h
    r.image(str(DEBUG / name), x=x, y=y, w=cw)
    r.set_xy(x, y + h + 1)
    r.set_font("Helvetica", "I", 8)
    r.multi_cell(cw, 5, clean(cap), align="C", new_x="LMARGIN", new_y="NEXT")
r.set_y(top + 2 * row_h + 2)
r.img("confusion_matrix.png", width=150, caption="Fig. 9 - Confusion matrix, combined held-out set (1344 squares).")
r.h2("Measured results (honest)")
r.table(["Metric", "Value"],
        [["Rectify success (15 synthetic boards)", "100%, grid within 10 px"],
         ["Corner error mean / p95", "1.5 px / 2.0 px"],
         ["Classifier per-square (synthetic held-out)", "100% (640 squares)"],
         ["Classifier per-square (combined held-out)", "89.4% (1344 squares)"],
         ["Live demo, 5 rendered boards", "53-60/64 squares each; 0/5 exact FEN"],
         ["Full CLI time", "about 0.08 s CPU"],
         ["Unit tests", "16/16 pass"]],
        widths=[70, 120])
r.h2("Preprocessing ablation (15 synthetic boards)")
r.table(["Setting", "Success", "Mean px", "p95 px"],
        [["CLAHE on + bilateral (shipped)", "100%", "1.5", "2.0"],
         ["CLAHE off + bilateral", "100%", "1.8", "2.6"],
         ["CLAHE on + gaussian", "100%", "1.2", "1.7"],
         ["CLAHE off + gaussian", "100%", "1.3", "1.9"]],
        widths=[70, 30, 30, 30])
r.body("On clean synthetic boards every setting succeeds and differences are sub-pixel "
       "noise (Gaussian even edges ahead slightly). The toggles exist so the same study "
       "can be repeated on real photos with lighting variation and wood grain, where "
       "CLAHE/bilateral are expected to matter - listed as future work.")

# 11. Testing
r.h1("11. Testing Approach")
r.bullets([
    "test_rectify.py: Phase-1 acceptance on synthetic boards (grid tolerance 10 px, 90% bar).",
    "test_squares.py: slicer determinism/bias, FEN mapping, no-leakage split, 5 MB budget.",
    "test_fen.py: round-trips, orientation flip/tie, two-kings error, bad-input rejection.",
    "test_pipeline.py: composed-board exact-FEN smoke test; engine degraded/live modes.",
    "Acceptance mapping: detection 15/15, classifier 89.4% realistic (spec bar 95% - documented gap), "
    "e2e exact-FEN on synthetic positions pass, 0/5 on realistic sample (reported, not hidden).",
])

# 12-15
r.add_page()
r.h1("12. Challenges Faced")
r.bullets([
    "OpenCV 5 returns HoughLinesP as (N,4) not (N,1,4): handled both shapes.",
    "Dashed board-border edges missed by strict Hough: gap-tolerant tuning (minLen 60, gap 30).",
    "Lattice hull locked onto interior grid lines: larger-quad-wins against contour fixed it.",
    "Class-pure pseudo-photos faked 100% accuracy: mixed dealing exposed true 77%, then bolder shapes + mild aug + wider net recovered.",
    "Train/serve skew (wood-tone training vs grey boards): mixed realistic data closed most of it.",
    "Laptop too slow for training: moved to Colab GPU via checked-in notebook.",
    "Streamlit data_editor has no help kwarg in deployed version: replaced with caption.",
])
r.h1("13. Learnings & Key Takeaways")
r.bullets([
    "Classical detection generalises across art styles for free; the CNN only knows what it was shown.",
    "Split discipline (by photo) and mixed-class photos are what make accuracy numbers believable.",
    "Exact-FEN is brutal: 89% per-square still gives 0/5 exact boards - hence human-in-the-loop correction.",
    "Typed per-stage results turn every failure into report evidence instead of a crash.",
])
r.h1("14. Future Enhancements")
r.bullets([
    "Retrain on ChessReD real smartphone photos; repeat ablation on real lighting.",
    "Install Stockfish for live eval.Customise: side-to-move inference from move history.",
    "Mobile capture app; video-frame averaging for robustness.",
])
r.h1("15. References")
r.bullets([
    "Masouris et al., End-to-End Chess Recognition + ChessReD, 4TU.ResearchData, DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f.",
    "honi05/chess-positions-cv, Hugging Face datasets (rendered boards with FEN filenames).",
    "Fenit annotated chessboard dataset, Mendeley Data, DOI 10.17632/wpp9xnbcp6.",
    "OpenCV docs (Canny, HoughLinesP, getPerspectiveTransform); python-chess; PyTorch; Streamlit.",
])

r.output(str(OUT))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {r.page_no()} pages)")
