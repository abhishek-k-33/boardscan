"""BoardScan global config — ALL tunable CV params live here. No magic numbers elsewhere."""
from pathlib import Path

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
DEBUG_DIR = BASE_DIR / "debug"
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# ---- Canvas ----
WARP_SIZE = 800          # rectified board is WARP_SIZE x WARP_SIZE
SQUARE_SIZE = 100        # WARP_SIZE // 8
NUM_SQUARES = 8

# ---- Preprocessing ----
DOWNSCALE_MAX_DIM = 1200  # NFR: downscale 12MP input before edge detection (<3s CPU)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
# Ablation alternative:
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 0.0
USE_CLAHE = True
USE_BILATERAL = True  # False -> Gaussian (for ablation study)

# ---- Canny ----
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
CANNY_APERTURE_SIZE = 3

# ---- Probabilistic Hough ----
HOUGH_RHO = 1.0
HOUGH_THETA_DEG = 1.0
HOUGH_THRESHOLD = 80  # votes; low enough to catch dashed board-border edges
HOUGH_MIN_LINE_LENGTH = 60  # below one square (~90px in photo) so split border runs survive
HOUGH_MAX_LINE_GAP = 30  # link dashed border segments across weak dark-square stretches

# ---- Line clustering ----
ANGLE_CLUSTER_TOL_DEG = 15.0   # near-horizontal vs near-vertical split tolerance
MERGE_RHO_TOL = 15.0           # px, near-duplicate line merging
MERGE_THETA_TOL_DEG = 10.0
MERGE_MIDPOINT_TOL = 20.0

# ---- Corners ----
INTERSECTION_CLUSTER_TOL = 12.0  # px, merge nearby intersections (9x9 lattice)
MIN_INTERSECTIONS_FOR_LATTICE = 20
APPROX_POLY_EPSILON_RATIO = 0.02  # * perimeter, for fallback quad contour
CONTOUR_MIN_AREA_RATIO = 0.05     # quad must cover >=5% of image

# ---- Rectify acceptance (Phase 1 test) ----
GRID_TOLERANCE_PX = 10.0
RECTIFY_SUCCESS_RATE = 0.90

# ---- Squares (Module 2) ----
SQUARE_CROP_SIZE = 100        # deterministic crops from 800x800 warped board
SQUARE_VERTICAL_BIAS_PX = 25  # shift crop window upward: pieces are taller than
                              # their square, this keeps the piece head in frame
MODEL_INPUT_SIZE = 64         # CNN input (crops resized to this)
MODEL_NUM_CLASSES = 13        # 6 white + 6 black + empty
CLASS_LABELS = ('empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k')

# ---- Training ----
TRAIN_SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train/val/test, split BY SOURCE PHOTO
TRAIN_EPOCHS = 15
TRAIN_BATCH_SIZE = 64
TRAIN_LR = 1e-3
TRAIN_SEED = 7
CLASSIFIER_ACCEPT_ACC = 0.95
# Augmentation (no horizontal flip: it would mirror piece asymmetry).
# Mild ranges: the square crops already contain lighting/pose jitter, so the
# loader must not drown the 64px piece signal in extra distortion.
AUG_BRIGHTNESS_RANGE = (0.85, 1.15)
AUG_CONTRAST_RANGE = (0.85, 1.15)
AUG_ROTATION_DEG = 3.0
AUG_TRANSLATE_PX = 2
# Synthetic bootstrap data (real photos replace this via build_squares_dataset.py)
SYNTH_SQUARES_PER_CLASS = 160
SYNTH_SEED = 0

# ---- Paths ----
CLASSIFIER_WEIGHTS = BASE_DIR / "models" / "piece_cnn.pt"
SQUARES_MANIFEST = BASE_DIR / "data" / "squares_manifest.csv"
CONFUSION_MATRIX_PNG = BASE_DIR / "docs" / "confusion_matrix.png"

# ---- Engine (Module 3) ----
STOCKFISH_PATH = "/usr/games/stockfish"
STOCKFISH_DEPTH = 15
