"""Phase 2c: square dataset with split BY SOURCE PHOTO (never by square).

Why: squares from one photo share lighting/angle/blur. A naive per-square
split puts near-duplicates in train and test and inflates accuracy. Splitting
by photo measures real generalization to unseen photos.
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np

import config

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH = True
except ImportError:
    _TORCH = False


def split_by_photo(rows: list[tuple[str, str, str]],
                   ratios: tuple[float, float, float] = config.TRAIN_SPLIT_RATIOS,
                   seed: int = config.TRAIN_SEED):
    """rows: (img_path, label, photo_id) -> (train, val, test) row lists."""
    by_photo: dict[str, list] = {}
    for row in rows:
        by_photo.setdefault(row[2], []).append(row)
    photos = sorted(by_photo)
    rng = random.Random(seed)
    rng.shuffle(photos)
    n = len(photos)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    tr = [r for p in photos[:n_tr] for r in by_photo[p]]
    va = [r for p in photos[n_tr:n_tr + n_va] for r in by_photo[p]]
    te = [r for p in photos[n_tr + n_va:] for r in by_photo[p]]
    return tr, va, te


def load_manifest(manifest: Path) -> list[tuple[str, str, str]]:
    with open(manifest, newline="") as f:
        return [(r["path"], r["label"], r["photo_id"]) for r in csv.DictReader(f)]


def augment(board_crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness/contrast jitter, +-deg rotation, slight translation.
    No horizontal flip (would mirror piece asymmetry). All params from config."""
    img = board_crop.astype(np.float32)
    b = float(rng.uniform(*config.AUG_BRIGHTNESS_RANGE))
    c = float(rng.uniform(*config.AUG_CONTRAST_RANGE))
    img = ((img - 127.5) * c + 127.5) * b
    h, w = img.shape[:2]
    ang = float(rng.uniform(-config.AUG_ROTATION_DEG, config.AUG_ROTATION_DEG))
    tx = float(rng.uniform(-config.AUG_TRANSLATE_PX, config.AUG_TRANSLATE_PX))
    ty = float(rng.uniform(-config.AUG_TRANSLATE_PX, config.AUG_TRANSLATE_PX))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(img, 0, 255).astype(np.uint8)


class SquareDataset(Dataset if _TORCH else object):
    def __init__(self, rows, img_root: Path, label_to_idx: dict,
                 train: bool = False, seed: int = config.TRAIN_SEED):
        if not _TORCH:
            raise ImportError("torch is required (pip install -r requirements.txt)")
        self.rows = rows
        self.root = Path(img_root)
        self.l2i = label_to_idx
        self.train = train
        self.rng = np.random.default_rng(seed + (0 if train else 999))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, label, _ = self.rows[i]
        img = cv2.imread(str(self.root / rel), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"missing square image: {rel}")
        img = cv2.resize(img, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE))
        if self.train:
            img = augment(img, self.rng)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
        return x, self.l2i[label]
