"""Phase 2f: CNN inference wrapper — BGR crop -> (label, confidence)."""
from pathlib import Path

import cv2
import numpy as np

import config

try:
    import torch
    import torch.nn.functional as F
    from src.model.architecture import PieceCNN
    _TORCH = True
except ImportError:
    _TORCH = False


class PieceClassifier:
    def __init__(self, weights: Path = config.CLASSIFIER_WEIGHTS):
        if not _TORCH:
            raise ImportError("torch is required (pip install -r requirements.txt)")
        self.net = PieceCNN()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval()
        self.labels = list(config.CLASS_LABELS)

    def predict(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        img = cv2.resize(crop_bgr, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        with torch.no_grad():
            probs = F.softmax(self.net(torch.from_numpy(rgb.transpose(2, 0, 1))[None]), 1)[0]
        i = int(probs.argmax())
        return self.labels[i], float(probs[i])

    def predict_grid(self, crops: list[tuple[str, np.ndarray]]) -> list[list[str]]:
        """64 (square_name, crop) pairs in slice_board order -> 8x8 label grid."""
        cells = [self.predict(c)[0] for _, c in crops]
        return [cells[r * 8:(r + 1) * 8] for r in range(8)]
