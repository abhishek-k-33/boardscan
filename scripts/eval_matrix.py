"""Eval-only: score committed weights on a manifest's by-photo test split and
refresh docs/confusion_matrix.png (no training).
Usage: python scripts/eval_matrix.py --manifest data/combined_manifest.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

import torch
from torch.utils.data import DataLoader

from src.model.architecture import PieceCNN
from src.model.dataset import SquareDataset, load_manifest, split_by_photo
from src.model.train import evaluate, save_confusion_png
from src.squares.labels import CLASS_TO_IDX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "combined_manifest.csv")
    ap.add_argument("--weights", type=Path, default=config.CLASSIFIER_WEIGHTS)
    ap.add_argument("--out", type=Path, default=config.CONFUSION_MATRIX_PNG)
    ap.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    args = ap.parse_args()
    rows = load_manifest(args.manifest)
    _, _, te = split_by_photo(rows, seed=args.seed)
    net = PieceCNN()
    net.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    loader = DataLoader(SquareDataset(te, ROOT, CLASS_TO_IDX, train=False),
                        batch_size=256, num_workers=0)
    acc, _, _, cm = evaluate(net, loader, torch.device("cpu"))
    save_confusion_png(cm, args.out)
    print(f"test squares={len(te)} acc={acc:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
