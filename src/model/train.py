"""Phase 2e: train the 13-class piece CNN. Stratified split by source photo.

Usage:
    python src/model/train.py --manifest data/synth_squares/manifest.csv --epochs 15
Outputs: models/piece_cnn.pt, docs/confusion_matrix.png, per-class P/R table.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.architecture import PieceCNN, count_parameters
from src.model.dataset import SquareDataset, load_manifest, split_by_photo
from src.squares.labels import CLASS_TO_IDX, IDX_TO_CLASS


def evaluate(net, loader, device):
    net.eval()
    C = config.MODEL_NUM_CLASSES
    cm = np.zeros((C, C), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = net(x.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                cm[t, p] += 1
    acc = float(np.trace(cm) / max(1, cm.sum()))
    prec, rec = {}, {}
    for i in range(C):
        prec[IDX_TO_CLASS[i]] = float(cm[i, i] / max(1, cm[:, i].sum()))
        rec[IDX_TO_CLASS[i]] = float(cm[i, i] / max(1, cm[i, :].sum()))
    return acc, prec, rec, cm


def save_confusion_png(cm: np.ndarray, path: Path):
    C = cm.shape[0]
    cell, head = 64, 170
    size = head + C * cell
    img = np.full((size, size, 3), 255, np.uint8)
    row_sum = cm.sum(1, keepdims=True).clip(min=1)
    for i in range(C):
        for j in range(C):
            v = cm[i, j] / row_sum[i, 0]
            color = (int(255 - 200 * v), int(255 - 120 * v), 255)
            x0, y0 = head + j * cell, head + i * cell
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), color, -1)
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (180, 180, 180), 1)
            cv2.putText(img, str(cm[i, j]), (x0 + 8, y0 + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        cv2.putText(img, IDX_TO_CLASS[i], (10, head + i * cell + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        cv2.putText(img, IDX_TO_CLASS[i], (head + i * cell + 8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    cv2.putText(img, "true \\ pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "synth_squares" / "manifest.csv")
    ap.add_argument("--img-root", type=Path, default=ROOT)
    ap.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    ap.add_argument("--batch", type=int, default=config.TRAIN_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.TRAIN_LR)
    ap.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    ap.add_argument("--out", type=Path, default=config.CLASSIFIER_WEIGHTS)
    ap.add_argument("--confusion", type=Path, default=config.CONFUSION_MATRIX_PNG)
    ap.add_argument("--device", type=str, default="auto",
                    help="'auto' (cuda if available), 'cpu', or 'cuda'")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")
    rows = load_manifest(args.manifest)
    tr, va, te = split_by_photo(rows, seed=args.seed)
    print(f"squares: train={len(tr)} val={len(va)} test={len(te)} "
          f"(photos: {len({r[2] for r in tr})}/{len({r[2] for r in va})}/{len({r[2] for r in te})})")

    def loader(rs, train):
        ds = SquareDataset(rs, args.img_root, CLASS_TO_IDX, train=train, seed=args.seed)
        return DataLoader(ds, batch_size=args.batch, shuffle=train, num_workers=0)

    tr_l, va_l, te_l = loader(tr, True), loader(va, False), loader(te, False)
    net = PieceCNN().to(device)
    print(f"params: {count_parameters(net):,} "
          f"({count_parameters(net) * 4 / 1024:.0f} KB fp32, limit 5MB)")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_va, best_state = 0.0, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        tot, n = 0.0, 0
        for x, y in tr_l:
            opt.zero_grad()
            loss = loss_fn(net(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        va_acc, _, _, _ = evaluate(net, va_l, device)
        print(f"epoch {ep:02d}/{args.epochs} loss={tot / n:.4f} val_acc={va_acc:.4f}", flush=True)
        if va_acc >= best_va:
            best_va, best_state = va_acc, {k: v.cpu() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)

    acc, prec, rec, cm = evaluate(net, te_l, device)
    print(f"\nTEST per-square accuracy: {acc:.4f} (acceptance >= {config.CLASSIFIER_ACCEPT_ACC}) "
          f"[{time.time() - t0:.0f}s]")
    print(f"{'class':>6} {'prec':>6} {'recall':>6}")
    for c in config.CLASS_LABELS:
        print(f"{c:>6} {prec[c]:6.3f} {rec[c]:6.3f}")
    save_confusion_png(cm, args.confusion)
    print(f"weights -> {args.out}\nconfusion matrix -> {args.confusion}")
    print("ACCEPTANCE:", "PASS" if acc >= config.CLASSIFIER_ACCEPT_ACC else "FAIL")
    return 0 if acc >= config.CLASSIFIER_ACCEPT_ACC else 3


if __name__ == "__main__":
    raise SystemExit(main())
