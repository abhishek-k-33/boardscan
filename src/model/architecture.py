"""Phase 2d: tiny piece-classifier CNN — 3 conv blocks -> GAP -> FC(13).

Why per-square CNN over whole-board detection: 64 easy 13-class problems beat
one hard 32-object detection problem at this dataset size (~4k squares).
Stays under the 5MB / CPU-only NFR by a wide margin (~0.4MB fp32).
"""
import torch
import torch.nn as nn

import config


class PieceCNN(nn.Module):
    def __init__(self, num_classes: int = config.MODEL_NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                      # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                      # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)        # global average pooling
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
