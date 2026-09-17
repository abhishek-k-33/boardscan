"""Phase 2d: tiny piece-classifier CNN — 3 conv blocks -> GAP -> FC(13).

Why per-square CNN over whole-board detection: 64 easy 13-class problems beat
one hard 32-object detection problem at this dataset size (~4k squares).
Stays under the 5MB / CPU-only NFR by a wide margin (~0.4MB fp32).
"""
import torch
import torch.nn as nn

import config


class PieceCNN(nn.Module):
    def __init__(self, num_classes: int = config.MODEL_NUM_CLASSES,
                 channels: tuple = config.MODEL_CHANNELS):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.MaxPool2d(2),                      # 64 -> 32
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.MaxPool2d(2),                      # 32 -> 16
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)        # global average pooling
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
