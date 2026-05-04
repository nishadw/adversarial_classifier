from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CNN for CIFAR-10 in PyTorch (NCHW), targeting >90% clean accuracy."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   128, 3, padding=1), nn.GroupNorm(16, 128),  nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(16, 128),  nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(32, 256),  nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(32, 256),  nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1), nn.GroupNorm(64, 512),  nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.GroupNorm(64, 512),  nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=[2, 3]))
