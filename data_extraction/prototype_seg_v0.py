import torch
import torch.nn as nn


class NucleusSegNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(2)

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.final_nucleus = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.final_boundary = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.pool(e3)

        x = nn.functional.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(x + e3)
        x = nn.functional.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(x + e2)
        x = nn.functional.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(x + e1)
        return self.final_nucleus(x), self.final_boundary(x)
