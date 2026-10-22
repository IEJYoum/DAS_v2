"""Tiny spatial encoder and swappable classifier head. No data IO here."""
import torch
from torch import nn
from torch.nn import functional as F


def conv_block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1),
                         nn.GroupNorm(4, outputs), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1),
                         nn.GroupNorm(4, outputs), nn.ReLU())


class PixelUNet(nn.Module):
    """Three superstain planes -> per-pixel class logits, at input resolution.

    Masks/labels never enter forward(). Swap head for a per-pixel classifier
    or change build_model for another architecture with the same BCHW contract.
    """
    def __init__(self, input_channels=3, class_count=4, base_width=8, depth=2, head="linear"):
        super().__init__()
        if depth < 1 or base_width < 4 or base_width % 4:
            raise ValueError("depth >= 1; base_width must be a positive multiple of 4")
        if head != "linear":
            raise ValueError("First pass implements the linear head; replace model.head for experiments")
        self.depth = depth
        widths = [base_width * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([conv_block(input_channels if i == 0 else widths[i-1], w)
                                   for i, w in enumerate(widths)])
        self.up = nn.ModuleList([conv_block(widths[i+1] + widths[i], widths[i])
                                 for i in reversed(range(depth))])
        self.head = nn.Conv2d(base_width, class_count, 1)

    def forward_features(self, x):
        """Return B x base_width x H x W features for the classifier head."""
        if x.ndim != 4 or any(n % (2 ** self.depth) for n in x.shape[-2:]):
            raise ValueError("Input must be BCHW with H/W divisible by 2**depth")
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < self.depth:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return x

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_model(model_cfg):
    # Deferred: reconstructible MLP/CoE heads and expanded lineage inputs/classes.
    # Keep factories checkpoint-driven; see classifier_recipe_control_surface_audit_v0.txt.
    return PixelUNet(**model_cfg)
