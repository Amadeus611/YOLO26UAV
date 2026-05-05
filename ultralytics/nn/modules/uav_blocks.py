# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""UAV-oriented lightweight feature blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv, DWConv

__all__ = ("TextureAwareEnhance", "CrossScaleSelectiveFusion")


class _ChannelGate(nn.Module):
    """Lightweight channel attention used by UAV-specific blocks."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))


class TextureAwareEnhance(nn.Module):
    """Enhance shallow texture cues for UAV small-object detection."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.shortcut = Conv(c1, c2, 1, act=False) if c1 != c2 else nn.Identity()
        self.texture = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c2, 1))
        self.channel_gate = _ChannelGate(c2)
        self.gamma = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        texture = self.texture(x)
        refined = self.channel_gate(texture)
        return identity + self.gamma * refined


class CrossScaleSelectiveFusion(nn.Module):
    """Selectively fuse concatenated multi-scale features with channel and spatial gating."""

    def __init__(self, c1: int, c2: int, reduction: int = 4, use_spatial: bool = True):
        super().__init__()
        self.use_spatial = use_spatial
        self.align = Conv(c1, c2, 1)
        self.local = nn.Sequential(DWConv(c2, c2, 3), Conv(c2, c2, 1))
        self.channel_gate = _ChannelGate(c2, reduction=reduction)
        if use_spatial:
            self.spatial_gate = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.align(x)
        local = self.local(base)
        channel = self.channel_gate(local)
        if self.use_spatial:
            spatial = self.spatial_gate(torch.cat((local.mean(1, keepdim=True), local.amax(1, keepdim=True)), 1))
            return base + self.gamma * local * channel * spatial
        return base + self.gamma * local * channel
