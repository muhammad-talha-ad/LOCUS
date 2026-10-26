"""Bake-off box probe on frozen features.

A near-linear attention probe: the last-token state h queries the image-token
states with a single bilinear attention. The centre is the soft-argmax over
token centres; width/height (and a small centre offset) are linear reads of
the attended feature and h. Deliberately weak, so that it ranks backbones by
how much box geometry their frozen features carry.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBoxProbe(nn.Module):
    def __init__(self, d: int, k: int = 256):
        super().__init__()
        # Parameter-free LayerNorm: frozen VLM states can carry huge outlier dims.
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.q, self.k = nn.Linear(d, k), nn.Linear(d, k)
        self.read = nn.Linear(2 * d, 4)
        self.scale = k**-0.5
        nn.init.zeros_(self.read.weight)
        nn.init.zeros_(self.read.bias)

    def forward(self, h, vis, xy, mask):
        """h [B,d], vis [B,N,d], xy [B,N,2], mask [B,N] (True = real token) → xyxy [B,4]."""
        h, vis = self.norm(h), self.norm(vis)
        att = torch.einsum("bk,bnk->bn", self.q(h), self.k(vis)) * self.scale
        a = att.masked_fill(~mask, float("-inf")).softmax(-1)
        centre = torch.einsum("bn,bnc->bc", a, xy)
        pooled = torch.einsum("bn,bnd->bd", a, vis)
        r = self.read(torch.cat([pooled, h], -1))
        c = (centre + 0.1 * torch.tanh(r[:, :2])).clamp(0, 1)
        wh = torch.sigmoid(r[:, 2:] - 1.0)
        return torch.cat([c - wh / 2, c + wh / 2], -1).clamp(0, 1)


def _area(x: torch.Tensor) -> torch.Tensor:
    return (x[:, 2:] - x[:, :2]).clamp(min=0).prod(-1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Element-wise IoU and union of paired boxes."""
    lt, rb = torch.maximum(a[:, :2], b[:, :2]), torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = _area(a) + _area(b) - inter
    return inter / (union + 1e-7), union


def box_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 + GIoU"""
    iou, union = box_iou_xyxy(pred, gt)
    lt, rb = torch.minimum(pred[:, :2], gt[:, :2]), torch.maximum(pred[:, 2:], gt[:, 2:])
    hull = (rb - lt).clamp(min=0).prod(-1) + 1e-7
    giou = iou - (hull - union) / hull
    return 5 * F.l1_loss(pred, gt) + 2 * (1 - giou).mean()
