"""Cross-attention box decoder for the [BOX] query.

The [BOX] token's state queries the image-token states of the selected image.
Keys mix a mid-depth layer and the last layer (the bake-off found box geometry
peaks mid-network) and carry a 2-D sine position code of each token's centre.
The last layer's attention map (first layer's with refine=True) gives a soft-argmax reference centre; the box is
a refinement of it in logit space, predicted as (centre, size) so that
left < right and top < bottom hold by construction. A second linear read gives
the confidence logit, trained towards P(IoU >= 0.5).

With refine=True every layer after the first predicts a box (deep supervision),
and the next layer is conditioned on it as in DAB-/Deformable-DETR: the query
gets a positional code of the current box, and attention gets a Gaussian bias
centred on it with a learned per-head sharpness. Boxes are detached between
layers.

With dfl=K (needs refine) the last layer instead predicts, for each edge, a
K-way distribution over offsets in [-½, +½] × the current box size around the
previous layer's edge, read out as its expectation (GFL/DFL; LLM coordinate
tokens are distributions too). It is trained with the distribution focal loss
(dfl_loss) on top of L1/GIoU; its entropy is a per-edge uncertainty.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sine_xy(xy: torch.Tensor, dim: int) -> torch.Tensor:
    """[..., c] fractions → [..., dim] sine/cosine code (dim divisible by 2c)."""
    c = xy.shape[-1]
    freqs = torch.exp(torch.arange(dim // (2 * c), device=xy.device) * (-math.log(1000.0) / (dim // (2 * c))))
    ang = 2 * math.pi * xy[..., :, None] * freqs  # [..., c, dim/2c]
    return torch.cat([ang.sin(), ang.cos()], -1).flatten(-2)


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1).clamp(0, 1)


class DecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.nf = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv, pos, mask, qpos=None, bias=None):
        """q [B,d], kv/pos [B,N,d], mask [B,N], bias [B,heads,N] → (q', attention [B,heads,N])."""
        B, N, d = kv.shape
        h = self.heads
        x = self.nkv(kv)
        qh = self.q(self.nq(q) + (0 if qpos is None else qpos)).view(B, h, 1, d // h)
        kh = self.k(x + pos).view(B, N, h, d // h).transpose(1, 2)
        vh = self.v(x).view(B, N, h, d // h).transpose(1, 2)
        att = (qh @ kh.transpose(-1, -2)) / math.sqrt(d // h)  # [B,h,1,N]
        if bias is not None:
            att = att + bias[:, :, None]
        att = att.masked_fill(~mask[:, None, None], float("-inf")).softmax(-1)
        q = q + self.o((att @ vh).reshape(B, d))
        q = q + self.ff(self.nf(q))
        return q, att[:, :, 0]


class BoxDecoder(nn.Module):
    def __init__(self, d_in: int, n_feat: int = 2, d: int = 512, layers: int = 3, heads: int = 8,
                 n_query: int = 1, refine: bool = False, dfl: int = 0):
        super().__init__()
        assert refine or not dfl, "dfl refines the previous layer's box"
        # Parameter-free LayerNorm on backbone states (outlier dims), then a learned mix of layers.
        self.norm = nn.LayerNorm(d_in, elementwise_affine=False)
        self.kv_in = nn.Linear(n_feat * d_in, d)
        self.q_in = nn.Linear(n_query * n_feat * d_in, d)
        self.pos = nn.Linear(d, d)
        self.layers = nn.ModuleList(DecoderLayer(d, heads) for _ in range(layers))
        self.refine = refine
        n_heads_out = layers if refine else 1
        self.out_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_heads_out))
        self.box = nn.ModuleList(nn.Linear(d, 4) for _ in range(n_heads_out))
        for lin in self.box:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.dfl = dfl
        if dfl:
            self.box[-1] = nn.Linear(d, 4 * dfl)  # zero init → uniform distribution → expectation = previous edge
            nn.init.zeros_(self.box[-1].weight)
            nn.init.zeros_(self.box[-1].bias)
            self.register_buffer("bins", torch.linspace(-0.5, 0.5, dfl), persistent=False)
        self.dfl_aux: tuple | None = None
        if refine:
            self.box_pos = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.sharp = nn.Parameter(torch.full((heads,), 0.5))
        self.conf = nn.Linear(d, 1)
        self.aux: list[torch.Tensor] = []

    def _step(self, i: int, q: torch.Tensor, prev: torch.Tensor | None, ref: torch.Tensor | None) -> torch.Tensor:
        """Box (cxcywh) from layer output q: refine `prev` if given, else start from reference centre `ref`."""
        r = self.box[i](self.out_norm[i](q))
        if prev is None:
            c = torch.sigmoid(torch.logit(ref) + r[:, :2])
            wh = torch.sigmoid(r[:, 2:] - 1.5)
            return torch.cat([c, wh], -1)
        if self.dfl and i == len(self.layers) - 1:
            prev_xyxy = cxcywh_to_xyxy(prev)
            size = prev[:, 2:].repeat(1, 2).clamp_min(1e-3)  # [B,4]: w, h, w, h
            logits = r.view(-1, 4, self.dfl).float()
            edges = prev_xyxy + (logits.softmax(-1) * self.bins).sum(-1) * size
            self.dfl_aux = (logits, prev_xyxy, size)
            lo, hi = torch.minimum(edges[:, :2], edges[:, 2:]), torch.maximum(edges[:, :2], edges[:, 2:])
            return torch.cat([(lo + hi) / 2, hi - lo], -1)
        return torch.sigmoid(torch.logit(prev.clamp(1e-4, 1 - 1e-4)) + r)

    def dfl_loss(self, gt: torch.Tensor) -> torch.Tensor:
        """Distribution focal loss: cross-entropy onto the two bins bracketing each gt edge offset."""
        logits, prev, size = self.dfl_aux
        K = self.dfl
        t = ((gt - prev) / size).clamp(-0.5, 0.5)  # gt offset in box-size units
        x = (t + 0.5) * (K - 1)  # continuous bin index
        lo = x.floor().long().clamp(max=K - 2)
        w_hi = x - lo
        logp = logits.log_softmax(-1)
        nll = -(logp.gather(-1, lo[..., None])[..., 0] * (1 - w_hi) + logp.gather(-1, lo[..., None] + 1)[..., 0] * w_hi)
        return nll.mean()

    def forward(self, h: list[torch.Tensor], vis: list[torch.Tensor], xy: torch.Tensor, mask: torch.Tensor):
        """h: per-layer [B,n_query,d_in] query states; vis: per-layer [B,N,d_in]; xy [B,N,2]; mask [B,N].

        Returns xyxy boxes [B,4] in [0,1] and confidence logits [B]. With refine=True, self.aux holds the
        intermediate xyxy boxes (for deep supervision).
        """
        q = self.q_in(torch.cat([self.norm(x.float()).flatten(1) for x in h], -1))
        kv = self.kv_in(torch.cat([self.norm(x.float()) for x in vis], -1))
        d = kv.shape[-1]
        pos = self.pos(sine_xy(xy, d))
        self.aux = []
        box, ref = None, None
        for i, layer in enumerate(self.layers):
            qpos = bias = None
            if self.refine and box is not None:
                b = box.detach()
                qpos = self.box_pos(sine_xy(b, d))
                dist = ((xy - b[:, None, :2]) / (b[:, None, 2:] / 2 + 1e-3)).pow(2).sum(-1)  # [B,N]
                bias = -F.softplus(self.sharp)[None, :, None] * dist[:, None]
            q, att = layer(q, kv, pos, mask, qpos, bias)
            if i == 0 or not self.refine:  # refine: first layer's map; else the last layer's (as in the S1 comparison)
                ref = torch.einsum("bn,bnc->bc", att.mean(1), xy).clamp(1e-4, 1 - 1e-4)
            if self.refine:
                box = self._step(i, q, box, ref)
                self.aux.append(cxcywh_to_xyxy(box))
        if not self.refine:
            box = self._step(0, q, None, ref)
        return cxcywh_to_xyxy(box), self.conf(q)[:, 0]


class MLPHead(nn.Module):
    """Jag-style control (A2): LN → MLP on the deepest [BOX] state only, sigmoid cxcywh; no image-token reads."""

    def __init__(self, d_in: int, d: int = 512, **_):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, 4))
        self.conf = nn.Linear(d_in, 1)
        self.aux: list[torch.Tensor] = []

    def forward(self, h, vis, xy, mask):
        x = h[-1][:, -1].float()
        b = torch.sigmoid(self.mlp(x))
        return cxcywh_to_xyxy(b), self.conf(F.layer_norm(x, x.shape[-1:]))[:, 0]


def upgrade_state_dict(sd: dict) -> dict:
    """Checkpoints from before refine/n_query had single out_norm/box modules."""
    return {(k.replace("out_norm.", "out_norm.0.").replace("box.", "box.0.")
             if k.split(".")[0] in ("out_norm", "box") and not k.split(".")[1].isdigit() else k): v for k, v in sd.items()}


def conf_loss(logit: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
    """IoU-aware BCE: the target is whether this prediction reaches IoU >= 0.5."""
    return F.binary_cross_entropy_with_logits(logit, (iou.detach() >= 0.5).float())
