"""Our re-implementation of Jag's box head over Qwen3_5Model (https://huggingface.co/xyzzzh/Jag, Apache-2.0).

LayerNorm -> Linear(h, 512) -> GELU -> Linear(512, 4) -> Sigmoid on the last valid token state,
normalised cxcywh out. Only the weights are taken from the release (data/external/jag_repo).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import Qwen3_5Config, Qwen3_5Model

JAG_DIR = Path(__file__).resolve().parents[2] / "data/external/jag_repo"
PROMPT = "Locate the object described by: {expression}"


class JagRef(nn.Module):
    def __init__(self, backbone: Qwen3_5Model, head_hidden: int = 512):
        super().__init__()
        self.backbone = backbone
        h = backbone.config.text_config.hidden_size
        self.bbox_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, head_hidden), nn.GELU(),
                                       nn.Linear(head_hidden, 4), nn.Sigmoid())

    @classmethod
    def load(cls, path: Path = JAG_DIR, dtype=torch.float32, device="cuda") -> "JagRef":
        cfg = json.loads((path / "config.json").read_text())
        bc = Qwen3_5Config.from_dict(cfg["backbone_config"])
        bc._attn_implementation = "sdpa"
        model = cls(Qwen3_5Model(bc), cfg["head_hidden_size"])
        model.load_state_dict(load_file(path / "model.safetensors"), strict=True)  # raises on any key/shape mismatch
        model.bbox_head.float()
        model.backbone.to(dtype)
        return model.to(device).eval()

    def forward(self, input_ids, attention_mask, **kw) -> torch.Tensor:
        """Normalised cxcywh [B, 4] (fp32) from the last valid token of each row."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kw).last_hidden_state
        pos = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
        last = pos.masked_fill(attention_mask == 0, -1).amax(1)
        pooled = hidden[torch.arange(len(hidden), device=hidden.device), last]
        with torch.autocast(pooled.device.type, enabled=False):
            return self.bbox_head(pooled.float())
