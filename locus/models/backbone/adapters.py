"""Frozen-backbone adapters for the week-1 bake-off.

Every adapter exposes the same three operations:
  encode(image, texts)    -> vision-token states, their (x, y) centres, last-token state per text
  option_logprobs(...)    -> log-probs over option letters (decision probe)
  prefill(image, text)    -> one timed forward pass (latency)

Prompt layout is always [image][text] so that, under causal attention, the
image-token states do not depend on the text; one image is encoded once and
its expressions are batched behind it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LETTERS = "ABCDEFGHIJ"

BACKBONES = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "internvl3.5-2b": "OpenGVLab/InternVL3_5-2B-HF",
}


@dataclass
class Encoded:
    vis: torch.Tensor  # [N, d] image-token states (last layer)
    vis_xy: torch.Tensor  # [N, 2] token centres, fractions of width/height
    h: torch.Tensor  # [B, d] state of the last prompt token, one per text


class VLMAdapter:
    def __init__(self, name: str, device: str = "cuda", max_pixels: int = 640 * 640, model=None):
        self.name, self.repo, self.device = name, BACKBONES[name], device
        self.processor = AutoProcessor.from_pretrained(self.repo)
        # `model` lets a training script reuse its (e.g. LoRA-wrapped) model for the decision probe.
        self.model = model if model is not None else AutoModelForImageTextToText.from_pretrained(
            self.repo, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()
        self.tok = self.processor.tokenizer
        self.is_qwen = "qwen" in name
        self.set_resolution(max_pixels)
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index")
        self.hidden = cfg.get_text_config().hidden_size

    def set_resolution(self, max_pixels: int) -> None:
        ip = self.processor.image_processor
        if self.is_qwen:
            # Keep aspect ratio; let images scale up or down to this pixel budget.
            ip.size = {"shortest_edge": 32 * 32, "longest_edge": max_pixels}
            ip.max_pixels, ip.min_pixels = max_pixels, 32 * 32
        # InternVL: one 448 tile (fixed token grid) up to 448², dynamic tiling above
        # (latency at 1024 px). Tiled token grids are not supported by _vis_xy.
        tiles = math.ceil(max_pixels / 448**2)
        self.proc_kwargs = {} if self.is_qwen else (
            {"crop_to_patches": False} if tiles <= 1 else {"crop_to_patches": True, "max_patches": tiles})

    # -- prompt building ---------------------------------------------------
    def _messages(self, text: str) -> list[dict]:
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    def template(self, text: str, generation_prompt: bool) -> str:
        # Thinking off everywhere: Qwen3.5-4B defaults to an open <think> block, Qwen3.5-2B to an
        # empty one; every answer here is one pass, so all models end on the same template.
        return self.processor.apply_chat_template(
            self._messages(text), add_generation_prompt=generation_prompt, tokenize=False, enable_thinking=False)

    def _inputs(self, image: Image.Image, texts: list[str], generation_prompt: bool, suffix: str = ""):
        prompts = [
            self.template(t, generation_prompt) + suffix
            for t in texts
        ]
        self.tok.padding_side = "left"
        return self.processor(images=[image] * len(texts), text=prompts, return_tensors="pt", padding=True, **self.proc_kwargs).to(self.device)

    def _vis_xy(self, inputs) -> torch.Tensor:
        if self.is_qwen:
            t, gh, gw = inputs["image_grid_thw"][0].tolist()
            m = self.model.config.vision_config.spatial_merge_size
            gh, gw = gh // m, gw // m
        else:
            n = int((inputs["input_ids"][0] == self.image_token_id).sum())
            gh = gw = int(round(n**0.5))
        ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        return torch.stack([(xs.flatten() + 0.5) / gw, (ys.flatten() + 0.5) / gh], -1).float()

    # -- probes ------------------------------------------------------------
    @torch.no_grad()
    def encode(self, image: Image.Image, texts: list[str], layer: int = -1) -> Encoded:
        inputs = self._inputs(image, texts, generation_prompt=True)
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer]
        mask = inputs["input_ids"][0] == self.image_token_id
        vis = hs[0, mask]
        xy = self._vis_xy(inputs)
        assert vis.shape[0] == xy.shape[0], (vis.shape, xy.shape)
        # left padding → the last position is the last real token for every row
        return Encoded(vis=vis.float().cpu(), vis_xy=xy, h=hs[:, -1].float().cpu())

    @torch.no_grad()
    def option_logprobs(self, image: Image.Image, question: str, options: list[str]) -> torch.Tensor:
        letters = LETTERS[: len(options)]
        text = question + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
        text += "\nAnswer with the option's letter from the given choices directly."
        prompt = self.template(text, True)
        inputs = self.processor(images=[image], text=[prompt], return_tensors="pt", **self.proc_kwargs).to(self.device)
        logits = self.model(**inputs, use_cache=False).logits[0, -1].float()
        ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in letters]
        return torch.log_softmax(logits[ids], -1).cpu()

    @torch.no_grad()
    def prefill(self, image: Image.Image, text: str) -> int:
        inputs = self._inputs(image, [text], generation_prompt=True)
        self.model(**inputs, use_cache=True)
        return int(inputs["input_ids"].shape[1])
