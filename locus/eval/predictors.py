"""Baseline predictors behind one interface, shared by accuracy (eval_baselines) and latency runs.

  prep(images, exprs)  CPU preprocessing        -> inputs
  to_device(inputs)    H2D                      -> inputs
  run(inputs)          model execution only     -> raw
  decode(raw, inputs)  parse / box conversion   -> Pred

Boxes are xyxy fractions of the original image; NaN marks an invalid output.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image


NAN4 = np.full(4, np.nan)

# Jag's "Base" prompt: JSON box in 0-1000 coordinates (data/external/jag_repo/jag/base_predict.py).
JAG_BASE_PROMPT = (
    "Locate the object described by: {}\n"
    'Return only one JSON object with the key "bbox" and four numbers: {{"bbox": [x1, y1, x2, y2]}}. '
    "The coordinates are the top-left and bottom-right corners of the object's bounding box, "
    "normalized to the range 0 to 1000 relative to the original image. "
    "Do not include explanations or any other objects.")


@dataclass
class Pred:
    box: np.ndarray  # [B, 4] xyxy fractions, NaN row = invalid
    conf: np.ndarray | None = None
    text: list[str] | None = None


def parse_strict(text: str) -> np.ndarray:
    """One JSON object (optionally fenced) with `bbox`/`bbox_2d` of four 0-1000 numbers; anything else is invalid."""
    try:
        s = text.strip()
        m = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", s, re.IGNORECASE)
        obj = json.loads(m.group(1).strip() if m else s)
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            obj = obj[0]
        if isinstance(obj, dict):
            keys = set(obj) & {"bbox", "bbox_2d"}
            if len(keys) != 1 or set(obj) - keys - {"label"}:
                return NAN4
            obj = obj[keys.pop()]
        if not isinstance(obj, list) or len(obj) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in obj):
            return NAN4
        if not all(math.isfinite(v) for v in obj) or obj[2] < obj[0] or obj[3] < obj[1]:
            return NAN4
        return np.clip(np.array(obj, float) / 1000, 0, 1)
    except (ValueError, TypeError, OverflowError):
        return NAN4


class JagPredictor:
    """Jag re-implementation: one forward pass, box from the head on the last token."""

    def __init__(self, px: int = 262144, dtype=torch.float32, device="cuda"):
        from transformers import AutoProcessor
        from locus.models.jag_ref import JAG_DIR, PROMPT, JagRef  # transformers 5 (Qwen3.5); lazy for the LocateAnything env

        self.device, self.px, self.prompt = device, px, PROMPT
        self.model = JagRef.load(dtype=dtype, device=device)
        self.processor = AutoProcessor.from_pretrained(JAG_DIR)
        self.processor.image_processor.size = {"shortest_edge": min(65536, px), "longest_edge": px}
        self.processor.tokenizer.padding_side = "left"
        self.dtype = dtype

    def prep(self, images, exprs):
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt.format(expression=e.strip())}]}] for e in exprs]
        text = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        return self.processor(images=images, text=text, return_tensors="pt", padding=True,
                              **({"device": self.prep_device} if getattr(self, "prep_device", None) else {}))

    def to_device(self, inputs):
        return inputs.to(self.device)

    def enable_graphs(self):
        """Batch-1 CUDA-graphed forward (graphed.GraphedJag), the same engineering as our --graph-prefill."""
        from locus.models.graphed import GraphedJag
        self.graphed = GraphedJag(self.model)

    @torch.inference_mode()
    def run(self, inputs):
        with torch.autocast("cuda", torch.bfloat16):  # fp32 weights + bf16 autocast = Jag's quality config
            if getattr(self, "graphed", None) is not None and inputs["input_ids"].shape[0] == 1:
                return self.graphed(inputs)
            return self.model(**inputs)

    def decode(self, raw, inputs) -> Pred:
        cx, cy, w, h = raw.float().cpu().numpy().T
        box = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).clip(0, 1)
        return Pred(box)


class GenPredictor:
    """Autoregressive coordinate-token VLM: greedy generate with KV cache, parse the text into a box."""

    def __init__(self, name: str, prompt: str, parse: Callable[[str], np.ndarray], px: int, max_new: int = 64,
                 min_px: int | None = None, device="cuda"):
        from locus.models.backbone.adapters import VLMAdapter
        self.ad = VLMAdapter(name, device, max_pixels=px)
        if min_px is not None:  # Jag's processor floor, for exact base reproduction
            self.ad.processor.image_processor.size = {"shortest_edge": min_px, "longest_edge": px}
        self.ad.tok.padding_side = "left"
        self.prompt, self.parse, self.max_new, self.device, self.px = prompt, parse, max_new, device, px
        self.scores = True  # token probabilities for the confidence; off for latency runs
        self.digit_ids = torch.tensor([i for t, i in self.ad.tok.get_vocab().items() if t.isdigit() and len(t) == 1], device=device)

    def prep(self, images, exprs):
        text = [self.ad.template(self.prompt.format(e.strip()), True) for e in exprs]
        kw = dict(self.ad.proc_kwargs, **({"device": self.prep_device} if getattr(self, "prep_device", None) else {}))
        return self.ad.processor(images=images, text=text, return_tensors="pt", padding=True, **kw)

    def to_device(self, inputs):
        return inputs.to(self.device)

    @torch.inference_mode()
    def run(self, inputs):
        return self.ad.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False, use_cache=True,
                                      return_dict_in_generate=True, output_scores=self.scores)

    def decode(self, raw, inputs) -> Pred:
        seq = raw.sequences[:, inputs["input_ids"].shape[1]:]
        texts = self.ad.tok.batch_decode(seq, skip_special_tokens=True)
        boxes = np.stack([self.parse(t) for t in texts])
        if not self.scores:
            return Pred(boxes, None, texts)
        # Confidence: mean probability of the generated digit tokens (the coordinate tokens).
        T = len(raw.scores)
        p = torch.stack([torch.softmax(s.float(), -1).gather(1, seq[:, t, None])[:, 0] for t, s in enumerate(raw.scores)], 1)
        is_digit = torch.isin(seq[:, :T], self.digit_ids)
        conf = ((p * is_digit).sum(1) / is_digit.sum(1).clamp_min(1)).cpu().numpy()
        return Pred(boxes, conf, texts)


LA_PROMPT = "Locate a single instance that matches the following description: {}."  # model card, ground_single


def _la_segment_attn():
    """Without flash-attn, LocateAnything's vision tower builds a dense [L, L] bool mask for SDPA, which forces the math
    kernel: about 40 GB at its native 25600-patch cap. Attending within each image's segment with no mask is the same
    computation (the mask is block-diagonal over cu_seqlens) on SDPA's memory-efficient kernels."""
    import sys
    import torch.nn.functional as F
    mods = [m for n, m in sys.modules.items() if n.endswith("modeling_vit") and hasattr(m, "VL_VISION_ATTENTION_FUNCTIONS")]
    assert mods, "load the model before patching"

    def sdpa(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
        cu = q_cu_seqlens.tolist()
        out = [F.scaled_dot_product_attention(*(t[a:b].transpose(0, 1)[None] for t in (q, k, v)))[0].transpose(0, 1)
               for a, b in zip(cu[:-1], cu[1:])]
        return torch.cat(out).reshape(q.shape[0], -1)

    for m in mods:
        m.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = m.sdpa_attention = sdpa


class LocateAnythingPredictor:
    """nvidia/LocateAnything-3B (NVIDIA non-commercial licence; research evaluation only), loaded from a local snapshot
    with its own remote code. Model-card prompt for single-instance grounding, native resolution, hybrid parallel box
    decoding, greedy (temperature 0, no repetition penalty). Its generate supports batch 1 only. Output
    "<box><x1><y1><x2><y2></box>" in 0-1000 coordinates; the first box is the answer."""

    def __init__(self, path: str, mode: str = "hybrid", max_new: int = 64, device="cuda", px: int | None = None,
                 segment_attn: bool = False):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        if px:  # its processor downscales (aspect kept) to at most in_token_limit 14x14 patches; native is 25600
            self.processor.image_processor.in_token_limit = px // 14**2
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        if segment_attn:
            _la_segment_attn()
        self.mode, self.max_new, self.device, self.px = mode, max_new, device, px or "native"

    def prep(self, images, exprs):
        assert len(images) == 1, "LocateAnything generate supports batch 1 only"
        msgs = [{"role": "user", "content": [{"type": "image", "image": images[0]},
                                             {"type": "text", "text": LA_PROMPT.format(exprs[0].strip())}]}]
        text = self.processor.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ims, vids = self.processor.process_vision_info(msgs)
        return self.processor(text=[text], images=ims, videos=vids, return_tensors="pt")

    def to_device(self, inputs):
        return inputs.to(self.device)

    @torch.inference_mode()
    def run(self, inputs):
        out = self.model.generate(pixel_values=inputs["pixel_values"].to(torch.bfloat16), input_ids=inputs["input_ids"],
                                  attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
                                  tokenizer=self.tok, max_new_tokens=self.max_new, use_cache=True,
                                  generation_mode=self.mode, temperature=0, repetition_penalty=1.0)
        return out[0] if isinstance(out, tuple) else out

    def decode(self, raw, inputs) -> Pred:
        text = raw if isinstance(raw, str) else raw[0]
        m = re.search(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text)
        box = NAN4 if m is None else np.clip(np.array([float(v) / 1000 for v in m.groups()]), 0, 1)
        return Pred(box[None], None, [text])


HT_PROMPT = "Please provide the bounding box coordinate of the region this sentence describes: {}."  # model card / paper
HT_DIGIT = re.compile(r"<([xy])_(hundreds|tens|ones)_(\d)>")


def parse_hitoken(text: str) -> np.ndarray:
    """Hi-Token's 12 axis-digit tokens, x_min y_min x_max y_max, each 100*hundreds + 10*tens + ones in 0-999; the
    model card normalises by 999. Anything but exactly 4 complete x/y/x/y triplets is invalid."""
    toks = HT_DIGIT.findall(text)
    if len(toks) < 12:
        return NAN4
    vals = []
    for k, axis in enumerate("xyxy"):
        t = toks[3 * k: 3 * k + 3]
        if [a for a, _, _ in t] != [axis] * 3 or [p for _, p, _ in t] != ["hundreds", "tens", "ones"]:
            return NAN4
        vals.append(100 * int(t[0][2]) + 10 * int(t[1][2]) + int(t[2][2]))
    b = np.array(vals, float) / 999
    return NAN4 if b[2] < b[0] or b[3] < b[1] else np.clip(b, 0, 1)


class HiTokenPredictor:
    """xyzzzh/Hi-Token (Hi-R1, Qwen2.5-VL-3B; the model card states no licence; research evaluation only), loaded from
    a local snapshot. Model-card prompt and inference: default processor (native resolution), bf16, greedy, decoded with
    special tokens kept."""

    def __init__(self, path: str, max_new: int = 128, device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(path)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        self.max_new, self.device, self.px = max_new, device, "native"

    def prep(self, images, exprs):
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": HT_PROMPT.format(e.strip())}]}] for e in exprs]
        text = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        return self.processor(text=text, images=images, padding=True, return_tensors="pt")

    def to_device(self, inputs):
        return inputs.to(self.device)

    @torch.inference_mode()
    def run(self, inputs):
        return self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False, use_cache=True,
                                   return_dict_in_generate=True)

    def decode(self, raw, inputs) -> Pred:
        seq = raw.sequences[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(seq, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return Pred(np.stack([parse_hitoken(t) for t in texts]), None, texts)
