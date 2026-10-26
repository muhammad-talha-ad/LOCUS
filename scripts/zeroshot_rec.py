"""Zero-shot coordinate-token grounding on RefCOCO val (bake-off reference for §2.3).

Asks the backbone for a box as text (Qwen convention: 0–1000 relative
coordinates), parses it, and reports Acc@0.5 plus generate latency per question.
Merges into results/bakeoff/<backbone>.json under "zeroshot_rec".
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from locus.data import refcoco
from locus.models.backbone.adapters import BACKBONES, VLMAdapter
from locus.spec.box import Box, iou

REPO = Path(__file__).resolve().parents[1]
# Each model's native grounding prompt; both answer in 0–1000 relative coordinates.
PROMPTS = {"internvl": "Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>"}
QWEN_PROMPT = 'Locate "{}" in the image and output its bbox coordinates in JSON format.'


def parse(text: str) -> Box | None:
    # First bracketed list of four numbers (keys like "bbox_2d" contain digits too).
    m = re.search(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", text)
    if not m:
        return None
    x0, y0, x1, y1 = (min(max(float(v) / 1000, 0.0), 1.0) for v in m.groups())
    return Box(top=min(y0, y1), bottom=max(y0, y1), left=min(x0, x1), right=max(x0, x1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--px", type=int, default=640)
    args = ap.parse_args()
    ad = VLMAdapter(args.backbone, "cuda", max_pixels=args.px * args.px)
    prompt_t = next((p for k, p in PROMPTS.items() if args.backbone.startswith(k)), QWEN_PROMPT)
    exprs = [e for g in refcoco.sample_images(refcoco.load("refcoco", "validation"), 10_000, seed=1).values() for e in g[:1]]
    exprs = exprs[: args.n]
    ious, ts, fails, raw = [], [], 0, []
    for e in exprs:
        img = Image.open(e.image_path).convert("RGB")
        prompt = ad.template(prompt_t.format(e.text), True)
        inputs = ad.processor(images=[img], text=[prompt], return_tensors="pt", **ad.proc_kwargs).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = ad.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        text = ad.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        b = parse(text)
        gt = Box(top=e.box[1], bottom=e.box[3], left=e.box[0], right=e.box[2])
        fails += b is None
        ious.append(iou(b, gt) if b else 0.0)
        raw.append(dict(image_id=e.image_id, text=e.text, output=text, iou=ious[-1]))
    ious = np.array(ious)
    r = dict(acc50=float((ious >= 0.5).mean()), miou=float(ious.mean()), n=len(ious), parse_fail=fails,
             gen_ms_median=1000 * float(np.median(ts)), px=args.px, prompt=prompt_t, coords="0-1000 relative")
    out = REPO / "results/bakeoff" / f"{args.backbone}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    res["zeroshot_rec"] = r
    out.write_text(json.dumps(res, indent=1))
    (REPO / "results/bakeoff" / f"raw_zeroshot_{args.backbone}.jsonl").write_text("\n".join(json.dumps(x) for x in raw))
    print("zeroshot_rec", r)


if __name__ == "__main__":
    main()
