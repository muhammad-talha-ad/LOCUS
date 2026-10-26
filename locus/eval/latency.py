"""Jag-comparable latency protocol, batch 1, nothing else on the GPU.

End-to-end = CUDA-synchronised timer over image open, resize, preprocessing, H2D, model execution and
decode/parse. Model-only = `predictor.run` on inputs already on the GPU. Load and warmup excluded.
`fit_px` resizes each image to that pixel area first (inside the timed region), so a 1024^2 run is a
1024^2 image rather than a 640 px RefCOCO image under a larger budget.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from PIL import Image


def fit_area(img: Image.Image, area: int) -> Image.Image:
    s = math.sqrt(area / (img.width * img.height))
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)


def stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)), n=len(a))


def measure(predictor, rows, warmup: int = 10, n: int = 256, fit_px: int | None = None) -> dict:
    """`rows` are RefExpr; the first `warmup` rows are untimed, the next `n` timed."""
    sync = torch.cuda.synchronize

    def e2e(e):
        img = Image.open(e.image_path).convert("RGB")
        if fit_px:
            img = fit_area(img, fit_px)
        inputs = predictor.to_device(predictor.prep([img], [e.text]))
        pred = predictor.decode(predictor.run(inputs), inputs)
        return pred, inputs

    for e in rows[:warmup]:
        e2e(e)
    timed = rows[warmup : warmup + n]
    torch.cuda.reset_peak_memory_stats()
    ms, tokens, out_tokens = [], [], []
    for e in timed:
        sync()
        t0 = time.perf_counter()
        pred, inputs = e2e(e)
        sync()
        ms.append(1000 * (time.perf_counter() - t0))
        tokens.append(int(inputs["input_ids"].shape[1]))
    peak = torch.cuda.max_memory_allocated() / 2**30
    model_ms = []
    for e in timed:  # model-only on prepared GPU inputs (prefill for Jag, prefill+decode loop for generate)
        img = Image.open(e.image_path).convert("RGB")
        if fit_px:
            img = fit_area(img, fit_px)
        inputs = predictor.to_device(predictor.prep([img], [e.text]))
        sync()
        t0 = time.perf_counter()
        raw = predictor.run(inputs)
        sync()
        model_ms.append(1000 * (time.perf_counter() - t0))
        if hasattr(raw, "sequences"):
            out_tokens.append(int(raw.sequences.shape[1] - inputs["input_ids"].shape[1]))
    r = dict(end_to_end=stats(ms), model_only=stats(model_ms), peak_alloc_gib=float(peak), warmup=warmup,
             mean_prompt_tokens=float(np.mean(tokens)), fit_px=fit_px)
    if out_tokens:
        r["mean_new_tokens"] = float(np.mean(out_tokens))
    return r
