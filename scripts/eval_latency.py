"""Batch-1 latency (locus/eval/latency.py) on 10 warmup + 256 RefCOCO testA rows.

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_latency.py --model jag --px 262144
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_latency.py --model jag --px 1048576 --fit
    CUDA_VISIBLE_DEVICES=0 envs/la/bin/python scripts/eval_latency.py --model locateanything-3b --per-image  # native resolution

Merges into results/baselines/latency.json under "<model>@<px>". Run with nothing else on the GPU.
"""
from __future__ import annotations

import argparse
import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from eval_baselines import ZEROSHOT, build
from locus.data import refcoco
from locus.eval import latency

REPO = Path(__file__).resolve().parents[1]


def latency_q(pred, args):
    """Jag has no shared prefill: Q questions on one image are Q requests. Both ways of serving them, end to end from
    the image file: Q sequential graphed batch-1 requests (image opened once), or one eager batch of Q image copies."""
    import numpy as np
    from eval_draft_verify import q_images
    groups = q_images(refcoco.load("refcoco", "test"), max(args.q), args.n, args.warmup)
    t0 = time.perf_counter()
    for c in groups:  # capture every graph the timed requests need
        img = Image.open(c[0].image_path).convert("RGB")
        for e in c:
            pred.run(pred.to_device(pred.prep([img], [e.text])))
    torch.cuda.synchronize()
    res = dict(images=len(groups) - args.warmup, qs=args.q, graphs=dict(n=len(pred.graphed.graphs), capture_s=time.perf_counter() - t0))
    for q in args.q:
        for how in ("sequential", "batched"):
            ts = []
            for j, c in enumerate(groups):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                img = Image.open(c[0].image_path).convert("RGB")
                if how == "sequential":
                    for e in c[:q]:
                        pred.run(pred.to_device(pred.prep([img], [e.text])))
                else:
                    pred.run(pred.to_device(pred.prep([img] * q, [e.text for e in c[:q]])))
                torch.cuda.synchronize()
                if j >= args.warmup:
                    ts.append((time.perf_counter() - t0) * 1000)
            ts = np.array(ts)
            res[f"q{q}_{how}"] = dict(mean_ms=float(ts.mean()), median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)),
                                      per_question_ms=float(ts.mean() / q))
            print("latency", q, how, res[f"q{q}_{how}"], flush=True)
    res.update(model=args.model, px=args.px, gpu=torch.cuda.get_device_name(), gpu_preprocess=args.gpu_preprocess,
               date=datetime.now(timezone.utc).isoformat(timespec="seconds"), torch=torch.__version__)
    out = REPO / "results/baselines/latency_q.json"
    out.write_text(json.dumps({f"jag@{args.px}{'+gpupre' if args.gpu_preprocess else ''}+graphs": res}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["jag", "qwen3.5-0.8b-base", "locateanything-3b", "hi-token", *ZEROSHOT])
    ap.add_argument("--px", type=int, default=262144)
    ap.add_argument("--fit", action="store_true", help="resize each image to --px area first (use for 1024^2)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-preprocess", action="store_true", help="image preprocessing on the GPU (same option as ours)")
    ap.add_argument("--per-image", action="store_true", help="first expression of each distinct image (the rows "
                    "scripts/eval_draft_verify.py --latency --split refcoco_testA times)")
    ap.add_argument("--q", type=int, nargs="+", default=None, help="jag only: Q questions per image, for each Q given, on "
                    "the images eval_draft_verify.py --latency-q uses; Q graphed batch-1 requests, and one eager batch of Q")
    ap.add_argument("--graphs", action="store_true", help="jag only: batch-1 CUDA-graphed forward (same as our --graph-prefill)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    # Jag's cost config: bf16 backbone weights, fp32 head.
    pred = build(args.model, args.px, "bf16")
    if pred.px == "native":  # LocateAnything, Hi-Token: their own processors at native resolution, as in accuracy runs
        args.px = "native"
    if hasattr(pred, "scores"):
        pred.scores = False  # confidence needs token scores; latency of the plain generate path
    if args.gpu_preprocess:
        pred.prep_device = "cuda"
    rows = refcoco.load("refcoco", "test")
    if args.per_image:
        rows = list({e.image_id: e for e in rows}.values())
    rows = rows[: args.warmup + args.n]
    pre = {}
    if args.graphs:  # capture every (grid, prompt length) graph up front, as our latency path does
        assert args.model == "jag" and not args.fit
        pred.enable_graphs()
        t0 = time.perf_counter()
        for e in rows:
            pred.run(pred.to_device(pred.prep([Image.open(e.image_path).convert("RGB")], [e.text])))
        torch.cuda.synchronize()
        pre = dict(n=len(pred.graphed.graphs), capture_s=time.perf_counter() - t0, mem_gib=torch.cuda.memory_reserved() / 2**30)
        print("graphs", pre, flush=True)
    if args.q:
        assert args.model == "jag" and args.graphs
        return latency_q(pred, args)
    r = latency.measure(pred, rows, args.warmup, args.n, fit_px=args.px if args.fit else None)
    if pre:
        r["graphs"] = pre
    r.update(model=args.model, px=args.px, split=("refcoco testA first %d images" if args.per_image else "refcoco testA first %d rows") % len(rows), seed=args.seed, gpu=torch.cuda.get_device_name(),
             date=datetime.now(timezone.utc).isoformat(timespec="seconds"), torch=torch.__version__,
             causal_conv1d_reference_path=args.model.startswith("qwen3.5") or args.model == "jag",
             git_commit=subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None)
    out = REPO / "results/baselines/latency.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    r["gpu_preprocess"] = args.gpu_preprocess
    res[f"{args.model}@{args.px}{'fit' if args.fit else ''}{'+gpupre' if args.gpu_preprocess else ''}{'+graphs' if args.graphs else ''}{'+perimage' if args.per_image else ''}"] = r
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
