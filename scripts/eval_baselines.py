"""Evaluate baselines on RefCOCO/+/g with one metrics function (locus/eval/grounding.py).

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_baselines.py --model jag
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_baselines.py --model qwen3-vl-2b --limit 3000

Models: jag (our re-implementation, released weights), locateanything-3b and hi-token (local snapshots, model-card
prompts), qwen3.5-0.8b-base (Jag's "Base" prompt and strict
parser), and zero-shot coordinate-token qwen3-vl-{2b,4b} / qwen3.5-{2b,4b} (scripts/zeroshot_rec.py prompt
and parser). Writes results/baselines/<model>.json and raw_<model>_<split>.npz.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from locus.data import refcoco
from locus.data.refcoco import EVAL_SPLITS  # same split table as the training script
from locus.eval import grounding
from locus.eval.predictors import JAG_BASE_PROMPT, GenPredictor, JagPredictor, LocateAnythingPredictor, HiTokenPredictor, parse_strict
from zeroshot_rec import QWEN_PROMPT, parse as parse_text_box

REPO = Path(__file__).resolve().parents[1]
JAG_SPLITS = ["refcoco_testA", "refcoco_testB", "refcoco+_testA", "refcoco+_testB", "refcocog_test"]  # Jag's published five
ZEROSHOT = ["qwen3-vl-2b", "qwen3-vl-4b", "qwen3.5-2b", "qwen3.5-4b"]
LA_PATH = Path.home() / "hf-models/LocateAnything-3B"  # local snapshot of nvidia/LocateAnything-3B (research use only)
HT_PATH = Path.home() / "hf-models/Hi-Token"  # local snapshot of xyzzzh/Hi-Token (no licence stated; research use only)


def parse_coord(text: str) -> np.ndarray:
    b = parse_text_box(text)
    return np.full(4, np.nan) if b is None else np.array([b.left, b.top, b.right, b.bottom])


def build(model: str, px: int | None, dtype: str, segment_attn: bool = False):
    if model == "jag":
        return JagPredictor(px or 262144, dtype=dict(fp32=torch.float32, bf16=torch.bfloat16)[dtype])
    if model == "qwen3.5-0.8b-base":
        return GenPredictor("qwen3.5-0.8b", JAG_BASE_PROMPT, parse_strict, px or 262144, max_new=128, min_px=65536)
    if model == "locateanything-3b":
        return LocateAnythingPredictor(str(LA_PATH), px=px, segment_attn=segment_attn)
    if model == "hi-token":
        return HiTokenPredictor(str(HT_PATH))
    return GenPredictor(model, QWEN_PROMPT, parse_coord, px or 640 * 640, max_new=64)


class Rows(Dataset):
    def __init__(self, exprs):
        self.exprs = exprs

    def __len__(self):
        return len(self.exprs)

    def __getitem__(self, i):
        e = self.exprs[i]
        return Image.open(e.image_path).convert("RGB"), e.text


def evaluate(pred, exprs, bs: int, workers: int) -> dict:
    loader = DataLoader(Rows(exprs), batch_size=bs, num_workers=workers,
                        collate_fn=lambda b: pred.prep([x[0] for x in b], [x[1] for x in b]))
    boxes, confs, texts, t0 = [], [], [], time.perf_counter()
    for inputs in loader:
        inputs = pred.to_device(inputs)
        p = pred.decode(pred.run(inputs), inputs)
        boxes.append(p.box)
        if p.conf is not None:
            confs.append(p.conf)
        texts += p.text or []
        if len(boxes) % 50 == 0:
            print(f"  {sum(map(len, boxes))}/{len(exprs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    return dict(box=np.concatenate(boxes), conf=np.concatenate(confs) if confs else None, texts=texts,
                seconds=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["jag", "qwen3.5-0.8b-base", "locateanything-3b", "hi-token", *ZEROSHOT])
    ap.add_argument("--splits", nargs="*", help="default: Jag's five test splits (jag, base) or RefCOCO val/testA/testB")
    ap.add_argument("--limit", type=int, default=0, help="first N rows of each split (0 = full)")
    ap.add_argument("--px", type=int, help="image pixel budget (default 262144 for jag/base, 640^2 for zero-shot)")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="jag backbone weights")
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", help="result name (default: model)")
    ap.add_argument("--la-segment-attn", action="store_true",
                    help="locateanything-3b: maskless per-image vision attention (same math; fits native-cap images)")
    args = ap.parse_args()
    name = args.name or args.model
    splits = args.splits or (list(EVAL_SPLITS) if args.model in ("jag", "locateanything-3b", "hi-token") else
                             JAG_SPLITS if args.model == "qwen3.5-0.8b-base" else ["refcoco_val", "refcoco_testA", "refcoco_testB"])
    bs = args.bs or (32 if args.model == "jag" else 1 if args.model == "locateanything-3b" else 16)
    torch.manual_seed(args.seed)

    out = REPO / "results/baselines" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    res.update(name=name, model=args.model, config=vars(args), seed=args.seed,
               date=datetime.now(timezone.utc).isoformat(timespec="seconds"), gpu=torch.cuda.get_device_name(),
               git_commit=subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None)
    res.setdefault("splits", {})
    pred = build(args.model, args.px, args.dtype, args.la_segment_attn)
    res["px"], res["batch_size"] = pred.px, bs
    for sp in splits:
        ds, part = refcoco.ALL_SPLITS[sp]
        exprs = refcoco.load(ds, part)
        if args.limit:
            exprs = exprs[: args.limit]
        print(sp, len(exprs), flush=True)
        o = evaluate(pred, exprs, bs, args.workers)
        gt = np.array([e.box for e in exprs])
        iou = grounding.iou_xyxy(o["box"], gt)
        invalid = np.isnan(o["box"]).any(1)
        ann = [e.ann_id for e in exprs]
        r = grounding.report(iou, o["conf"], ann, invalid)
        r.update(rows=len(exprs), full_split=not args.limit or args.limit >= len(refcoco.load(ds, part)), seconds=o["seconds"])
        res["splits"][sp] = r
        grounding.save_raw(REPO / "results/baselines" / f"raw_{name}_{sp}.npz", o["box"], iou, ann, [e.image_id for e in exprs], o["conf"])
        if o["texts"]:
            (REPO / "results/baselines" / f"raw_{name}_{sp}.jsonl").write_text("\n".join(json.dumps(dict(text=e.text, output=t)) for e, t in zip(exprs, o["texts"])))
        out.write_text(json.dumps(res, indent=1))
        a = r["all"]
        print(f"{sp}: Acc@0.5 {100 * a['acc50']:.2f}  mIoU {100 * a['miou']:.2f}  invalid {a['invalid']}  "
              f"clean {100 * r['clean']['acc50']:.2f} ({r['clean']['n']})  leaked {100 * r['leaked']['acc50']:.2f} ({r['leaked']['n']})", flush=True)


if __name__ == "__main__":
    main()
