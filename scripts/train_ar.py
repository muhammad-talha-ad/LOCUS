"""Coordinate-token control: same backbone, data, LoRA and budget as train_box.py,
but the box is generated as text, "[x0, y0, x1, y1]" in Qwen's 0–1000 relative coordinates.

Confidence for calibration comparisons = mean probability of the generated tokens (the natural
token-likelihood confidence an AR grounding model exposes).

    python scripts/train_ar.py --backbone qwen3-vl-2b --steps 12000 --name ar_qwen3-vl-2b_lora64
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from train_box import EVAL_SPLITS, Rows, auroc, ece, train_set  # noqa: E402
from zeroshot_rec import QWEN_PROMPT, parse  # noqa: E402

from locus.data import refcoco  # noqa: E402
from locus.models.backbone.adapters import BACKBONES, VLMAdapter  # noqa: E402
from locus.models.boxvlm import LORA_TARGETS  # noqa: E402
from locus.spec.box import Box, iou  # noqa: E402

END = "<|im_end|>"


def answer(box) -> str:
    return "[" + ", ".join(str(round(v * 1000)) for v in box) + "]"


class CollateAR:
    def __init__(self, ad: VLMAdapter, train: bool):
        self.ad, self.train = ad, train

    def __call__(self, rows):
        imgs, texts, boxes = zip(*rows)
        tok = self.ad.tok
        prompts = [self.ad.template(QWEN_PROMPT.format(t), True) for t in texts]
        if not self.train:
            tok.padding_side = "left"
            return self.ad.processor(images=list(imgs), text=prompts, return_tensors="pt", padding=True,
                                     **self.ad.proc_kwargs), torch.stack(boxes)
        ans = [answer(b.tolist()) + END for b in boxes]
        tok.padding_side = "left"  # answers end every row → only the last K positions need logits
        inputs = self.ad.processor(images=list(imgs), text=[p + a for p, a in zip(prompts, ans)], return_tensors="pt",
                                   padding=True, **self.ad.proc_kwargs)
        n = [len(tok.encode(a, add_special_tokens=False)) for a in ans]
        K = max(n) + 1
        labels = inputs["input_ids"][:, -K:].clone()
        for i, k in enumerate(n):
            labels[i, : K - k] = -100
        inputs["labels"] = labels  # [B, K]; position j predicts labels[:, j] from logits[:, j - 1]
        return inputs, torch.stack(boxes)


@torch.no_grad()
def evaluate(ad: VLMAdapter, exprs, bs: int = 16, raw: Path | None = None) -> dict:
    ad.model.eval()
    ious, confs, fails, ts = [], [], 0, []
    end_id = ad.tok.convert_tokens_to_ids(END)
    for inputs, gt in DataLoader(Rows(exprs), batch_size=bs, num_workers=6, collate_fn=CollateAR(ad, False)):
        inputs = inputs.to("cuda")
        t0 = time.perf_counter()
        out = ad.model.generate(**inputs, max_new_tokens=24, do_sample=False, output_scores=True,
                                return_dict_in_generate=True)
        ts.append(time.perf_counter() - t0)
        seq = out.sequences[:, inputs["input_ids"].shape[1]:]
        probs = torch.stack([s.float().softmax(-1).gather(1, seq[:, k : k + 1])[:, 0] for k, s in enumerate(out.scores)], 1)
        for i in range(len(seq)):
            valid = (seq[i] != end_id).cumprod(0).bool() & (seq[i] != ad.tok.pad_token_id)
            b = parse(ad.tok.decode(seq[i], skip_special_tokens=True))
            g = gt[i].tolist()
            fails += b is None
            ious.append(iou(b, Box(top=g[1], bottom=g[3], left=g[0], right=g[2])) if b else 0.0)
            confs.append(float(probs[i][valid].mean()) if valid.any() else 0.0)
    ad.model.train()
    iou_a, conf = np.array(ious), np.array(confs)
    if raw is not None:
        np.savez_compressed(raw, iou=iou_a, conf=conf, ann_id=np.array([e.ann_id for e in exprs]),
                            image_id=np.array([e.image_id for e in exprs]))
    ok = iou_a >= 0.5
    return dict(n=len(iou_a), acc50=float(ok.mean()), acc75=float((iou_a >= 0.75).mean()), acc90=float((iou_a >= 0.9).mean()),
                miou=float(iou_a.mean()), parse_fail=fails, conf_ece=ece(conf, ok), conf_auroc=auroc(conf, ok),
                mean_conf=float(conf.mean()), rows_per_s=len(iou_a) / sum(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--name", required=True)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--px", type=int, default=640)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--final-splits", nargs="*", default=list(EVAL_SPLITS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = REPO / "results/box_train" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = dict(name=args.name, config=vars(args), seed=args.seed, repo=BACKBONES[args.backbone], output="coordinate tokens",
               date=datetime.now(timezone.utc).isoformat(timespec="seconds"), gpu=torch.cuda.get_device_name(),
               git_commit=subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip() or None, curve=[])
    save = lambda: out.write_text(json.dumps(res, indent=1))

    ad = VLMAdapter(args.backbone, "cuda", args.px * args.px)
    for p in ad.model.parameters():
        p.requires_grad_(False)
    ad.model = get_peft_model(ad.model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                   target_modules=rf".*language_model.*\.({LORA_TARGETS})"))
    params = [p for n, p in ad.model.named_parameters() if "lora_" in n]
    res["trainable_params_m"] = {"lora": sum(p.numel() for p in params) / 1e6}
    opt = torch.optim.AdamW(params, lr=args.lr_lora, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    exprs = train_set(args.seed)
    res["train_rows"], res["epochs"] = len(exprs), args.steps * args.bs / len(exprs)
    val_curve = [e for g in refcoco.sample_images(refcoco.load("refcoco", "validation"), 10_000, seed=1).values() for e in g]
    val_curve = val_curve[: args.eval_n]
    save()

    loader = DataLoader(Rows(exprs), batch_size=args.bs, num_workers=args.workers, collate_fn=CollateAR(ad, True),
                        drop_last=True, persistent_workers=True, prefetch_factor=4)
    ad.model.train()
    step, t0, hist = 0, time.time(), []
    while step < args.steps:
        for inputs, _ in loader:
            inputs = inputs.to("cuda")
            labels = inputs.pop("labels")
            with torch.autocast("cuda", torch.bfloat16):
                logits = ad.model(**inputs, use_cache=False, logits_to_keep=labels.shape[1]).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            hist.append(loss.item())
            if step % 50 == 0:
                l = float(np.mean(hist))
                hist = []
                print(f"step {step} loss {l:.3f} {(time.time() - t0) / step:.2f}s/step "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                res["curve"].append(dict(step=step, train_loss=l, elapsed_h=(time.time() - t0) / 3600,
                                         **evaluate(ad, val_curve)))
                print("eval", res["curve"][-1], flush=True)
                save()
            if step >= args.steps:
                break
    res["train_h"] = (time.time() - t0) / 3600
    ckpt = REPO / "data/ckpt" / f"{args.name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in ad.model.named_parameters() if "lora_" in n}, ckpt)
    res["ckpt"], res["final"] = str(ckpt.relative_to(REPO)), {}
    for split in args.final_splits:
        res["final"][split] = evaluate(ad, refcoco.load(*EVAL_SPLITS[split]), raw=out.parent / f"raw_{args.name}_{split}.npz")
        print(split, res["final"][split], flush=True)
        save()


if __name__ == "__main__":
    main()
