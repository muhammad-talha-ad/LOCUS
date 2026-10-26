"""Train the [BOX] head (S0: frozen backbone, S1-lite: + LoRA) on RefCOCO/+/g train.

Used for the Qwen3-VL-2B vs Qwen3.5-2B LoRA comparison
and the scale check. Train images that appear in any RefCOCO/+/g eval split are
dropped (RefCOCOg-umd and the UNC splits partition COCO differently).

    python scripts/train_box.py --backbone qwen3-vl-2b --lora-r 64 --steps 12000 --name qvl2b_lora

Writes results/box_train/<name>.json (config, commit, seed, curve, final evals)
and data/ckpt/<name>.pt (trainable weights only).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from locus.data import refcoco
from locus.models.backbone.adapters import BACKBONES
from locus.models.boxvlm import BoxVLM
from locus.models.draft_verify import DraftVerify
from locus.models.heads.box_decoder import conf_loss, upgrade_state_dict
from locus.models.heads.probe import box_iou_xyxy, box_loss

REPO = Path(__file__).resolve().parents[1]
EVAL_SPLITS = refcoco.EVAL_SPLITS  # name → (dataset, parquet split)


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def auroc(score, label):
    score, label = np.asarray(score), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    r = score.argsort().argsort() + 1
    return float((r[label].sum() - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum()))


def train_set(seed: int) -> list[refcoco.RefExpr]:
    held = {e.image_id for ds, sp in EVAL_SPLITS.values() for e in refcoco.load(ds, sp)}
    exprs = [e for ds in ("refcoco", "refcocoplus", "refcocog") for e in refcoco.load(ds, "train") if e.image_id not in held]
    random.Random(seed).shuffle(exprs)
    return exprs


class Rows(Dataset):
    def __init__(self, exprs):
        self.exprs = exprs

    def __len__(self):
        return len(self.exprs)

    def __getitem__(self, i):
        e = self.exprs[i]
        return Image.open(e.image_path).convert("RGB"), e.text, torch.tensor(e.box)


class Collate:
    def __init__(self, model: BoxVLM):
        self.processor, self.prompt, self.kw = model.ad.processor, model.prompt, model.ad.proc_kwargs

    def __call__(self, rows):
        imgs, texts, boxes = zip(*rows)
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(images=list(imgs), text=[self.prompt(t) for t in texts], return_tensors="pt",
                                padding=True, **self.kw)
        return inputs, torch.stack(boxes)


class Images(Dataset):
    """Shared-prefill layout: item = one image + up to q of its expressions (resampled each pass)."""

    def __init__(self, exprs, q: int, seed: int, passes: int = 1):
        g = refcoco.by_image(exprs)
        self.groups, self.q, self.seed, self.passes = [g[k] for k in sorted(g)], q, seed, passes
        random.Random(seed).shuffle(self.groups)

    def __len__(self):
        return len(self.groups) * self.passes

    def __getitem__(self, i):
        g = self.groups[i % len(self.groups)]
        es = random.Random(self.seed * 1_000_003 + i).sample(g, min(self.q, len(g))) if self.q else g
        return Image.open(g[0].image_path).convert("RGB"), [e.text for e in es], torch.tensor([e.box for e in es])


class CollateShared:
    def __init__(self, model: BoxVLM):
        self.processor, self.prefix, self.kw = model.ad.processor, model.prefix, model.ad.proc_kwargs
        self.tokenize = model.tokenize_branches

    def __call__(self, items):
        imgs, texts, boxes = zip(*items)
        self.processor.tokenizer.padding_side = "left"
        pre = self.processor(images=list(imgs), text=[self.prefix] * len(imgs), return_tensors="pt", padding=True, **self.kw)
        ids, mask = self.tokenize([t for ts in texts for t in ts])
        row_image = torch.cat([torch.full((len(ts),), i) for i, ts in enumerate(texts)])
        return (pre, ids, mask, row_image, [t for ts in texts for t in ts]), torch.cat(boxes)


def ar_loss(model: BoxVLM, dv: DraftVerify, state: dict, texts, gt, row_image) -> torch.Tensor:
    """Cross-entropy of the ground-truth coordinate tokens in verifier branches over the cached base prefix."""
    prompts, answers = dv.verifier_inputs(texts, gt.float().cpu().numpy())
    seqs = [p + a for p, a in zip(prompts, answers)]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), model.ad.tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, : len(s)], mask[j, : len(s)] = torch.tensor(s), 1
    h = model.branch_pass(state, ids.cuda(), mask.cuda(), row_image, keep_cache=True).last_hidden_state
    sel = torch.cat([h[j, len(p) - 1 : len(p) + len(a) - 1] for j, (p, a) in enumerate(zip(prompts, answers))])
    tgt = torch.cat([torch.tensor(a) for a in answers]).cuda()
    return torch.nn.functional.cross_entropy(model.base.lm_head(sel).float(), tgt)


def run(model: BoxVLM, batch):
    """Forward either layout; batch comes from Collate or CollateShared."""
    if isinstance(batch, tuple):
        pre, ids, mask, row_image = batch[:4]
        return model.forward_shared(pre.to("cuda"), ids.cuda(), mask.cuda(), row_image.cuda())
    return model(batch.to("cuda"))


@torch.no_grad()
def evaluate(model: BoxVLM, exprs, bs: int = 32, workers: int = 6, raw: Path | None = None, shared: bool = False) -> dict:
    model.eval()
    ious, confs, ts = [], [], []
    if shared:  # all expressions of an image as branches of one prefill; keep row order = image-grouped order
        g = refcoco.by_image(exprs)
        exprs = [e for k in g for e in g[k]]
        ds = Images([e for k in g for e in g[k]], 0, 0)
        ds.groups = list(g.values())
        loader = DataLoader(ds, batch_size=max(1, bs // 8), num_workers=workers, collate_fn=CollateShared(model))
    else:
        loader = DataLoader(Rows(exprs), batch_size=bs, num_workers=workers, collate_fn=Collate(model))
    for inputs, gt in loader:
        t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16):
            pred, logit = run(model, inputs)
        ious.append(box_iou_xyxy(pred.float(), gt.cuda())[0].cpu())
        confs.append(torch.sigmoid(logit.float()).cpu())
        ts.append(time.perf_counter() - t0)
    model.train()
    iou, conf = torch.cat(ious).numpy(), torch.cat(confs).numpy()
    if raw is not None:  # per-row outputs, for clean/leaked splits and calibration plots
        np.savez_compressed(raw, iou=iou, conf=conf, ann_id=np.array([e.ann_id for e in exprs]),
                            image_id=np.array([e.image_id for e in exprs]))
    ok = iou >= 0.5
    return dict(n=len(iou), acc50=float(ok.mean()), acc75=float((iou >= 0.75).mean()), acc90=float((iou >= 0.9).mean()),
                miou=float(iou.mean()), conf_ece=ece(conf, ok), conf_auroc=auroc(conf, ok), mean_conf=float(conf.mean()),
                rows_per_s=len(iou) / sum(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--name", required=True)
    ap.add_argument("--lora-r", type=int, default=64, help="0 = frozen backbone (S0)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--px", type=int, default=640)
    ap.add_argument("--mid", type=float, default=0.5, help="decoder reads this fraction of depth + the last layer")
    ap.add_argument("--decoder-layers", type=int, default=3)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=2000, help="RefCOCO val rows for the training curve")
    ap.add_argument("--final-splits", nargs="*", default=list(EVAL_SPLITS))
    ap.add_argument("--decision-items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shared", action="store_true", help="shared prefill: --bs images per step, --q questions each")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--exit-layer", type=float, default=1.0, help="run only this fraction of the LLM's layers")
    ap.add_argument("--n-query", type=int, default=1, help="[BOX] query tokens per question")
    ap.add_argument("--refine", action="store_true", help="box-conditioned iterative refinement + deep supervision")
    ap.add_argument("--train-targets", default="", help="JSONL, one expression per line: image_id, text, path (image "
                    "file; omit for COCO train2017), and the box under --target as [x0, y0, x1, y1] fractions of the "
                    "image size (custom data, or drafter distillation)")
    ap.add_argument("--target", default="greedy", choices=["greedy", "gt"], help="box column of --train-targets")
    ap.add_argument("--init", default="", help="start from this run's checkpoint (data/ckpt/<run>.pt)")
    ap.add_argument("--ar-weight", type=float, default=0.0, help="also train the adapter on coordinate tokens in a "
                    "verifier branch (draft_verify.py) with this loss weight (needs --shared; without --alora the prefix is adapted too)")
    ap.add_argument("--box-weight", type=float, default=1.0, help="weight of the head losses (0 with --ar-weight: verifier-only training, the separate-training ablation)")
    ap.add_argument("--full-ft", action="store_true", help="full fine-tuning of the LM + merger at --lr-lora, fp32 "
                    "master weights (dual mode: needs --lora-r 0, no --alora)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation: --bs images per step in --accum "
                    "micro-batches of --bs/--accum (each micro-batch loss weighted 1/--accum)")
    ap.add_argument("--fused-adam", action="store_true", help="fused AdamW (no foreach temporaries of the full "
                    "parameter set; needed to fit 2B full FT in 48 GB)")
    ap.add_argument("--merger-lora", action="store_true", help="LoRA on the vision merger(s) too (not with --alora)")
    ap.add_argument("--alora", action="store_true", help="Activated LoRA: adapter only on branch tokens (needs --shared)")
    ap.add_argument("--dfl", type=int, default=0, help="bins of the last layer's per-edge offset distribution (needs --refine)")
    ap.add_argument("--head", default="xattn", choices=["xattn", "mlp"], help="mlp = Jag-style last-token control")
    args = ap.parse_args()
    assert args.shared or not args.alora, "--alora needs the shared layout"
    assert not args.ar_weight or args.shared, "--ar-weight needs --shared"
    assert args.box_weight > 0 or args.ar_weight, "--box-weight 0 trains nothing without --ar-weight"

    torch.manual_seed(args.seed)
    out = REPO / "results/box_train" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = dict(name=args.name, config=vars(args), seed=args.seed, repo=BACKBONES[args.backbone],
               date=datetime.now(timezone.utc).isoformat(timespec="seconds"), gpu=torch.cuda.get_device_name(),
               git_commit=subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip() or None, curve=[])
    save = lambda: out.write_text(json.dumps(res, indent=1))

    model = BoxVLM(args.backbone, max_pixels=args.px * args.px, lora_r=args.lora_r, mid=args.mid,
                   decoder_layers=args.decoder_layers, grad_ckpt=args.grad_ckpt, exit_layer=args.exit_layer,
                   n_query=args.n_query, refine=args.refine, head=args.head, dfl=args.dfl,
                   alora=args.alora, merger_lora=args.merger_lora, full_ft=args.full_ft)
    if args.full_ft:
        model.ad.model.float()  # fp32 master weights; compute stays bf16 under autocast
    if args.init:
        sd = torch.load(REPO / "data/ckpt" / f"{args.init}.pt")
        with torch.no_grad():
            model.box_embed.copy_(sd["box_embed"].reshape(model.box_embed.shape))
            model.decoder.load_state_dict(upgrade_state_dict(sd["decoder"]))
            named = dict(model.ad.model.named_parameters())
            for k, v in sd.get("lora", {}).items():
                named[k].copy_(v)
    groups = model.trainable()
    res["trainable_params_m"] = {k: sum(p.numel() for p in v) / 1e6 for k, v in groups.items()}
    res["decoder_reads_layers"] = model.layers
    opt = torch.optim.AdamW([dict(params=groups["head"], lr=args.lr_head)] +
                            ([dict(params=groups["lora"], lr=args.lr_lora)] if groups["lora"] else []), weight_decay=0.01, **(dict(fused=True) if args.fused_adam else {}))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    if args.train_targets:
        exprs = []
        for line in (REPO / args.train_targets).open():
            r = json.loads(line)
            if r[args.target] is not None:
                e = refcoco.RefExpr(r["image_id"], 0, 0, r["text"], tuple(r[args.target]), r.get("ann_id", -1),
                                    r.get("path", ""))
                exprs.append(e)
        res["train_targets"] = dict(file=args.train_targets, target=args.target)
    else:
        exprs = train_set(args.seed)
    res["train_rows"], res["train_images"] = len(exprs), len({e.image_id for e in exprs})
    rows_per_step = args.bs * args.q if args.shared else args.bs  # shared: upper bound (images with < q expressions)
    res["epochs"] = args.steps * rows_per_step / len(exprs)
    val_curve = []  # --eval-n 0: no RefCOCO curve, so custom data trains without RefCOCO on disk
    if args.eval_n:
        val_curve = [e for g in refcoco.sample_images(refcoco.load("refcoco", "validation"), 10_000, seed=1).values()
                     for e in g][: args.eval_n]
    print(f"train rows {len(exprs)}  epochs {res['epochs']:.2f}  trainable {res['trainable_params_m']}", flush=True)
    save()

    if args.shared:
        ds = Images(exprs, args.q, args.seed, passes=math.ceil(args.steps * args.bs / len(refcoco.by_image(exprs))))
        assert args.bs % args.accum == 0
        loader = DataLoader(ds, batch_size=args.bs // args.accum, num_workers=args.workers, collate_fn=CollateShared(model),
                            drop_last=True, persistent_workers=True, prefetch_factor=4)
    else:
        loader = DataLoader(Rows(exprs), batch_size=args.bs, shuffle=False, num_workers=args.workers,
                            collate_fn=Collate(model), drop_last=True, persistent_workers=True, prefetch_factor=4)
    model.train()
    assert args.accum == 1 or args.shared, "--accum needs the shared layout"
    step, micro, t0, hist, hist_ar = 0, 0, time.time(), [], []
    dv = DraftVerify(model) if args.ar_weight else None
    while step < args.steps:
        for inputs, gt in loader:
            gt = gt.cuda()
            with torch.autocast("cuda", torch.bfloat16):
                if args.ar_weight:  # one base prefill; box branches (head) + coordinate-token branches (verifier)
                    pre, ids, mask, ri, texts = inputs
                    ri = ri.cuda()
                    state = model.prefill(pre.to("cuda"))
                    pred, logit = model.forward_shared(None, ids.cuda(), mask.cuda(), ri, state=state)
                    ar = ar_loss(model, dv, state, texts, gt, ri)
                else:
                    pred, logit = run(model, inputs)
            pred = pred.float()
            iou = box_iou_xyxy(pred, gt)[0]
            loss = box_loss(pred, gt) + conf_loss(logit.float(), iou)
            for aux in model.decoder.aux[:-1]:  # deep supervision (refine); the last aux is pred
                loss = loss + box_loss(aux.float(), gt)
            if args.dfl:
                loss = loss + 0.5 * model.decoder.dfl_loss(gt)
            loss = args.box_weight * loss
            if args.ar_weight:
                loss = loss + args.ar_weight * ar
                hist_ar.append(ar.item())
            (loss / args.accum).backward()
            hist.append((loss.item(), float((iou >= 0.5).float().mean())))
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                l, a = np.mean(hist, 0)
                hist = []
                if hist_ar:
                    print(f"step {step} ar_ce {np.mean(hist_ar):.3f}", flush=True)
                    hist_ar = []
                print(f"step {step} loss {l:.3f} train_acc50 {a:.3f} {(time.time() - t0) / step:.2f}s/step "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
            if val_curve and (step % args.eval_every == 0 or step == args.steps):
                ev = evaluate(model, val_curve, shared=args.shared)
                res["curve"].append(dict(step=step, train_loss=float(l), elapsed_h=(time.time() - t0) / 3600, **ev))
                print("eval", res["curve"][-1], flush=True)
                save()
            if step >= args.steps:
                break
    res["train_h"] = (time.time() - t0) / 3600
    ckpt = REPO / "data/ckpt" / f"{args.name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict_trainable(), ckpt)
    res["ckpt"] = str(ckpt.relative_to(REPO))

    res["final"] = {}
    for split in args.final_splits:
        res["final"][split] = evaluate(model, refcoco.load(*EVAL_SPLITS[split]),
                                       raw=out.parent / f"raw_{args.name}_{split}.npz", shared=args.shared)
        print(split, res["final"][split], flush=True)
        save()
    if args.decision_items and args.lora_r:
        # Decision retention: the LoRA'd LM answering A-OKVQA MC through its normal LM head.
        import sys
        sys.path.insert(0, str(REPO / "scripts"))
        from bakeoff import decision_probe
        model.eval()
        if args.alora:  # decision branches never pass through the adapter; measured anyway as a check
            with model.ad.model.disable_adapter():
                res["decision_with_lora"] = decision_probe(model.ad, args.decision_items)
        else:
            res["decision_with_lora"] = decision_probe(model.ad, args.decision_items)
        print("decision", res["decision_with_lora"], flush=True)
    save()


if __name__ == "__main__":
    main()
