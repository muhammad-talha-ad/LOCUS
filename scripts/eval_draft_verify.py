"""Evaluate a trained box head alone and as a drafter for coordinate tokens (locus/models/draft_verify.py).

Accuracy: every expression of an image is a branch of one prefill. For each --max-passes value: head boxes, final
(greedy-exact) boxes, sequential verifier passes. max_passes=0 is plain greedy decoding = the coordinate-token
control on the same prompt, cache and kernels.
Latency (--latency): batch 1, one expression per image, end-to-end incl. image open + preprocessing.

    python scripts/eval_draft_verify.py --run cmp_qwen3-vl-2b_lora64 --split refcoco_val --n 2000 --max-passes 0 4
    python scripts/eval_draft_verify.py --run cmp_qwen3-vl-2b_lora64 --latency 256 --max-passes 0 4
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from train_box import EVAL_SPLITS  # noqa: E402

from locus.data import refcoco  # noqa: E402
from locus.eval.grounding import iou_xyxy, report  # noqa: E402
from locus.models.boxvlm import BoxVLM  # noqa: E402
from locus.models.draft_verify import DraftVerify  # noqa: E402
from locus.models.heads.box_decoder import upgrade_state_dict  # noqa: E402


def load_trained(run: str) -> BoxVLM:
    cfg = json.loads((REPO / "results/box_train" / f"{run}.json").read_text())["config"]
    m = BoxVLM(cfg["backbone"], max_pixels=cfg["px"] ** 2, lora_r=cfg["lora_r"], mid=cfg["mid"],
               decoder_layers=cfg["decoder_layers"], exit_layer=cfg.get("exit_layer", 1.0),
               n_query=cfg.get("n_query", 1), refine=cfg.get("refine", False), head=cfg.get("head", "xattn"),
               dfl=cfg.get("dfl", 0), alora=cfg.get("alora", False),
               merger_lora=cfg.get("merger_lora", False))
    sd = torch.load(REPO / "data/ckpt" / f"{run}.pt")
    with torch.no_grad():
        m.box_embed.copy_(sd["box_embed"].reshape(m.box_embed.shape))
        m.decoder.load_state_dict(upgrade_state_dict(sd["decoder"]))
        named = dict(m.ad.model.named_parameters())
        for k, v in sd.get("lora", {}).items():
            named[k].copy_(v)
    return m.eval()


FP32 = False  # --fp32: float32 weights, no autocast (exactness control; the paper numbers are bf16)
GPU_PREPROCESS = False  # resize/normalise/patchify on the GPU (torchvision backend): ~3 ms instead of ~36 ms per image


def prefix_inputs(m: BoxVLM, imgs):
    m.ad.tok.padding_side = "left"
    kw = dict(m.ad.proc_kwargs, **({"device": "cuda"} if GPU_PREPROCESS else {}))
    return m.ad.processor(images=imgs, text=[m.prefix] * len(imgs), return_tensors="pt", padding=True, **kw).to("cuda")


def accuracy(m, dv_by_k, exprs, images_per_batch, verify=True, drafter=None):
    """`drafter`: (model, DraftVerify) whose head drafts for m's verifier (separately trained models; two prefills)."""
    g = refcoco.by_image(exprs)
    groups = list(g.values())
    out = {k: dict(head=[], box=[], passes=[], conf=[]) for k in dv_by_k}
    t = {k: 0.0 for k in dv_by_k}
    for s in range(0, len(groups), images_per_batch):
        chunk = groups[s : s + images_per_batch]
        imgs = [Image.open(c[0].image_path).convert("RGB") for c in chunk]
        texts = [e.text for c in chunk for e in c]
        ri = torch.cat([torch.full((len(c),), i) for i, c in enumerate(chunk)]).cuda()
        pin = prefix_inputs(m, imgs)
        for k, dv in dv_by_k.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with contextlib.nullcontext() if FP32 else torch.autocast("cuda", torch.bfloat16):
                d = None if drafter is None else drafter[1](prefix_inputs(drafter[0], imgs), texts, ri, verify=False)
                r = dv(pin, texts, ri, verify=verify, draft=None if d is None else d["head"])
                if d is not None:  # report the drafter's head and confidence
                    r.update(head=d["head"], conf=d["conf"])
            if not verify:
                r.update(box=r["head"], passes=np.zeros(len(texts), int))
            torch.cuda.synchronize()
            t[k] += time.perf_counter() - t0
            for key in out[k]:
                out[k][key].append(r[key])
    rows = [e for c in groups for e in c]
    return rows, {k: {key: np.concatenate(v) for key, v in o.items()} for k, o in out.items()}, t


def latency(m, dv_by_k, exprs, n, warmup=10):
    """Batch 1, one expression per image; e2e = image open + preprocessing + model; head-only and each max_passes."""
    picks = list({e.image_id: e for e in exprs}.values())[: n + warmup]
    res = {}
    if m.pgraph is not None:  # capture every graph the timed rows need up front (a deployment pre-captures its shapes)
        t0 = time.perf_counter()
        for e in picks:
            pin = prefix_inputs(m, [Image.open(e.image_path).convert("RGB")])
            for dv in dv_by_k.values():
                with torch.autocast("cuda", torch.bfloat16):
                    dv(pin, [e.text], torch.zeros(1, dtype=torch.long, device="cuda"))
        torch.cuda.synchronize()
        g = m.pgraph
        hybrid = hasattr(g, "branches")  # GraphedHybrid holds both kinds
        res["prefill_graphs"] = dict(n=len(g.prefills if hybrid else g.graphs),
                                     n_branch=len(g.branches if hybrid else m.graph.graphs), capture_s=time.perf_counter() - t0,
                                     mem_gib=torch.cuda.memory_reserved() / 2**30)
        print("prefill graphs", res["prefill_graphs"], flush=True)
    for k, dv in [("head", next(iter(dv_by_k.values())))] + list(dv_by_k.items()):
        ts, passes = [], []
        for j, e in enumerate(picks):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = Image.open(e.image_path).convert("RGB")
            pin = prefix_inputs(m, [img])
            with torch.autocast("cuda", torch.bfloat16):
                r = dv(pin, [e.text], torch.zeros(1, dtype=torch.long, device="cuda"), verify=k != "head")
            torch.cuda.synchronize()
            if j >= warmup:
                ts.append((time.perf_counter() - t0) * 1000)
                if k != "head":
                    passes.append(int(r["passes"][0]))
        ts = np.array(ts)
        res[str(k)] = dict(mean_ms=float(ts.mean()), median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)),
                           mean_passes=float(np.mean(passes)) if passes else 0.0, n=len(ts))
        print("latency", k, res[str(k)], flush=True)
    return res


def q_images(exprs, qmax, n, warmup):
    """The first n + warmup images with at least qmax expressions, each with its first qmax expressions."""
    g = [c[:qmax] for c in refcoco.by_image(exprs).values() if len(c) >= qmax]
    return g[: n + warmup]


def latency_q(m, dv_by_k, exprs, n, qs, warmup=10):
    """Batch 1 image with Q questions on one shared prefill; e2e as latency() (image open + preprocessing + model).
    The same images at every Q (those with >= max(qs) expressions)."""
    groups = q_images(exprs, max(qs), n, warmup)
    res = dict(images=len(groups) - warmup, qs=list(qs))
    for q in qs:
        if m.pgraph is not None:  # pre-capture this Q's shapes, as latency() does
            for c in groups:
                pin = prefix_inputs(m, [Image.open(c[0].image_path).convert("RGB")])
                for dv in dv_by_k.values():
                    with torch.autocast("cuda", torch.bfloat16):
                        dv(pin, [e.text for e in c[:q]], torch.zeros(q, dtype=torch.long, device="cuda"))
            torch.cuda.synchronize()
        for k, dv in [("head", next(iter(dv_by_k.values())))] + list(dv_by_k.items()):
            ts, passes = [], []
            for j, c in enumerate(groups):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pin = prefix_inputs(m, [Image.open(c[0].image_path).convert("RGB")])
                with torch.autocast("cuda", torch.bfloat16):
                    r = dv(pin, [e.text for e in c[:q]], torch.zeros(q, dtype=torch.long, device="cuda"), verify=k != "head")
                torch.cuda.synchronize()
                if j >= warmup:
                    ts.append((time.perf_counter() - t0) * 1000)
                    if k != "head":
                        passes.append(float(np.max(r["passes"])))
            ts = np.array(ts)
            res[f"q{q}_{k}"] = dict(mean_ms=float(ts.mean()), median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)),
                                   per_question_ms=float(ts.mean() / q), mean_max_passes=float(np.mean(passes)) if passes else 0.0)
            print("latency", q, k, res[f"q{q}_{k}"], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="refcoco_val", choices=sorted(refcoco.ALL_SPLITS))
    ap.add_argument("--n", type=int, default=0, help="first n rows (whole images; 0 = all)")
    ap.add_argument("--max-passes", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--accept", type=float, nargs="+", default=[1.0], help="typical-acceptance ratios (1 = exact "
                    "greedy); each < 1 is saved as _p<k>a<ratio>")
    ap.add_argument("--images-per-batch", type=int, default=4)
    ap.add_argument("--base-verifier", action="store_true", help="verify with the base model even if the adapter "
                    "was trained on verifier branches")
    ap.add_argument("--fast", action="store_true", help="plain base + merged-LoRA models (BoxVLM.enable_fast)")
    ap.add_argument("--graphs", action="store_true", help="with --fast: CUDA-graphed branch passes (graphed.py)")
    ap.add_argument("--graph-prefill", action="store_true", help="with --graphs: CUDA-graphed batch-1 prefill (latency)")
    ap.add_argument("--gpu-preprocess", action="store_true", help="image preprocessing on the GPU")
    ap.add_argument("--draft-run", default="", help="draft with this run's head instead of --run's own (separately "
                    "trained head and verifier)")
    ap.add_argument("--no-verify", action="store_true", help="accuracy of the head only (no verifier passes)")
    ap.add_argument("--fp32", action="store_true", help="float32 weights, no autocast (eager path only)")
    ap.add_argument("--px", type=int, default=0, help="pixel budget override (default: the run's training budget)")
    ap.add_argument("--latency", type=int, default=0, help="batch-1 latency over this many images instead of accuracy")
    ap.add_argument("--latency-q", type=int, nargs="+", default=None, help="with --latency: Q questions per image on one "
                    "prefill, for each Q given (images with >= max Q expressions)")
    args = ap.parse_args()

    global GPU_PREPROCESS, FP32
    GPU_PREPROCESS, FP32 = args.gpu_preprocess, args.fp32
    assert not (args.fp32 and args.fast), "--fp32 runs the eager path"
    m = load_trained(args.run)
    if args.fp32:
        m = m.float()
    if args.px:
        m.ad.set_resolution(args.px)
    if args.fast:
        m.enable_fast(graphs=args.graphs, graph_prefill=args.graph_prefill)
    cfg = json.loads((REPO / "results/box_train" / f"{args.run}.json").read_text())["config"]
    joint = cfg.get("ar_weight", 0) > 0
    if not joint and cfg.get("init") and cfg.get("lr_lora", 1) == 0:  # frozen adapter inherited from --init
        joint = json.loads((REPO / "results/box_train" / f"{cfg['init']}.json").read_text())["config"].get("ar_weight", 0) > 0
    va = joint and not args.base_verifier  # adapter was trained on verifier branches too
    dv_by_k = {(f"{k}" if a >= 1 else f"{k}a{a:g}"): DraftVerify(m, max_passes=k, verifier_adapter=va, accept=a)
               for k in args.max_passes for a in args.accept}
    exprs = refcoco.load(*refcoco.ALL_SPLITS[args.split])
    if args.n:  # first n rows, completed to whole images
        last = exprs[min(args.n, len(exprs)) - 1].image_id
        exprs = [e for i, e in enumerate(exprs) if i < args.n or e.image_id == last]
    out = REPO / "results/draft_verify"
    out.mkdir(parents=True, exist_ok=True)
    res = dict(run=args.run, split=args.split, config=vars(args), gpu=torch.cuda.get_device_name(),
               date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               git_commit=subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip() or None)
    tag_a = "" if args.accept == [1.0] else "_acc" + "-".join(f"{a:g}" for a in args.accept)
    if args.latency:
        if args.latency_q:
            res["latency_q"] = latency_q(m, dv_by_k, exprs, args.latency, args.latency_q)
            tag_a += "_q" + "-".join(map(str, args.latency_q))
        else:
            res["latency"] = latency(m, dv_by_k, exprs, args.latency)
        (out / f"latency_{args.run}{'_px%d' % args.px if args.px else ''}{'_gpupre' if args.gpu_preprocess else ''}{'_fast' if args.fast else ''}{'_graph' if args.graphs else ''}{'_gprefill' if args.graph_prefill else ''}{'_' + args.split if args.split != 'refcoco_val' else ''}{tag_a}.json").write_text(json.dumps(res, indent=1))
        return
    drafter = None
    if args.draft_run:
        dm = load_trained(args.draft_run)
        if args.px:
            dm.ad.set_resolution(args.px)
        drafter = (dm, DraftVerify(dm))
    rows, preds, t = accuracy(m, dv_by_k, exprs, args.images_per_batch, verify=not args.no_verify, drafter=drafter)
    tag = ("_draft-" + args.draft_run if args.draft_run else "") + ("_n%d" % args.n if args.n else "") + ("_fp32" if args.fp32 else "") + ("_px%d" % args.px if args.px else "") + ("_fast" if args.fast else "") + ("_graph" if args.graphs else "") + ("_gpupre" if args.gpu_preprocess else "") + ("_head" if args.no_verify else "")
    gt = np.array([e.box for e in rows])
    ann = np.array([e.ann_id for e in rows])
    res["n"], res["results"] = len(rows), {}
    for k, p in preds.items():
        ih, iv = iou_xyxy(p["head"], gt), iou_xyxy(np.nan_to_num(p["box"]), gt)
        res["results"][str(k)] = dict(head=report(ih, p["conf"], ann), verified=report(iv, p["conf"], ann),
                                      mean_passes=float(p["passes"].mean()), pass_hist=np.bincount(p["passes"]).tolist(),
                                      invalid=int(np.isnan(p["box"]).any(1).sum()), seconds=t[k])
        np.savez_compressed(out / f"raw_{args.run}_{args.split}{tag}_p{k}.npz", head=p["head"], box=p["box"], conf=p["conf"],
                            passes=p["passes"], iou_head=ih, iou=iv, ann_id=ann, image_id=np.array([e.image_id for e in rows]))
        a, h = res["results"][str(k)]["verified"]["all"], res["results"][str(k)]["head"]["all"]
        print(f"max_passes={k}: head acc50 {h['acc50']:.4f} acc90 {h['acc90']:.4f} | verified acc50 {a['acc50']:.4f} "
              f"acc75 {a['acc75']:.4f} acc90 {a['acc90']:.4f} | passes {p['passes'].mean():.2f} | {t[k]:.0f}s", flush=True)
    (out / f"{args.run}_{args.split}{'_draft-' + args.draft_run if args.draft_run else ''}{'_n%d' % args.n if args.n else ''}{'_fp32' if args.fp32 else ''}{'_px%d' % args.px if args.px else ''}{'_gpupre' if args.gpu_preprocess else ''}{'_fast' if args.fast else ''}{'_graph' if args.graphs else ''}"
     f"{'_head' if args.no_verify else ''}{tag_a}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
