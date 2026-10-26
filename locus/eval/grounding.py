"""Grounding metrics on per-row predictions, plus the train-leakage split.

Boxes are xyxy fractions of image width/height. An invalid output is a NaN box:
it scores IoU 0 in the full denominator and is counted in `invalid`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)  # mAcc: Acc at IoU 0.50, 0.55, ..., 0.95
TRAIN_DATASETS = ("refcoco", "refcocoplus", "refcocog")


def iou_xyxy(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Aligned IoU of [N, 4] xyxy boxes; NaN or empty-union rows give 0."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    wh = np.clip(np.minimum(pred[:, 2:], gt[:, 2:]) - np.maximum(pred[:, :2], gt[:, :2]), 0, None)
    inter = wh[:, 0] * wh[:, 1]
    area = lambda b: np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area(pred) + area(gt) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return np.nan_to_num(out, nan=0.0)


def ece(conf, correct, bins: int = 15) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def auroc(score, label) -> float:
    """P(score of a positive > score of a negative), ties counted half (rank-sum)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    order = np.argsort(score, kind="stable")
    sorted_s = score[order]
    ranks = np.empty(len(score))
    i = 0
    while i < len(score):  # average ranks over ties
        j = i
        while j + 1 < len(score) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(iou, conf=None, invalid: int = 0) -> dict:
    """Acc@0.5/0.75/0.9, mAcc, mIoU, invalid count; ECE and AUROC (vs IoU>=0.5) given a confidence."""
    iou = np.asarray(iou, float)
    ok = iou >= 0.5
    r = dict(n=int(len(iou)), acc50=float(ok.mean()), acc75=float((iou >= 0.75).mean()), acc90=float((iou >= 0.9).mean()),
             macc=float(np.mean([(iou >= t - 1e-9).mean() for t in THRESHOLDS])), miou=float(iou.mean()), invalid=int(invalid))
    if conf is not None:
        conf = np.asarray(conf, float)
        r.update(conf_ece=ece(conf, ok), conf_auroc=auroc(conf, ok), mean_conf=float(conf.mean()))
    return r


@lru_cache(maxsize=1)
def train_ann_ids() -> frozenset[int]:
    """COCO ann_ids of every target in the RefCOCO, RefCOCO+ and RefCOCOg train splits (Jag's 321,327-row union)."""
    from locus.data import refcoco

    return frozenset(e.ann_id for ds in TRAIN_DATASETS for e in refcoco.load(ds, "train"))


def leak_split(ann_ids) -> np.ndarray:
    """True where the eval row's target object appears in any RefCOCO-family train split."""
    train = train_ann_ids()
    # Non-numeric ids (Ref-L4's "coco_..."/"o365_...") are not RefCOCO annotations; their image overlap is checked apart.
    return np.array([str(a).isdigit() and int(a) in train for a in ann_ids], bool)


def report(iou, conf=None, ann_id=None, invalid=None) -> dict:
    """metrics() for all rows and, given ann_ids, for the clean and leaked subsets."""
    iou = np.asarray(iou, float)
    inv = np.zeros(len(iou), bool) if invalid is None else np.asarray(invalid, bool)
    sub = lambda m: metrics(iou[m], None if conf is None else np.asarray(conf)[m], int(inv[m].sum()))
    out = dict(all=sub(np.ones(len(iou), bool)))
    if ann_id is not None:
        leaked = leak_split(ann_id)
        out.update(clean=sub(~leaked), leaked=sub(leaked), leaked_frac=float(leaked.mean()))
    return out


def save_raw(path: Path, pred_box, iou, ann_id, image_id, conf=None) -> None:
    """Per-row raw predictions (xyxy fractions, NaN = invalid) for re-scoring and calibration plots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if conf is None else dict(conf=np.asarray(conf, np.float32))
    np.savez_compressed(path, pred_box=np.asarray(pred_box, np.float32), iou=np.asarray(iou, np.float32),
                        ann_id=np.asarray(ann_id), image_id=np.asarray(image_id), **extra)
