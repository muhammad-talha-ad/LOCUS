"""RefCOCO / + / g loader over jxu124 parquet annotations and COCO-2017 images.

RefCOCO images are COCO train2014; the same files live in train2017 under the
same image id, so we read them from data/coco/train2017.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass
class RefExpr:
    image_id: int
    width: int
    height: int
    text: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions
    ann_id: int = -1  # COCO annotation id of the target (leakage checks across RefCOCO/+/g)
    path: str = ""  # image file outside COCO train2017 (Ref-L4)

    @property
    def image_path(self) -> Path:
        return Path(self.path) if self.path else ROOT / "coco" / "train2017" / f"{self.image_id:012d}.jpg"


def load(dataset: str = "refcoco", split: str = "validation") -> list[RefExpr]:
    import json

    if dataset == "refl4":
        return load_refl4(split)
    df = pd.read_parquet(ROOT / "refcoco" / dataset / f"{split}.parquet")
    out = []
    for r in df.itertuples():
        info = json.loads(r.raw_image_info)
        w, h = info["width"], info["height"]
        x0, y0, x1, y1 = (float(v) for v in r.bbox)
        box = (x0 / w, y0 / h, x1 / w, y1 / h)
        for s in r.sentences:
            out.append(RefExpr(int(r.image_id), w, h, s["sent"], box, int(r.ann_id)))
    return out


def load_refl4(split: str) -> list[RefExpr]:
    """Ref-L4 (Chen et al. 2024; CC BY-NC 4.0, evaluation only): COCO val2017 and Objects365 images, one expression per
    row, boxes xywh in pixels. Image and annotation ids are strings ("coco_...", "o365_..."), so they never match a
    RefCOCO id in the leakage checks; the image overlap with our training images is checked in scripts/refl4_overlap.py."""
    df = pd.read_parquet(ROOT / "refl4" / f"ref-l4-{split}.parquet")
    out = []
    for r in df.itertuples():
        x, y, w, h = (float(v) for v in r.bbox)
        out.append(RefExpr(r.image_id, int(r.width), int(r.height), r.caption,
                           (x / r.width, y / r.height, (x + w) / r.width, (y + h) / r.height), r.bbox_id,
                           str(ROOT / "refl4" / "images" / r.file_name)))
    return out


def by_image(exprs: list[RefExpr]) -> dict[int, list[RefExpr]]:
    g = defaultdict(list)
    for e in exprs:
        g[e.image_id].append(e)
    return dict(g)


def sample_images(exprs: list[RefExpr], n_images: int, seed: int = 0) -> dict[int, list[RefExpr]]:
    g = by_image(exprs)
    ids = sorted(g)
    random.Random(seed).shuffle(ids)
    return {i: g[i] for i in ids[:n_images]}


EVAL_SPLITS = {  # the 8 evaluation splits: name → (dataset, parquet split)
    "refcoco_val": ("refcoco", "validation"), "refcoco_testA": ("refcoco", "test"), "refcoco_testB": ("refcoco", "testB"),
    "refcoco+_val": ("refcocoplus", "validation"), "refcoco+_testA": ("refcocoplus", "test"),
    "refcoco+_testB": ("refcocoplus", "testB"), "refcocog_val": ("refcocog", "validation"), "refcocog_test": ("refcocog", "test"),
}
OOD_SPLITS = {"refl4_val": ("refl4", "val"), "refl4_test": ("refl4", "test")}  # outside RefCOCO; not in default loops
ALL_SPLITS = {**EVAL_SPLITS, **OOD_SPLITS}
