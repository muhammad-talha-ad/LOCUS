"""Paired significance vs Jag-0.8B, from results/draft_verify/raw_<run>_<split><tag>_p<p>.npz -> results/mcnemar/ + markdown.

Per split and pooled over the 8 splits, at each Acc threshold: exact two-sided McNemar test on the per-row correct/incorrect
pairs (b = we are right and Jag wrong, c = the reverse), plus a 95% CI for the pooled Δ from a paired bootstrap over images.
Several expressions share an image (and RefCOCO/RefCOCO+ share images), so per-row McNemar is optimistic; the image
bootstrap is the clustered check. Jag's raw rows are in loader order, ours are grouped by image (eval_draft_verify.py
accuracy()); they are aligned by the same stable grouping and checked on (ann_id, image_id).

    python paper/tables/mcnemar.py dual_q35_08b_mlp_ar1_fullft_54k --typical 8a0.3 [--clean]

--clean keeps only rows whose target object is in no RefCOCO-family train split (grounding.leak_split): Jag trains on
that union, we train on none of the evaluation images. Output: results/mcnemar/<run><tag>_clean.json.
"""
import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locus.eval.grounding import leak_split  # noqa: E402

SPLITS = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcoco+_val", "refcoco+_testA", "refcoco+_testB",
          "refcocog_val", "refcocog_test"]
THR = [0.5, 0.75, 0.9]
N_BOOT = 2000


def mcnemar_p(b, c):
    """Exact two-sided McNemar: binomial(b + c, 1/2) tail, doubled."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    logpmf = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logpmf)
    return min(1.0, 2 * math.exp(m) * sum(math.exp(x - m) for x in logpmf))


def jag_rows(sp, ann, img):
    z = np.load(ROOT / "results/baselines" / f"raw_jag_{sp}.npz")
    first = {}
    for i, g in enumerate(z["image_id"]):
        first.setdefault(int(g), i)
    perm = np.array(sorted(range(len(z["image_id"])), key=lambda i: first[int(z["image_id"][i])]))  # stable by image
    assert (z["ann_id"][perm] == ann).all() and (z["image_id"][perm] == img).all(), sp
    return np.nan_to_num(z["iou"][perm])


def main(run, tag, modes, clean=False):
    out = dict(run=run, tag=tag, modes=modes, clean=clean, thresholds=THR, n_boot=N_BOOT, splits={}, pooled={},
               date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               git_commit=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True).stdout.strip() or None)
    pool = {m: [] for m in modes}
    jag_all, img_all = [], []
    for sp in SPLITS:
        out["splits"][sp] = {}
        for m, (p, field) in modes.items():
            z = np.load(ROOT / "results/draft_verify" / f"raw_{run}_{sp}{tag}_p{p}.npz")
            keep = ~leak_split(z["ann_id"]) if clean else np.ones(len(z["ann_id"]), bool)
            j = jag_rows(sp, z["ann_id"], z["image_id"])[keep]
            ours = np.nan_to_num(z[field])[keep]
            pool[m].append(ours)
            if m == next(iter(modes)):
                jag_all.append(j)
                img_all.append(z["image_id"][keep])
            res = []
            for t in THR:
                a, q = ours >= t, j >= t
                b, c = int((a & ~q).sum()), int((~a & q).sum())
                res.append(dict(thr=t, delta=float(100 * (a.mean() - q.mean())), b=b, c=c, p=mcnemar_p(b, c)))
            out["splits"][sp][m] = res
    j = np.concatenate(jag_all)
    img = np.concatenate(img_all)
    _, cl = np.unique(img, return_inverse=True)
    rng = np.random.default_rng(0)
    n_img = cl.max() + 1
    W = np.stack([np.bincount(rng.integers(0, n_img, n_img), minlength=n_img) for _ in range(N_BOOT)]).astype(float)
    rows_per_img = W @ np.bincount(cl, minlength=n_img)
    print(f"### {run}{tag}{' (clean rows)' if clean else ''}: paired tests vs Jag-0.8B ({len(j)} rows, {n_img} images, 8 splits)\n")
    print("| mode | thr | Δ pooled | 95% CI (image bootstrap) | b / c | McNemar p | splits p<0.05 (+/−) |\n|---|---|---|---|---|---|---|")
    for m in modes:
        o = np.concatenate(pool[m])
        out["pooled"][m] = []
        for ti, t in enumerate(THR):
            d = (o >= t).astype(float) - (j >= t)
            boot = 100 * (W @ np.bincount(cl, d, minlength=n_img)) / rows_per_img
            b, c = int((d > 0).sum()), int((d < 0).sum())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            sig = [s[m][ti] for s in out["splits"].values()]
            npos = sum(s["p"] < 0.05 and s["delta"] > 0 for s in sig)
            nneg = sum(s["p"] < 0.05 and s["delta"] < 0 for s in sig)
            out["pooled"][m].append(dict(thr=t, delta=float(100 * d.mean()), ci=[float(lo), float(hi)], b=b, c=c, p=mcnemar_p(b, c),
                                         splits_sig_pos=int(npos), splits_sig_neg=int(nneg)))
            print(f"| {m} | {t} | {100 * d.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {b} / {c} | {mcnemar_p(b, c):.2g} | {npos} / {nneg} |")
    print()
    dst = ROOT / "results/mcnemar"
    dst.mkdir(exist_ok=True)
    (dst / f"{run}{tag}{'_clean' if clean else ''}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default="")
    ap.add_argument("--typical", default=None, help="raw-file pass suffix of the typical arm, e.g. 8a0.3")
    ap.add_argument("--clean", action="store_true", help="only rows whose target object is in no train split")
    a = ap.parse_args()
    modes = {"head": ("8", "iou_head"), "verified": ("8", "iou")}
    if a.typical:
        modes["typical"] = (a.typical, "iou")
    for r in a.runs:
        main(r, a.tag, modes, a.clean)
