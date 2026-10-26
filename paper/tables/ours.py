"""Our trained box heads vs Jag and the zero-shot VLMs, from results/box_train and results/baselines -> markdown on stdout.

Every number is recomputed from per-row raw outputs (raw_*.npz), so all / clean / leaked subsets and matched-row
comparisons against the zero-shot subsets use one code path (locus.eval.grounding.report).

    python paper/tables/ours.py [run names...]   # default: every results/box_train/*.json with final evals
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locus.eval.grounding import report  # noqa: E402

TRAIN, BASE = ROOT / "results/box_train", ROOT / "results/baselines"
SPLITS = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcoco+_val", "refcoco+_testA", "refcoco+_testB",
          "refcocog_val", "refcocog_test"]
ZS = ["qwen3-vl-2b", "qwen3-vl-4b", "qwen3.5-2b", "qwen3.5-4b"]
pct = lambda x: f"{100 * x:.2f}"


def raw(path: Path):
    if not path.exists():
        return None
    z = np.load(path)
    return dict(iou=z["iou"].astype(float), conf=z["conf"] if "conf" in z else None, ann_id=z["ann_id"])


def runs(names):
    if not names:
        names = sorted(f.stem for f in TRAIN.glob("*.json") if json.loads(f.read_text()).get("final"))
    return {n: json.loads((TRAIN / f"{n}.json").read_text()) for n in names}


def per_split(names):
    print("### Final evals, all rows / clean rows (target object absent from every RefCOCO-family train split)\n")
    print("| run | split | rows | Acc@0.5 | Acc@0.75 | Acc@0.9 | mIoU | ECE | AUROC | clean Acc@0.5 | clean Acc@0.9 | leaked − clean (Acc@0.5) |\n|"
          + "---|" * 12)
    for name in names:
        for sp in SPLITS:
            r = raw(TRAIN / f"raw_{name}_{sp}.npz")
            if r is None:
                continue
            rep = report(r["iou"], r["conf"], r["ann_id"])
            a, c, l = rep["all"], rep["clean"], rep["leaked"]
            print(f"| {name} | {sp} | {a['n']} | {pct(a['acc50'])} | {pct(a['acc75'])} | {pct(a['acc90'])} | {pct(a['miou'])} | "
                  f"{a.get('conf_ece', float('nan')):.3f} | {a.get('conf_auroc', float('nan')):.3f} | {pct(c['acc50'])} | {pct(c['acc90'])} | "
                  f"{100 * (l['acc50'] - c['acc50']):+.1f} |")
    print()


def vs(names, other: str, other_dir: Path, title: str):
    """Head-to-head on identical rows. Zero-shot runs cover a prefix of each split; Jag covers every row."""
    print(f"### {title} (identical rows; Δ = ours − {other}, points)\n")
    print("| run | split | rows | subset | Acc@0.5 Δ | Acc@0.75 Δ | Acc@0.9 Δ | mIoU Δ | clean Acc@0.5 Δ | clean Acc@0.9 Δ |\n|" + "---|" * 10)
    for name in names:
        for sp in SPLITS:
            a, b = raw(TRAIN / f"raw_{name}_{sp}.npz"), raw(other_dir / f"raw_{other}_{sp}.npz")
            if a is None or b is None:
                continue
            n = len(b["iou"])
            assert n <= len(a["iou"]) and (a["ann_id"][:n] == b["ann_id"]).all(), (name, other, sp)
            ra, rb = report(a["iou"][:n], None, a["ann_id"][:n]), report(b["iou"], None, b["ann_id"])
            d = lambda part, k: f"{100 * (ra[part][k] - rb[part][k]):+.2f}"
            sub = "full" if n == len(a["iou"]) else f"first {n}"
            print(f"| {name} | {sp} | {n} | {sub} | {d('all', 'acc50')} | {d('all', 'acc75')} | {d('all', 'acc90')} | {d('all', 'miou')} | "
                  f"{d('clean', 'acc50')} | {d('clean', 'acc90')} |")
    print()


def curves(names):
    print("### Training curves (2,000 RefCOCO val rows)\n")
    print("| run | step | Acc@0.5 | Acc@0.75 | Acc@0.9 | mIoU | ECE | rows/s (eval) |\n|---|---|---|---|---|---|---|---|")
    for name, res in runs(names).items():
        for c in res["curve"]:
            print(f"| {name} | {c['step']} | {pct(c['acc50'])} | {pct(c['acc75'])} | {pct(c['acc90'])} | {pct(c['miou'])} | "
                  f"{c['conf_ece']:.3f} | {c['rows_per_s']:.1f} |")
    print()


def decision(names):
    print("### Decision retention (A-OKVQA val MC, same items; base = frozen backbone from the bake-off)\n")
    print("| run | items | base acc | with LoRA | Δ | base ECE | ECE with LoRA |\n|---|---|---|---|---|---|---|")
    for name, res in runs(names).items():
        d = res.get("decision_with_lora")
        bf = ROOT / "results/bakeoff" / f"{res['config']['backbone']}.json"
        if d and bf.exists():
            b = json.loads(bf.read_text())["decision"]
            assert b["n"] == d["n"], name
            print(f"| {name} | {d['n']} | {b['acc']:.3f} | {d['acc']:.3f} | {d['acc'] - b['acc']:+.3f} | {b['ece']:.3f} | {d['ece']:.3f} |")
    print()


if __name__ == "__main__":
    names = list(runs(sys.argv[1:]))
    curves(names)
    per_split(names)
    vs(names, "jag", BASE, "vs Jag (published weights, our harness)")
    for z in ZS:
        vs(names, z, BASE, f"vs {z} zero-shot")
    decision(names)
