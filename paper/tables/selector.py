"""Learned head-vs-verified selector, from results/draft_verify/raw_<run>_<split><tag>_p<p>.npz -> results/selector/ + markdown.

Per row, a small MLP picks the head box or the verified box from features available at inference time (head confidence,
Jacobi passes, head/verified agreement, box geometry). It is trained on RefCOCO val only, to maximise the expected
mean of the Acc@0.5/0.75/0.9 indicators, and applied unchanged to the other 7 splits (held out). Reported as the mean
over 5 seeds, with the seed std. Baseline: the one-parameter agreement rule of paper/tables/draft_verify.py (verified box
if IoU(head, verified) >= τ, else head; τ on RefCOCO val). Δ is vs Jag-0.8B, same as draft_verify.py.

    python paper/tables/selector.py dual_q35_08b_mlp_ar1_fullft_36k
    python paper/tables/selector.py dual_vl2b_mlp_ar1_merger --tag _fast_graph_gpupre --p 8a0.7
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locus.eval.grounding import iou_xyxy, report  # noqa: E402

SPLITS = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcoco+_val", "refcoco+_testA", "refcoco+_testB",
          "refcocog_val", "refcocog_test"]
K = ["acc50", "acc75", "acc90"]
THR = np.array([0.5, 0.75, 0.9])
TAUS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]  # agreement rule, as in draft_verify.py (0.0 = always verified, 1.01 = always head)
SEEDS = range(5)


def features(z):
    h, v = z["head"].astype(np.float64), z["box"].astype(np.float64)
    bad = ~np.isfinite(v).all(1)
    v = np.where(bad[:, None], h, v)
    geo = lambda b: np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    gh, gv = geo(h), geo(v)
    area = lambda g: np.sqrt(np.clip(g[:, 0] * g[:, 1], 1e-6, None))
    return np.column_stack([z["conf"], np.log1p(z["passes"]), iou_xyxy(h, v), bad, gh, gv, area(gh), area(gv),
                            np.abs(h - v).sum(1), np.log(area(gv) / area(gh))]).astype(np.float32)


def utility(iou):
    return (np.nan_to_num(iou)[:, None] >= THR).mean(1)


def fit(x, uh, uv, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(),
                              torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-2)
    x, uh, uv = map(torch.tensor, (x, uh.astype(np.float32), uv.astype(np.float32)))
    g = torch.Generator().manual_seed(seed)
    for _ in range(30):
        for idx in torch.randperm(len(x), generator=g).split(512):
            p = torch.sigmoid(net(x[idx])[:, 0])  # P(choose head)
            loss = -(p * uh[idx] + (1 - p) * uv[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def main(run, tag, p):
    zs = {sp: np.load(ROOT / "results/draft_verify" / f"raw_{run}_{sp}{tag}_p{p}.npz") for sp in SPLITS}
    xs = {sp: features(z) for sp, z in zs.items()}
    mu, sd = xs["refcoco_val"].mean(0), xs["refcoco_val"].std(0) + 1e-6
    norm = lambda x: (x - mu) / sd
    zv = zs["refcoco_val"]
    agree = {sp: iou_xyxy(z["head"], np.nan_to_num(z["box"])) for sp, z in zs.items()}
    rule = lambda sp, t: np.where(agree[sp] >= t, zs[sp]["iou"], zs[sp]["iou_head"])
    tau = max(TAUS, key=lambda t: np.mean([report(rule("refcoco_val", t))["all"][k] for k in K]))
    nets = [fit(norm(xs["refcoco_val"]), utility(zv["iou_head"]), utility(zv["iou"]), s) for s in SEEDS]
    out = dict(run=run, tag=tag, p=p, train_split="refcoco_val", seeds=list(SEEDS), splits={},
               date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               git_commit=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True).stdout.strip() or None)
    print(f"### {run}{tag} p{p}: learned selector (trained on RefCOCO val, 5 seeds)\n")
    print("| split | head | verified | selector (mean ± std over seeds) | Δ selector vs Jag | frac. head |\n|---|---|---|---|---|---|")
    dl = {m: [] for m in ["head", "verified", f"agreement rule (τ = {tau})", "selector"]}
    for sp, z in zs.items():
        with torch.no_grad():
            pick = [(net(torch.tensor(norm(xs[sp])))[:, 0] > 0).numpy() for net in nets]
        sel = np.array([[report(np.where(pk, z["iou_head"], z["iou"]))["all"][k] for k in K] for pk in pick])
        J = report(np.load(ROOT / "results/baselines" / f"raw_jag_{sp}.npz")["iou"])["all"]
        jv = np.array([J[k] for k in K])
        rh, rv = [np.array([report(z[f])["all"][k] for k in K]) for f in ("iou_head", "iou")]
        rr = np.array([report(rule(sp, tau))["all"][k] for k in K])
        out["splits"][sp] = dict(head=rh.tolist(), verified=rv.tolist(), selector_mean=sel.mean(0).tolist(),
                                 selector_std=sel.std(0).tolist(), frac_head=float(np.mean(pick)), jag=jv.tolist(),
                                 rule=rr.tolist(), tau=tau)
        f = lambda a: " / ".join(f"{100 * x:.2f}" for x in a)
        print(f"| {sp}{' (train)' if sp == 'refcoco_val' else ''} | {f(rh)} | {f(rv)} | {f(sel.mean(0))} ± "
              f"{100 * sel.std(0).max():.2f} | {' / '.join(f'{100 * x:+.2f}' for x in sel.mean(0) - jv)} | {np.mean(pick):.2f} |")
        if sp != "refcoco_val":
            for m, a in zip(dl, (rh, rv, rr, sel.mean(0))):
                dl[m].append(a - jv)
    print("\n**Mean Δ vs Jag over the 7 held-out splits (Acc@0.5 / 0.75 / 0.9):**\n")
    for m, d in dl.items():
        d = 100 * np.array(d)
        out[f"mean_delta_{m.split(' (')[0].replace(' ', '_')}"] = d.mean(0).tolist()
        print(f"- {m}: " + " / ".join(f"{x:+.2f}" for x in d.mean(0)) + f"  (wins at 0.5: {(d[:, 0] > 0).sum()}/7)")
    print()
    dst = ROOT / "results/selector"
    dst.mkdir(exist_ok=True)
    (dst / f"{run}{tag}_p{p}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default="")
    ap.add_argument("--p", default="8", help="raw-file pass suffix: 8 (exact) or 8a<alpha> (typical)")
    a = ap.parse_args()
    for r in a.runs:
        main(r, a.tag, a.p)
