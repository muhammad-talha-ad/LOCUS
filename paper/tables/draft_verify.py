"""Draft-and-verify vs Jag, from results/draft_verify/raw_<run>_<split>_p8.npz and results/baselines -> markdown on stdout.

Modes per split: head only (1 branch pass), verified (greedy-exact coordinate tokens), selected: the verified box
when it agrees with the head (IoU >= tau), else the head box, and typical: relaxed verification that accepts a draft
token at >= alpha x the verifier's argmax probability (raw_..._p8a<alpha>.npz, DraftVerify(accept=alpha)). tau and
alpha are chosen on RefCOCO val only (best mean of Acc@0.5/0.75/0.9) and applied unchanged to every other split.

    python paper/tables/draft_verify.py joint_vl2b_mlp_alora_ar1 [--tag _fast_graph_gpupre]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locus.eval.grounding import iou_xyxy, report  # noqa: E402

SPLITS = ["refcoco_val", "refcoco_testA", "refcoco_testB", "refcoco+_val", "refcoco+_testA", "refcoco+_testB",
          "refcocog_val", "refcocog_test"]
TAUS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]  # 0.0 = always verified, 1.01 = always head
K = ["acc50", "acc75", "acc90"]
ALPHAS = [0.7, 0.5, 0.3]
pct = lambda x: f"{100 * x:.2f}"


def load(run, sp, tag="", p="8"):
    f = ROOT / "results/draft_verify" / f"raw_{run}_{sp}{tag}_p{p}.npz"
    return np.load(f) if f.exists() else None


def modes(z, tau):
    agree = iou_xyxy(z["head"], np.nan_to_num(z["box"]))
    return dict(head=z["iou_head"], verified=z["iou"], selected=np.where(agree >= tau, z["iou"], z["iou_head"]))


def choose(run, tag="", ttag=None):
    """tau (agreement rule) and alpha (typical acceptance) chosen on RefCOCO val: best mean of Acc@0.5/0.75/0.9."""
    ttag = tag if ttag is None else ttag
    zv = load(run, "refcoco_val", tag)
    tau = max(TAUS, key=lambda t: np.mean([report(modes(zv, t)["selected"])["all"][k] for k in K]))
    # only alphas run on every split (a partial sweep, e.g. 0.5 on the latency rows only, is not a candidate)
    av = {a: z for a in ALPHAS if (z := load(run, "refcoco_val", ttag, f"8a{a:g}")) is not None
          and all(load(run, sp, ttag, f"8a{a:g}") is not None for sp in SPLITS)}
    alpha = max(av, key=lambda a: np.mean([report(av[a]["iou"])["all"][k] for k in K])) if av else None
    return tau, alpha


def main(run, tag="", ttag=None):
    ttag = tag if ttag is None else ttag
    tau, alpha = choose(run, tag, ttag)
    print(f"### {run}: draft-and-verify vs Jag (τ = {tau}, α = {alpha}, chosen on RefCOCO val)\n")
    print("| split | rows | mode | Acc@0.5 | Acc@0.75 | Acc@0.9 | clean Acc@0.5 | clean Acc@0.9 | Δ vs Jag (0.5 / 0.75 / 0.9) | passes |\n|"
          + "---|" * 10)
    deltas = {m: [] for m in ("head", "verified", "selected", "typical")}
    for sp in SPLITS:
        z = load(run, sp, tag)
        if z is None:
            continue
        jz = np.load(ROOT / "results/baselines" / f"raw_jag_{sp}.npz")
        assert (jz["ann_id"] == z["ann_id"]).all() or sorted(jz["ann_id"]) == sorted(z["ann_id"]), sp
        J = report(jz["iou"].astype(float), None, jz["ann_id"])
        rows = [("jag", None, J)]
        for m, iou in modes(z, tau).items():
            rows.append((m, z["passes"].mean() if m != "head" else 1.0, report(iou, None, z["ann_id"])))
        zt = load(run, sp, ttag, f"8a{alpha:g}") if alpha else None
        if zt is not None:
            rows.append(("typical", zt["passes"].mean(), report(zt["iou"], None, zt["ann_id"])))
        for m, passes, R in rows:
            a, c = R["all"], R["clean"]
            d = "" if m == "jag" else " / ".join(f"{100 * (a[k] - J['all'][k]):+.2f}" for k in K)
            if m != "jag":
                deltas[m].append([a[k] - J["all"][k] for k in K])
            print(f"| {sp} | {a['n']} | {m} | {pct(a['acc50'])} | {pct(a['acc75'])} | {pct(a['acc90'])} | {pct(c['acc50'])} | "
                  f"{pct(c['acc90'])} | {d} | {'' if passes is None else f'{passes:.2f}'} |")
    print("\n**Mean Δ vs Jag over the splits above (Acc@0.5 / 0.75 / 0.9):**\n")
    for m, d in deltas.items():
        if d:
            d = 100 * np.array(d)
            print(f"- {m}: " + " / ".join(f"{x:+.2f}" for x in d.mean(0)) + f"  (wins at 0.5: {(d[:, 0] > 0).sum()}/{len(d)}, "
                  f"at 0.9: {(d[:, 2] > 0).sum()}/{len(d)})")
    print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default="", help="raw-file infix, e.g. _fast_graph_gpupre")
    ap.add_argument("--typical-tag", default=None, help="infix of the typical-acceptance files if it differs from --tag")
    a = ap.parse_args()
    for r in a.runs:
        main(r, a.tag, a.typical_tag)
