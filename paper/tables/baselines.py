"""Baseline tables (week 2) from results/baselines/*.json and Jag's published results.json -> markdown on stdout."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/baselines"
PUB = json.loads((ROOT / "data/external/jag_repo/evaluation/results.json").read_text())
PUB_KEY = {"refcoco_testA": ("refcoco", "testA"), "refcoco_testB": ("refcoco", "testB"), "refcoco+_testA": ("refcocop", "testA"),
           "refcoco+_testB": ("refcocop", "testB"), "refcocog_test": ("refcocog", "test")}
ZS = ["qwen3-vl-2b", "qwen3-vl-4b", "qwen3.5-2b", "qwen3.5-4b"]
pct = lambda x: f"{100 * x:.2f}"


def load(name):
    f = R / f"{name}.json"
    return json.loads(f.read_text())["splits"] if f.exists() else {}


def published(model, sp):
    ds, s = PUB_KEY[sp]
    row = next(r for r in PUB["quality"]["rows"] if r["dataset"] == ds and r["split"] == s)
    return row["models"][model]


def repro():
    print("### Reproduction of Jag's published Acc@0.5 (%)\n")
    print("| model | split | published | ours | delta | published mIoU | our mIoU | published invalid | our invalid | rows |\n|---|---|---|---|---|---|---|---|---|---|")
    for name, pub_name in (("jag", "Jag"), ("qwen3.5-0.8b-base", "Qwen3.5-0.8B")):
        res = load(name)
        for sp in PUB_KEY:
            if sp not in res:
                continue
            p, o = published(pub_name, sp), res[sp]["all"]
            n = o["n"]
            print(f"| {name} | {sp} | {pct(p['iou_at_05'])} | {pct(o['acc50'])} | {100 * (o['acc50'] - p['iou_at_05']):+.2f} | "
                  f"{pct(p['miou'])} | {pct(o['miou'])} | {n - p['valid_count']} | {o['invalid']} | {n} |")
    print()


def full(name, title, splits=None):
    res = load(name)
    if not res:
        return
    print(f"### {title}\n")
    print("| split | rows | Acc@0.5 | Acc@0.75 | Acc@0.9 | mAcc | mIoU | invalid | leaked % | clean Acc@0.5 | leaked Acc@0.5 | gap (leaked-clean) |\n|" + "---|" * 12)
    for sp, r in res.items():
        a, c, l = r["all"], r["clean"], r["leaked"]
        sub = "" if r["full_split"] else " (subset)"
        print(f"| {sp}{sub} | {a['n']} | {pct(a['acc50'])} | {pct(a['acc75'])} | {pct(a['acc90'])} | {pct(a['macc'])} | {pct(a['miou'])} | "
              f"{a['invalid']} | {100 * r['leaked_frac']:.1f} | {pct(c['acc50'])} ({c['n']}) | {pct(l['acc50'])} ({l['n']}) | {100 * (l['acc50'] - c['acc50']):+.1f} |")
    print()


def calibration():
    print("### Confidence of the autoregressive baselines (mean probability of the coordinate digits), Acc@0.5 as target\n")
    print("| model | split | rows | ECE (15 bins) | AUROC | mean conf | Acc@0.5 |\n|---|---|---|---|---|---|---|")
    for name in ZS:
        for sp, r in load(name).items():
            a = r["all"]
            if "conf_ece" in a:
                print(f"| {name} | {sp} | {a['n']} | {a['conf_ece']:.3f} | {a['conf_auroc']:.3f} | {a['mean_conf']:.3f} | {pct(a['acc50'])} |")
    print()


def latency():
    f = R / "latency.json"
    if not f.exists():
        return
    L = json.loads(f.read_text())
    print("### Latency, batch 1, 256 RefCOCO testA rows after 10 warmup (ms)\n")
    print("| model | px budget | resize | prompt tok | new tok | e2e mean | e2e median | e2e p95 | model-only mean | model-only median | peak alloc GiB |\n|" + "---|" * 11)
    for k, r in sorted(L.items(), key=lambda kv: (kv[1]["px"], kv[0])):
        e, m = r["end_to_end"], r["model_only"] or dict(mean_ms=float('nan'), median_ms=float('nan'))
        print(f"| {r['model']} | {r['px']} | {'fit to area' if r.get('fit_px') else 'native (cap only)'} | {r['mean_prompt_tokens']:.0f} | "
              f"{r.get('mean_new_tokens', 0):.1f} | {e['mean_ms']:.1f} | {e['median_ms']:.1f} | {e['p95_ms']:.1f} | {m['mean_ms']:.1f} | {m['median_ms']:.1f} | {r['peak_alloc_gib']:.2f} |".replace('nan', '—').replace('| 0.00 |', '| — |'))
    print()


if __name__ == "__main__":
    repro()
    full("jag", "Jag (our re-implementation, fp32 weights + bf16 autocast), all RefCOCO/+/g eval splits")
    full("jag_bf16", "Jag with bf16 backbone weights (latency config), two splits")
    full("qwen3.5-0.8b-base", "Qwen3.5-0.8B base (Jag's JSON prompt), Jag's five test splits")
    for z in ZS:
        full(z, f"{z} zero-shot coordinate tokens")
    calibration()
    latency()
