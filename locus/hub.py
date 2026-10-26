"""Release format for trained runs: a directory (or Hugging Face repo) with model.safetensors + locus_config.json.

model.safetensors holds only what training changed: the box query embedding (box_embed), the head (decoder.*) and
the LoRA or fully fine-tuned backbone weights (weights.<parameter name>). The base VLM comes from its own Hub repo
(locus_config.json: base_model), so a LoRA release stays small.

    python -m locus.hub export <run> <out_dir>      # from results/box_train/<run>.json + data/ckpt/<run>.pt
    m = from_pretrained("MTalha2001/LOCUS-2B")       # or a local directory
    ground(m, Image.open("cat.jpg"), ["the cat on the left"])  # -> one dict per expression (box in pixels)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from locus.models.backbone.adapters import BACKBONES
from locus.models.boxvlm import BoxVLM
from locus.models.draft_verify import DraftVerify
from locus.models.heads.box_decoder import upgrade_state_dict

REPO = Path(__file__).resolve().parents[1]
# BoxVLM constructor arguments a release needs (training-only options such as grad_ckpt are dropped)
ARCH = ["backbone", "px", "lora_r", "mid", "decoder_layers", "exit_layer", "n_query", "refine", "head", "dfl", "alora",
        "merger_lora", "full_ft"]


def export(run: str, out: Path) -> Path:
    res = json.loads((REPO / "results/box_train" / f"{run}.json").read_text())
    cfg = res["config"]
    sd = torch.load(REPO / "data/ckpt" / f"{run}.pt")
    flat = {"box_embed": sd["box_embed"].contiguous()}
    flat.update({f"decoder.{k}": v.contiguous() for k, v in upgrade_state_dict(sd["decoder"]).items()})
    flat.update({f"weights.{k}": v.contiguous() for k, v in sd.get("lora", {}).items()})
    out.mkdir(parents=True, exist_ok=True)
    save_file(flat, out / "model.safetensors")
    meta = dict(run=run, base_model=BACKBONES[cfg["backbone"]], arch={k: cfg[k] for k in ARCH if k in cfg},
                trained_params_m=res.get("trainable_params_m"), train_rows=res.get("train_rows"),
                epochs=res.get("epochs"), steps=cfg["steps"], seed=res.get("seed"), date=res.get("date"))
    (out / "locus_config.json").write_text(json.dumps(meta, indent=1))
    return out


def from_pretrained(path_or_repo: str | Path, device: str = "cuda") -> BoxVLM:
    p = Path(path_or_repo)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(str(path_or_repo)))
    a = json.loads((p / "locus_config.json").read_text())["arch"]
    m = BoxVLM(a["backbone"], device=device, max_pixels=a["px"] ** 2, lora_r=a["lora_r"], mid=a["mid"],
               decoder_layers=a["decoder_layers"], exit_layer=a.get("exit_layer", 1.0), n_query=a.get("n_query", 1),
               refine=a.get("refine", False), head=a.get("head", "xattn"), dfl=a.get("dfl", 0),
               alora=a.get("alora", False), merger_lora=a.get("merger_lora", False))
    sd = load_file(p / "model.safetensors")
    with torch.no_grad():
        m.box_embed.copy_(sd["box_embed"].reshape(m.box_embed.shape))
        m.decoder.load_state_dict({k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")})
        named = dict(m.ad.model.named_parameters())
        for k, v in sd.items():
            if k.startswith("weights."):
                named[k[len("weights."):]].copy_(v)
    return m.eval()


@torch.no_grad()
def ground(m: BoxVLM, image, expressions: list[str], accept: float = 1.0, verify: bool = True, max_passes: int = 8):
    """Ground expressions in one PIL image: one prefill, all expressions as branches. accept = 1 returns exactly the
    model's greedy coordinate-token box; accept < 1 is typical acceptance; verify=False returns the head's box."""
    m.ad.tok.padding_side = "left"
    pi = m.ad.processor(images=[image.convert("RGB")], text=[m.prefix], return_tensors="pt", **m.ad.proc_kwargs).to("cuda")
    dv = DraftVerify(m, max_passes=max_passes, verifier_adapter=True, accept=accept)
    with torch.autocast("cuda", torch.bfloat16):
        r = dv(pi, list(expressions), torch.zeros(len(expressions), dtype=torch.long, device="cuda"), verify=verify)
    px = lambda b: [float(v) * s for v, s in zip(b, [image.width, image.height] * 2)]
    return [dict(expression=e, box=px(r["box"][i] if verify else r["head"][i]), head_box=px(r["head"][i]),
                 confidence=float(r["conf"][i]), passes=int(r["passes"][i]) if verify else 0)
            for i, e in enumerate(expressions)]


if __name__ == "__main__":
    assert sys.argv[1] == "export" and len(sys.argv) == 4, __doc__
    print(export(sys.argv[2], Path(sys.argv[3])))
