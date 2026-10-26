"""Gradio demo: ground expressions in an uploaded image with a released model.

    pip install -e ".[demo]"
    python scripts/demo.py       # then open http://127.0.0.1:7860
"""
from functools import cache

import gradio as gr

from locus.hub import from_pretrained, ground

MODELS = ["LOCUS-2B", "LOCUS-0.8B"]
load = cache(lambda name: from_pretrained(f"MTalha2001/{name}"))  # each model loads on first use
MODES = {"Verified, exact": dict(accept=1.0), "Verified, typical (α = 0.5)": dict(accept=0.5),
         "Head only (one pass)": dict(verify=False)}


def run(image, text, model, mode):
    exprs = [e.strip() for e in text.splitlines() if e.strip()]
    if image is None or not exprs:
        raise gr.Error("Upload an image and enter at least one expression.")
    out = ground(load(model), image, exprs, **MODES[mode])
    boxes = [(tuple(int(v) for v in r["box"]), r["expression"]) for r in out]
    rows = [[r["expression"], [round(v) for v in r["box"]], round(r["confidence"], 3), r["passes"]] for r in out]
    return (image, boxes), rows


with gr.Blocks(title="LOCUS") as demo:
    gr.Markdown("# LOCUS: Locate Once, Check Until Settled\n"
                "A box head drafts each box; the same model verifies it as coordinate tokens by Jacobi iteration. "
                "[Code](https://github.com/muhammad-talha-ad/LOCUS) · "
                "[Paper](https://muhammad-talha-ad.github.io/LOCUS/static/locus.pdf)")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Image")
            text = gr.Textbox(label="Expressions (one per line)", lines=3, placeholder="the cat on the left")
            model = gr.Radio(MODELS, value="LOCUS-2B", label="Model")
            mode = gr.Radio(list(MODES), value="Verified, exact", label="Output")
            btn = gr.Button("Ground", variant="primary")
        with gr.Column():
            result = gr.AnnotatedImage(label="Boxes")
            table = gr.Dataframe(headers=["expression", "box [x0, y0, x1, y1]", "confidence", "passes"],
                                 label="Results")
    gr.Examples([["http://images.cocodataset.org/val2017/000000039769.jpg", "the cat on the left\nthe remote on the right",
                  "LOCUS-2B", "Verified, exact"]], [image, text, model, mode])
    btn.click(run, [image, text, model, mode], [result, table])

demo.launch()
