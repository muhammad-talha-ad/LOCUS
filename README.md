# Locus

**Locate Once, Check Until Settled.** Draft-and-verify referring-expression grounding for vision-language models.

A VLM can ground an expression in two ways. It can write the box as coordinate tokens, which is precise but takes
about 18.46 sequential decoding steps per box. Or it can regress the box from one hidden state with a small
head, which takes one step but places edges less exactly. Locus uses the head as a speculative drafter for the model's
own coordinate tokens. The draft is rendered as the tokens the model would write, and the same model checks all of them
in parallel by Jacobi iteration, keeping the prefix it agrees with and correcting the next token until the draft stops
changing. With exact acceptance the result is the model's greedy coordinate-token box, reached in
5.60 passes on average.

```
image + "the man in the red shirt"
   │
   ├─ prefill once ──────────────── image KV cache, shared by every expression about this image
   │
   ├─ box branch ─── MLP head ───── draft box ──► rendered as coordinate tokens  "312, 88, 540, 702]"
   │
   └─ verifier branch (Jacobi) ──── keep the longest agreed prefix, correct the next token, repeat
                                    until the draft stops changing ──► the model's greedy box
```

Head and verifier are one set of weights, trained jointly on RefCOCO, RefCOCO+ and RefCOCOg.

## Results

Mean over the 8 RefCOCO / RefCOCO+ / RefCOCOg val and test splits. Every row was run in this repository's harness on
the same rows; baselines use their released weights. Locus runs at its 640² training budget, Jag at 512² and the
other baselines at their native resolution.

| Model | Output | Acc@0.5 | Acc@0.75 | Acc@0.9 | Passes |
|---|---|---|---|---|---|
| Locus-2B | head (one pass) | 90.01 | 79.79 | 54.87 | 1 |
| Locus-2B | verified, exact | 89.30 | 80.31 | 59.13 | 5.60 |
| Locus-2B | verified, typical (α = 0.5) | 89.60 | 80.49 | 58.95 | 3.43 |
| Locus-0.8B | head (one pass) | 87.97 | 75.58 | 47.03 | 1 |
| Locus-0.8B | verified, exact | 87.05 | 75.93 | 53.10 | 6.68 |
| Locus-0.8B | verified, typical (α = 0.7) | 87.17 | 76.14 | 53.29 | 4.37 |
| Jag-0.8B | regression head | 88.16 | 76.51 | 48.76 | 1 |
| LocateAnything-3B | box tokens | 88.68 | 80.82 | 65.55 | – |
| Hi-Token-3B | digit tokens | 87.24 | 76.93 | 54.75 | – |

- On the same weights, verification raises Acc@0.9 over the model's own head, at the cost of a few extra passes. The
  head is the more accurate output at Acc@0.5.
- Locus-0.8B uses the backbone and 2-epoch training budget of Jag-0.8B. It ties Jag at Acc@0.5 and, with typical
  acceptance, gains +4.52 points at Acc@0.9 (paired McNemar tests and an image-level bootstrap).
- LocateAnything-3B, a larger model trained on far more grounding data, is more accurate at high IoU.

**Evaluation hygiene.** The RefCOCO-family splits are drawn per dataset, so 46.9% of evaluation rows ask about
an object that is in another dataset's training split. Locus is trained with every evaluation image removed, and
`paper/tables/mcnemar.py --clean` repeats every comparison on the rows whose target no model has seen.

### Latency

One RTX A6000, batch 1, 512² inputs, end to end from the image file to the parsed box, 256 RefCOCO testA questions.
Jag, Locus and vLLM use CUDA graphs and GPU preprocessing.

| System | ms per question |
|---|---|
| Jag-0.8B, regression head | 20.3 |
| Locus-0.8B, head / exact | 25.5 / 63.5 |
| Locus-2B, head / typical (α = 0.5) / exact | 43.5 / 84.3 / 102.2 |
| Locus-2B weights decoding the same coordinate tokens in vLLM | 151.6 |
| Qwen3-VL-2B writing its full JSON answer in vLLM | 297.2 |

Exact verification returns the same box as greedy decoding 1.5× faster than vLLM decoding
those tokens with the same weights. A pure regression head is faster still.

## Install

```bash
pip install -e .               # Python 3.12 and a CUDA GPU; versions in pyproject.toml
python scripts/setup_data.py   # RefCOCO/+/g annotations, COCO-2017 train images (~18 GB), Jag baseline weights
```

`setup_data.py` is only needed to train or evaluate; inference with a released model needs `pip install -e .` alone.

## Inference with a released model

```python
from PIL import Image
from locus.hub import from_pretrained, ground

m = from_pretrained("<org>/Locus-2B")   # trained weights from this repo; the base VLM from its own Hub repo
image = Image.open("street.jpg")
for r in ground(m, image, ["the man in the red shirt", "the bus on the left"]):
    print(r["expression"], r["box"], r["passes"])
```

All expressions about one image share a single image prefill. Each result holds:

| Key | Meaning |
|---|---|
| `box` | answer box `[x0, y0, x1, y1]` in pixels |
| `head_box` | the head's one-pass draft |
| `confidence` | the head's IoU-aware confidence |
| `passes` | verifier passes used (0 with `verify=False`) |

Options:
- `accept=1.0` (default): exact acceptance; the box is the model's greedy coordinate-token answer. In bf16 this holds
  up to the same rounding that already makes two greedy runs with different batch shapes differ.
- `accept=0.7` or `0.5`: typical acceptance; fewer passes, no longer exactly greedy.
- `verify=False`: return the head's draft only (one pass, fastest).

## Reproduce

| What | Command |
|---|---|
| Locus-2B (train + evaluate) | `bash scripts/reproduce/locus_2b.sh` |
| Locus-0.8B (train + evaluate) | `bash scripts/reproduce/locus_08b.sh` |
| Jag-0.8B baseline | `bash scripts/reproduce/baselines.sh` |
| Latency | `bash scripts/reproduce/latency.sh` |
| Tables and significance tests | `bash scripts/reproduce/tables.sh` |

Training uses the RefCOCO, RefCOCO+ and RefCOCOg train splits with every image that appears in an evaluation split
removed. Every run writes its config, seed and git commit to `results/`.

## Limitations

- Trained on the RefCOCO family only. On Ref-L4's Objects365 images, whose targets are much smaller than any RefCOCO
  training target, Locus and Jag both fall far below the zero-shot base model.
- Slower than a pure regression head; verification adds passes in proportion to how often the draft is wrong.
- At 0.8B, exact verification lowers Acc@0.5 relative to the head.
- Released backbones stop at 2B.

## License

Code: Apache-2.0. Released weights are derived from Qwen3-VL-2B-Instruct and Qwen3.5-0.8B (Apache-2.0) and were
trained on RefCOCO/+/g annotations over COCO images; check those datasets' terms for your use.

## Acknowledgements

The box head follows [Jag](https://arxiv.org/abs/2610.13239); the backbones are from the Qwen team.

## Citation

Paper under review; citation to follow.
