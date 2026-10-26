#!/usr/bin/env bash
# Locus-2B (paper Table 1): Qwen3-VL-2B, dual mode, full fine-tuning of the language model and vision merger (lr 1e-5),
# 36k steps of 4 images x 4 expressions = 2.00 epochs (Jag's budget), then draft-and-verify on the 8 RefCOCO/+/g splits
# (exact and typical acceptance). One 48 GB GPU (fused AdamW, 2 micro-batches of 2 images).
set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True
python scripts/train_box.py --backbone qwen3-vl-2b --lora-r 0 --full-ft --lr-lora 1e-5 --fused-adam --accum 2 --steps 36000 \
  --shared --bs 4 --q 4 --head mlp --ar-weight 1.0 --name dual_vl2b_mlp_ar1_fullft_36k
bash scripts/dv_all_splits.sh dual_vl2b_mlp_ar1_fullft_36k --fast --graphs --graph-prefill --gpu-preprocess --accept 1 0.7 0.5 0.3
# Ablation table: the 12k-step LoRA variant, r = 64 + vision-merger LoRA
python scripts/train_box.py --backbone qwen3-vl-2b --lora-r 64 --steps 12000 --shared --bs 4 --q 4 --head mlp \
  --ar-weight 1.0 --merger-lora --name dual_vl2b_mlp_ar1_merger
bash scripts/dv_all_splits.sh dual_vl2b_mlp_ar1_merger --fast --graphs --gpu-preprocess --accept 1 0.7 0.5 0.3
