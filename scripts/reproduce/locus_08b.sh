#!/usr/bin/env bash
# Locus-0.8B (paper Table 1, equal-size comparison): Qwen3.5-0.8B (Jag's backbone), dual mode, full fine-tuning of the
# LM + vision merger at lr 1e-5, 36k steps = 2.00 epochs (Jag's budget), then draft-and-verify on the 8 splits.
set -e
python scripts/train_box.py --backbone qwen3.5-0.8b --lora-r 0 --full-ft --lr-lora 1e-5 --steps 36000 --shared --bs 4 \
  --q 4 --head mlp --ar-weight 1.0 --name dual_q35_08b_mlp_ar1_fullft_36k
bash scripts/dv_all_splits.sh dual_q35_08b_mlp_ar1_fullft_36k --accept 1 0.7 0.5 0.3
