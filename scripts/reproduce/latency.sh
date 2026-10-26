#!/usr/bin/env bash
# Paper Table 2: batch-1 end-to-end latency on the same 256 RefCOCO testA images, 512^2 pixels, GPU preprocessing,
# every system CUDA-graphed with all graphs captured before timing. Run on an otherwise idle GPU.
set -e
(cd scripts && python eval_latency.py --model jag --px 262144 --gpu-preprocess --graphs --per-image)
for run in dual_vl2b_mlp_ar1_fullft_36k dual_q35_08b_mlp_ar1_fullft_36k; do
  python scripts/eval_draft_verify.py --run $run --split refcoco_testA --latency 256 --max-passes 8 --accept 1 0.7 0.5 \
    --fast --graphs --graph-prefill --gpu-preprocess --px 262144
done
# Latency vs questions per image (Table: Q = 1, 2, 4, 8 on the first 128 + 10 testA images with >= 8 expressions)
python scripts/eval_draft_verify.py --run dual_vl2b_mlp_ar1_fullft_36k --split refcoco_testA --latency 128 --latency-q 1 2 4 8 \
  --max-passes 8 --accept 1 0.7 0.5 --fast --graphs --graph-prefill --gpu-preprocess --px 262144
(cd scripts && python eval_latency.py --model jag --px 262144 --gpu-preprocess --graphs --n 128 --q 1 2 4 8)
