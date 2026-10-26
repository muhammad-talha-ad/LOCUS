#!/usr/bin/env bash
# Draft-verify (max 8 Jacobi passes) on every RefCOCO/+/g eval split for one aLoRA/frozen run.
R=${1:?run}; shift  # extra args (e.g. --fast --gpu-preprocess) pass through
for s in refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test; do
  python scripts/eval_draft_verify.py --run $R --split $s --max-passes 8 --images-per-batch 2 "$@"
done
