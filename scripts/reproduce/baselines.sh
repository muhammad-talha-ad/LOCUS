#!/usr/bin/env bash
# Jag-0.8B (released weights, data/external/jag_repo) on the same 8 splits and rows as Locus.
set -e
python scripts/eval_baselines.py --model jag --splits refcoco_val refcoco_testA refcoco_testB refcoco+_val \
  refcoco+_testA refcoco+_testB refcocog_val refcocog_test
