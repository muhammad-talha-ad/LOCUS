#!/usr/bin/env bash
# Every table from results/ (markdown on stdout; McNemar and selector also write JSON under results/).
set -e
python paper/tables/draft_verify.py dual_vl2b_mlp_ar1_merger --tag _fast_graph_gpupre
python paper/tables/draft_verify.py dual_q35_08b_mlp_ar1_fullft_36k
python paper/tables/mcnemar.py dual_q35_08b_mlp_ar1_fullft_36k --typical 8a0.7
python paper/tables/selector.py dual_vl2b_mlp_ar1_merger --tag _fast_graph_gpupre
python paper/tables/mcnemar.py dual_vl2b_mlp_ar1_merger --tag _fast_graph_gpupre --typical 8a0.7
# the same tests on rows whose target object is in no RefCOCO-family train split
python paper/tables/mcnemar.py dual_q35_08b_mlp_ar1_fullft_36k --typical 8a0.7 --clean
python paper/tables/mcnemar.py dual_vl2b_mlp_ar1_merger --tag _fast_graph_gpupre --typical 8a0.7 --clean
# 2B full fine-tune (alpha 0.5 chosen on RefCOCO val by paper_numbers.choose)
python paper/tables/mcnemar.py dual_vl2b_mlp_ar1_fullft_36k --tag _fast_graph_gpupre --typical 8a0.5
python paper/tables/mcnemar.py dual_vl2b_mlp_ar1_fullft_36k --tag _fast_graph_gpupre --typical 8a0.5 --clean
