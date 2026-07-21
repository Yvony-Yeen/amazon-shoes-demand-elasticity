#!/bin/bash
# run_lora_code.sh -- run the code/ Phase-3 pipeline for all 14 *_lora buckets
# to produce LoRA demand elasticity (txtimg + txt), keeping vartheta = 0.6.
#
# Usage:
#   bash run_lora_code.sh            # run both rounds (txtimg then txt)
#   bash run_lora_code.sh --dry-run  # just print the plan, run nothing
#
# Notebooks run with kernel "shoes-venv" (driver default). CPU only, no GPU.
# Safe to delete after use.

cd /Users/yvony/PycharmProjects/Amazon_demand_analysis_seperate || exit 1

BUCKETS="men_adidas_lora,men_asics_lora,men_heydude_lora,men_olukai_lora,men_reebok_lora,men_rest_lora,men_top2_nike_lora,women_adidas_lora,women_asics_lora,women_dr_scholl_s_shoes_lora,women_heydude_lora,women_nike_lora,women_reebok_lora,women_rest_lora"

EXTRA="--continue-on-error --allow-errors"
if [ "$1" = "--dry-run" ]; then
  EXTRA="--dry-run"
fi

echo "############################################################"
echo "# ROUND 1 / 2  (txtimg): 01_1 + 01_2 + 04_evaluation_final_all"
echo "############################################################"
python run_code_pipeline.py --buckets "$BUCKETS" \
  --steps 01_1,01_2,04_evaluation_final_all --skip-vartheta $EXTRA

echo
echo "############################################################"
echo "# ROUND 2 / 2  (txt): 04_evaluation_final_all --txt-only"
echo "############################################################"
python run_code_pipeline.py --buckets "$BUCKETS" \
  --steps 04_evaluation_final_all --txt-only --skip-vartheta $EXTRA

echo
echo "############################################################"
echo "# ALL DONE. Elasticity logs are under:"
echo "#   data/<bucket>_lora/_run_logs/04_evaluation_final_all.ipynb      (txtimg)"
echo "#   data/<bucket>_lora/_run_logs/04_evaluation_final_all_txt.ipynb  (txt)"
echo "############################################################"
