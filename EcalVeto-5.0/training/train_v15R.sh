#!/bin/bash

set -e
cd /home/jihoonyoo/LDMX/v4.5.11/ldmx-sw/LDMX-scripts/EcalVeto-4.0
R=/home/jihoonyoo/LDMX/8GeV_Samples/v15_pres_reco
BASE="-s $R/train_sig2 -b $R/train_bkg --max_evt 3300000 --bkg_max_evt 3300000 \
  --bkg_branch EcalVeto_reco_v1 --sig_branch EcalVeto_reco_v1 \
  --train_frac 0.9 --subsample 0.8 --colsample_bytree 0.75 --scale_pos_weight 1.0"

case "$1" in
  base)
    denv python3 bdtMaker.py $BASE --depth 6 --min_child_weight 60 --gamma 0.5 --reg_lambda 1.5 \
      --eta 0.05 --tree_number 4000 --early_stopping_rounds 100 -o V15R_base ;;
  ll)
    denv python3 bdtMaker.py $BASE --depth 6 --min_child_weight 60 --gamma 0.5 --reg_lambda 1.5 \
      --eta 0.05 --tree_number 15000 --early_stopping_rounds 200 --eval_metric logloss -o V15R_ll ;;
  lleta02)
    denv python3 bdtMaker.py $BASE --depth 6 --min_child_weight 60 --gamma 0.5 --reg_lambda 1.5 \
      --eta 0.02 --tree_number 30000 --early_stopping_rounds 300 --eval_metric logloss -o V15R_lleta02 ;;
  lld8)
    denv python3 bdtMaker.py $BASE --depth 8 --min_child_weight 60 --gamma 0.5 --reg_lambda 1.5 \
      --eta 0.05 --tree_number 10000 --early_stopping_rounds 200 --eval_metric logloss -o V15R_lld8 ;;
  *) echo "usage: $0 {base|ll|lleta02|lld8}"; exit 1 ;;
esac
echo "DONE $1 $(date)"
