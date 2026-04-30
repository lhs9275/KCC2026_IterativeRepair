#!/bin/bash
# Sequential seed=123 experiment runner: EA → BR
# Path A: both strategies on test-fail subgroup (N=139)

set -euo pipefail

cd "$(dirname "$0")"

TEST_FAIL_IDS="1,3,4,5,8,9,12,13,20,22,24,25,27,32,33,35,36,37,40,41,42,46,47,49,50,52,53,54,56,57,58,59,60,62,63,66,67,68,70,72,74,75,78,80,81,83,84,85,86,87,88,91,92,93,94,96,97,98,99,101,103,104,107,109,112,113,115,116,117,118,120,122,126,129,130,133,135,136,137,138,139,141,142,146,148,150,151,152,153,154,155,157,158,159,160,161,162,163,167,168,170,175,181,183,184,186,188,189,192,194,195,198,202,204,210,214,216,217,220,221,222,223,227,228,230,232,237,238,240,241,242,243,244,245,246,248,252,253,255"

echo "[$(date)] === Run seed=123 Path A START ==="

echo "[$(date)] --- Phase EA: error_aware seed=123 ---"
START_EA=$(date +%s)
bash run.sh \
  --config config/ablation_error_aware_seed123.yaml \
  --bug_id_list "$TEST_FAIL_IDS" \
  --output Results/qwen_error_aware_seed123.json \
  2>&1 | tee logs/ea_seed123.log
EA_EXIT=${PIPESTATUS[0]}
END_EA=$(date +%s)
echo "[$(date)] EA finished. Exit code: $EA_EXIT. Elapsed: $(( (END_EA - START_EA) / 60 ))m"

echo "[$(date)] --- Phase BR: blind_retry seed=123 ---"
START_BR=$(date +%s)
bash run.sh \
  --config config/ablation_blind_retry_seed123.yaml \
  --bug_id_list "$TEST_FAIL_IDS" \
  --output Results/qwen_blind_retry_seed123.json \
  2>&1 | tee logs/br_seed123.log
BR_EXIT=${PIPESTATUS[0]}
END_BR=$(date +%s)
echo "[$(date)] BR finished. Exit code: $BR_EXIT. Elapsed: $(( (END_BR - START_BR) / 60 ))m"

echo "[$(date)] === Run seed=123 Path A DONE (EA=$EA_EXIT, BR=$BR_EXIT) ==="
