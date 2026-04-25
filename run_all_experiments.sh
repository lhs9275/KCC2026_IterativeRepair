#!/bin/bash
# Run all 5 experiments sequentially (single GPU)
# Usage: bash run_all_experiments.sh

set -eo pipefail

# Environment setup (same as run.sh)
source /home/selab/miniconda3/etc/profile.d/conda.sh
conda activate kcc
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="$JAVA_HOME/bin:/home/selab/EMSE/defects4j/framework/bin:$PATH"
cd "$(dirname "$0")"

BUGS=$(seq -s, 1 255)
LOG_DIR="logs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Starting all experiments at $(date) ==="
echo "Log dir: $LOG_DIR"

# 1. Qwen one_shot
echo "[1/5] Qwen one_shot — $(date)"
python 5.IterativeRepair.py --config config/ablation_1shot.yaml --bug_id_list "$BUGS" --max_model_len 4096 \
  2>&1 | tee "$LOG_DIR/1_qwen_1shot.log"
echo "[1/5] Done — $(date)"

# 2. Qwen blind_retry
echo "[2/5] Qwen blind_retry — $(date)"
python 5.IterativeRepair.py --config config/ablation_blind_retry.yaml --bug_id_list "$BUGS" --max_model_len 4096 \
  2>&1 | tee "$LOG_DIR/2_qwen_blind.log"
echo "[2/5] Done — $(date)"

# 3. Qwen error_aware
echo "[3/5] Qwen error_aware — $(date)"
python 5.IterativeRepair.py --config config/ablation_error_aware.yaml --bug_id_list "$BUGS" --max_model_len 4096 \
  2>&1 | tee "$LOG_DIR/3_qwen_error_aware.log"
echo "[3/5] Done — $(date)"

# 4. DeepSeek one_shot
echo "[4/5] DeepSeek one_shot — $(date)"
python 5.IterativeRepair.py --config config/ablation_1shot_deepseek.yaml --bug_id_list "$BUGS" --max_model_len 4096 \
  2>&1 | tee "$LOG_DIR/4_deepseek_1shot.log"
echo "[4/5] Done — $(date)"

# 5. DeepSeek error_aware
echo "[5/5] DeepSeek error_aware — $(date)"
python 5.IterativeRepair.py --config config/ablation_error_aware_deepseek.yaml --bug_id_list "$BUGS" --max_model_len 4096 \
  2>&1 | tee "$LOG_DIR/5_deepseek_error_aware.log"
echo "[5/5] Done — $(date)"

echo "=== All experiments completed at $(date) ==="
