#!/bin/bash
# Environment setup for KCC2026 Iterative Repair experiments

# Activate conda env FIRST (it resets PATH)
source /home/selab/miniconda3/etc/profile.d/conda.sh
conda activate kcc

# THEN set Java and Defects4J paths (after conda, so they survive)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH="$JAVA_HOME/bin:/home/selab/EMSE/defects4j/framework/bin:$PATH"

# Verify
java -version 2>&1 | head -1
which defects4j
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"

# Run experiment
cd "$(dirname "$0")"
python 5.IterativeRepair.py "$@"
