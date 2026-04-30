#!/bin/bash
# Environment setup for KCC2026 Iterative Repair experiments
#
# Configurable via environment variables (override defaults below):
#   CONDA_BASE   - path to miniconda/anaconda root  (default: ~/miniconda3)
#   CONDA_ENV    - conda environment name            (default: kcc)
#   JAVA_HOME    - Java 11 home                      (default: /usr/lib/jvm/java-11-openjdk-amd64)
#   D4J_HOME     - Defects4J root directory          (default: ~/defects4j)

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-kcc}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
D4J_HOME="${D4J_HOME:-$HOME/defects4j}"

# Activate conda env FIRST (it resets PATH)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# THEN set Java and Defects4J paths (after conda, so they survive)
export PATH="$JAVA_HOME/bin:$D4J_HOME/framework/bin:$PATH"

# Verify
java -version 2>&1 | head -1
which defects4j
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"

# Run experiment
cd "$(dirname "$0")"
python 5.IterativeRepair.py "$@"
