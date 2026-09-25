#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -u paper/aistats_followup/h200_runner.py 2>&1 | tee -a paper/aistats_followup/h200_run.log
