#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-configs/local_cpu_test.json}"
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN=python3
fi

export PYTHONPATH="$REPO_ROOT/notebook_src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

run_adapter() {
  "$PYTHON_BIN" -m speech_adapter --config "$CONFIG" "$@"
}

echo "[1/7] Preflight"
run_adapter doctor
echo "[2/7] Mimi eight-codebook contract"
run_adapter codec-contract
echo "[3/7] Limited manifest"
run_adapter prepare-manifest
echo "[4/7] Plain-answer Qwen features"
run_adapter prepare-features
echo "[5/7] Causality, KV-cache, and CB0 checks"
run_adapter structural-checks
echo "[6/7] One bounded real-data optimizer/rollout check"
run_adapter smoke
echo "[7/7] Twenty-record, one-epoch training-loop check"
run_adapter train --experiment smoke_cpu --limit 20 --epochs 1

echo "Local smoke checks passed. Report: local_work/qwen_0.5b_cpu/smoke_report.json"
