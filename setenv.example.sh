#!/usr/bin/env bash
# Copy to setenv.sh (gitignored). Never commit real tokens.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VL_HARNESS_DATA="${VL_HARNESS_DATA:-$ROOT/data}"
export VL_HARNESS_CONFIG="${VL_HARNESS_CONFIG:-$ROOT/vl_harness/config_k40.yaml}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/vl_harness/results}"
mkdir -p "$VL_HARNESS_DATA" "$RESULTS_DIR"

# export HF_TOKEN="hf_..."
# export OPENAI_API_KEY="..."
# export ANTHROPIC_API_KEY="..."
# export CLAUDE_CLI_BIN="claude"

echo "[setenv] VL_HARNESS_DATA=$VL_HARNESS_DATA"
echo "[setenv] VL_HARNESS_CONFIG=$VL_HARNESS_CONFIG"
