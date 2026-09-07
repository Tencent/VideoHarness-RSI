#!/usr/bin/env bash
# Copy to setenv.sh (gitignored). Never commit real tokens.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VL_HARNESS_DATA="${VL_HARNESS_DATA:-$ROOT/data}"
export VL_HARNESS_CONFIG="${VL_HARNESS_CONFIG:-$ROOT/configs/config_k40.yaml}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/runs/results}"
# Context traces (val_contexts.jsonl) are on by default. Set 0 to disable.
export VL_LOG_CONTEXT="${VL_LOG_CONTEXT:-1}"
export PROPOSER_NUM_CANDIDATES="${PROPOSER_NUM_CANDIDATES:-1}"
mkdir -p "$VL_HARNESS_DATA" "$RESULTS_DIR"

# Proposer (evolution loop) runs on the official Claude Code CLI by default.
# Point CLAUDE_CLI_BIN at any Claude-Code-compatible launcher to relay elsewhere.
# export CLAUDE_CLI_BIN="claude"
# export PROPOSER_MODEL="claude-sonnet-4-6"
# export HF_TOKEN="hf_..."
# export OPENAI_API_KEY="..."
# export ANTHROPIC_API_KEY="..."

echo "[setenv] VL_HARNESS_DATA=$VL_HARNESS_DATA"
echo "[setenv] VL_HARNESS_CONFIG=$VL_HARNESS_CONFIG"
