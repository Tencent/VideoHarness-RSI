#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/vl_harness"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VL_HARNESS_CONFIG="${VL_HARNESS_CONFIG:-config_k40.yaml}"

python - <<'PY'
from vl_harness.agents.pilot_uniform_k import *
from vl_harness.agents.aks import *
from vl_harness.agents.embed_navigate_hybrid_iter2 import *
from vl_harness.agents.timestamped_aks_iter5 import *
print("import: pilot_uniform_k aks embed_navigate_hybrid_iter2 timestamped_aks_iter5 OK")
PY

python -m vl_harness.inner_loop \
  --memory agents/pilot_uniform_k.py \
  --dataset mock_niah --model stub --mode offline \
  --num-train 10 --num-val 40 --num-test 40 \
  --frame-budget 40 \
  --val-output /tmp/vlh-rsi/val.json --log /tmp/vlh-rsi/log.jsonl \
  --force

echo "smoke ok"
