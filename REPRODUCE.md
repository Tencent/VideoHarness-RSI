# Reproduce reported scores (no proposer required)

Default config is `vl_harness/config_k40.yaml`: **K=40**, LVBench **num_val=350 / num_test=882**, seed **42**, temperature **0**.

Inner VLM: Qwen3-VL-8B-Instruct at `http://127.0.0.1:8080/v1`.  
CLIP: `clip-ViT-B-32` at `http://127.0.0.1:8181` (needed by `aks`, `cardinality_ledger`, and AKS-90).

Frozen scores: `paper/scores.json`. McNemar: `paper/mcnemar.json`.

Packed-context logs for the table endpoints live in the supplementary archive (`dumps/table_dev350/*/val_contexts.jsonl`). Re-runs here should set `VL_LOG_CONTEXT=1`.

LVBench / Video-MME / MLVU are **not** Apache-2.0. Read [`DATASETS.md`](DATASETS.md) before downloading. Videos stay local; do not commit them.

## Environment

```bash
cp setenv.example.sh setenv.sh
source setenv.sh
cd vl_harness
```

`setenv.sh` sets `VL_HARNESS_CONFIG` to `config_k40.yaml`.

## 0. Smoke (no GPU, no videos)

```bash
bash ../scripts/run_smoke.sh
```

## 1. Uniform-40 (floor)

```bash
VL_LOG_CONTEXT=1 PYTHONPATH=.. python -m vl_harness.inner_loop \
  --memory agents/pilot_uniform_k.py \
  --dataset lvbench --seed 42 --mode offline \
  --num-train 0 --num-val 350 --num-test 0 \
  --frame-budget 40 \
  --model Qwen3-VL-8B-Instruct --api-base http://127.0.0.1:8080/v1 \
  --val-output logs/repro/uniform_k40/val.json \
  --log logs/repro/uniform_k40/log.jsonl
```

Expect ≈36.0–36.3% on the 350 (evolution seed 126/350; table 127/350).

## 2. AKS

Needs CLIP.

```bash
VL_LOG_CONTEXT=1 PYTHONPATH=.. python -m vl_harness.inner_loop \
  --memory agents/aks.py \
  --dataset lvbench --seed 42 --mode offline \
  --num-train 0 --num-val 350 --num-test 0 \
  --frame-budget 40 \
  --model Qwen3-VL-8B-Instruct --api-base http://127.0.0.1:8080/v1 \
  --val-output logs/repro/aks/val.json \
  --log logs/repro/aks/log.jsonl
```

Expect **174/350 = 49.7%**.

## 3. Uniform-seeded endpoint (StatedTimeAddressDecode / WeakFT)

```bash
VL_LOG_CONTEXT=1 PYTHONPATH=.. python -m vl_harness.inner_loop \
  --memory agents/stated_time_address_decode_iter9.py \
  --dataset lvbench --seed 42 --mode offline \
  --num-train 0 --num-val 350 --num-test 0 \
  --frame-budget 40 \
  --model Qwen3-VL-8B-Instruct --api-base http://127.0.0.1:8080/v1 \
  --val-output logs/repro/weakft/val.json \
  --log logs/repro/weakft/log.jsonl
```

Expect **178/350 = 50.9%**. Held-out is **432/882 = 49.0%**.

## 4. AKS-seeded endpoint (CardinalityLedger)

Needs CLIP. Multi-stage; cumulative frames ≈ 88–90.

```bash
VL_LOG_CONTEXT=1 PYTHONPATH=.. python -m vl_harness.inner_loop \
  --memory agents/cardinality_ledger.py \
  --dataset lvbench --seed 42 --mode offline \
  --num-train 0 --num-val 350 --num-test 0 \
  --frame-budget 40 \
  --model Qwen3-VL-8B-Instruct --api-base http://127.0.0.1:8080/v1 \
  --val-output logs/repro/cardinality_ledger/val.json \
  --log logs/repro/cardinality_ledger/log.jsonl
```

Expect **204/350 = 58.3%**. Held-out is **483/882 = 54.8%**.

## 5. Official held-out 882

Do **not** quote `accuracy` on a 1232-long `test.json` as held-out.

```bash
VL_LOG_CONTEXT=1 PYTHONPATH=.. python -m vl_harness.inner_loop \
  --memory agents/pilot_uniform_k.py \
  --dataset lvbench --seed 42 --mode offline \
  --num-train 0 --num-val 0 --num-test 1232 \
  --frame-budget 40 \
  --model Qwen3-VL-8B-Instruct --api-base http://127.0.0.1:8080/v1 \
  --test-output logs/repro/uniform_k40/test1232.json
```

```python
import json
d = json.load(open("logs/repro/uniform_k40/test1232.json"))
held = d["results"][350:]
print(sum(r["was_correct"] for r in held), "/", len(held))
```

Replace `--memory` with `aks.py`, `stated_time_address_decode_iter9.py`, or `cardinality_ledger.py`. For AKS-90 use `aks.py` with `--frame-budget 90`.

The paper's AKS held-out row is the **411/882** paired re-eval, not an older 410/882 dump.

## 6. McNemar

```bash
PYTHONPATH=.. python -m vl_harness.stats compare \
  logs/repro/aks/val.json \
  logs/repro/cardinality_ledger/val.json
```

Frozen discordant counts (including held-out 882): `paper/mcnemar.json`.

## Split

`manifests/lvbench_split_seed42.json` lists every QA index after `random.Random(42).shuffle`. Changing the local video set changes the 1232 pool.
