# Reproduce reported scores (no proposer required)

Default config is `vl_harness/config_k40.yaml`: **K=40**, LVBench **num_val=350 / num_test=882**, seed **42**, temperature **0**.

Inner VLM: Qwen3-VL-8B-Instruct at `http://127.0.0.1:8080/v1`.  
CLIP: `clip-ViT-B-32` at `http://127.0.0.1:8181` (needed by `aks` and `embed_navigate_hybrid_iter2`).

Frozen scores: `paper/scores.json`. McNemar: `paper/mcnemar.json`.

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

## 3. k40 champion (hybrid)

Needs CLIP. Captions the 320-frame ingest pool (slow).

```bash
VL_LOG_CONTEXT=1 PYTHONPATH=.. python -m vl_harness.inner_loop \
  --memory agents/embed_navigate_hybrid_iter2.py \
  --dataset lvbench --seed 42 --mode offline \
  --num-train 0 --num-val 350 --num-test 0 \
  --frame-budget 40 \
  --model Qwen3-VL-8B-Instruct --api-base http://127.0.0.1:8080/v1 \
  --val-output logs/repro/hybrid/val.json \
  --log logs/repro/hybrid/log.jsonl
```

Search val was **170/350 = 48.6%**. The paper table uses a later re-eval **48.3%**.

## 4. AKS-run champion

```bash
VL_LOG_CONTEXT=1 PYTHONPATH=.. python -m vl_harness.inner_loop \
  --memory agents/timestamped_aks_iter5.py \
  --dataset lvbench --seed 42 --mode offline \
  --num-train 0 --num-val 350 --num-test 0 \
  --frame-budget 40 \
  --model Qwen3-VL-8B-Instruct --api-base http://127.0.0.1:8080/v1 \
  --val-output logs/repro/timestamped_aks/val.json \
  --log logs/repro/timestamped_aks/log.jsonl
```

Expect **182/350 = 52.0%**. Same val as the iter2 clock bypass; iter5 adds per-frame `[Xs]` labels.

## 5. Official held-out 882

Do **not** quote `accuracy` on a 1232-long `test.json` as held-out.

```bash
PYTHONPATH=.. python -m vl_harness.inner_loop \
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

Replace `--memory` with `aks.py`, `embed_navigate_hybrid_iter2.py`, or `timestamped_aks_iter5.py`.

## 6. McNemar

```bash
PYTHONPATH=.. python -m vl_harness.stats compare \
  logs/repro/uniform_k40/val.json \
  logs/repro/hybrid/val.json
```

Frozen discordant counts: `paper/mcnemar.json`.

## Split

`manifests/lvbench_split_seed42.json` lists every QA index after `random.Random(42).shuffle`. Changing the local video set changes the 1232 pool.
