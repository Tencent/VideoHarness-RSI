# `vl_harness` package

Core library for **VL-Harness-RSI**: evolvable long-video memory harnesses around a frozen VLM.

Default config: **`config_k40.yaml`** (K=40, LVBench val 350 / test 882, seed 42).

## Layout

- `vlm.py` — multimodal VLM client + embedder (+ StubVLM)
- `harness.py` — `VideoMemoryHarness` base
- `video.py` — frame decode / sampling / `frame_budget`
- `data.py` / `loaders_real.py` — episodes + MCQ eval
- `inner_loop.py` / `benchmark.py` / `meta_harness.py` — eval + RSI outer loop
- `agents/` — paper systems (`pilot_uniform_k`, `aks`, `stated_time_address_decode_iter9` / WeakFT, `cardinality_ledger`) plus seed sketches
- `.claude/skills/vl-harness/SKILL.md` — proposer prior

## Smoke

From the repo root: `bash scripts/run_smoke.sh`

Code is Apache-2.0; Meta-Harness-derived files are MIT (see `../NOTICE`).
`../manifests/` is CC-BY-NC-SA-4.0. Dataset terms: `../DATASETS.md`.

```bash
PYTHONPATH=.. python -m vl_harness.inner_loop \
  --memory agents/pilot_uniform_k.py \
  --dataset mock_niah --model stub --mode offline \
  --num-train 10 --num-val 40 --num-test 40 \
  --frame-budget 40 \
  --val-output /tmp/vlh/val.json --log /tmp/vlh/log.jsonl
```
