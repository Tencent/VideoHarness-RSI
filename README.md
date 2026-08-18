# VL-Harness-RSI

**Code-level Recursive Self-Improvement (RSI) for long-video VLM harnesses.**

Freeze a vision-language model. Evolve the *Python harness around it* — what to store, how to retrieve, which frames to show — with a coding-agent proposer.

Default protocol: **K=40**, LVBench **val 350 / held-out 882**, seed **42**, temperature **0**. Config: `vl_harness/config_k40.yaml`.

This is a **1,232-QA subset** of LVBench (83 locally available videos). It is not the full LVBench release. Videos are not redistributed. LVBench annotations and `manifests/` are **CC-BY-NC-SA-4.0** (academic / non-commercial); see [`DATASETS.md`](DATASETS.md) and [`NOTICE`](NOTICE).

```text
Frozen VLM  +  evolvable VideoMemoryHarness  →  accuracy (optional cost Pareto)
```

## Reported scores (Qwen3-VL-8B-Instruct)

Frozen inner VLM. CLIP `clip-ViT-B-32` for AKS / hybrid. Exact counts: [`paper/scores.json`](paper/scores.json).

### LVBench subset (K=40)

| Harness | Dev 350 | Held-out 882 |
|---|---:|---:|
| Uniform `pilot_uniform_k` | 36.0 search / **36.3** table | **36.3** (320/882) |
| AKS `aks` | **49.7** | **46.5** (410/882) |
| k40 champion `embed_navigate_hybrid_iter2` | 48.6 search / **48.3** table | **45.4** (400/882) |
| AKS-run champion `timestamped_aks_iter5` | **52.0** | **50.3** (444/882) |

`timestamped_aks_iter5` is the *report* champion of the AKS-seeded run. The +2.3pp val jump vs AKS is `aks_anchor_bypass_iter2` (clock bypass). Iter5 keeps 52.0 and adds `[Xs]` timestamps; it is Pareto-preferred on visual tokens, not a second accuracy jump.

AKS here is a matched-K selector (CLIP-B/32, 320-frame @ 2 fps pool). It is not a reproduction of the AKS paper's BLIP-ITM / 1 fps tables. See the header of `vl_harness/agents/aks.py`.

### Transfer (same K=40, official test)

| Benchmark | n | Uniform | Hybrid | AKS |
|---|---:|---:|---:|---:|
| Video-MME | 2700 | 59.9 | 61.5 | 66.9 |
| MLVU | 2174 | 63.2 | 65.9 | 72.0 |

Hybrid transfer is small vs Uniform. AKS is stronger on these two sets. McNemar for the LVBench claims: [`paper/mcnemar.json`](paper/mcnemar.json).

## What is shipped

Runnable code:

- `VideoMemoryHarness` ingest-once / answer-many interface
- Floor: `agents/pilot_uniform_k.py`
- AKS: `agents/aks.py`
- k40-evolution champion: `agents/embed_navigate_hybrid_iter2.py`
- AKS-run champion: `agents/timestamped_aks_iter5.py` (imports `aks_select`)
- Search / eval / log: `meta_harness.py`, `inner_loop.py`, `stats.py`
- Smoke: `scripts/run_smoke.sh`

Not shipped (on purpose):

- The other 14 k40-evolution `.py` files (score table only: [`paper/search_k40.json`](paper/search_k40.json))
- Video files, sha256 checksums, per-video download-failure logs
- Full proposer sessions / `val.json` dumps

## Quick start (no GPU)

```bash
git clone https://github.com/X-G-Y/vl-harness-rsi.git
cd vl-harness-rsi
python -m venv .venv && source .venv/bin/activate
pip install -e ./vl_harness   # or: uv sync --project vl_harness

bash scripts/run_smoke.sh
```

## Reproduce paper scores

See [`REPRODUCE.md`](REPRODUCE.md). You need:

1. The 83 LVBench videos listed in `manifests/lvbench_videos.json` (download yourself; see `DATASETS.md`).
2. Frozen VLM at `http://127.0.0.1:8080/v1` (paper: Qwen3-VL-8B-Instruct).
3. CLIP embedder at `http://127.0.0.1:8181` for AKS and hybrid.

You do **not** need a proposer to re-score the frozen harnesses.

## Evolution loop

Requires a coding-agent CLI or LiteLLM proposer:

```bash
source setenv.sh
cd vl_harness
PYTHONPATH=.. python -m vl_harness.meta_harness --config config_k40.yaml
```

Phase 0 of that config evaluates Uniform and AKS only. Re-eval the two champions with `inner_loop` as in `REPRODUCE.md`.

## Layout

```text
vl_harness/           # package
  config_k40.yaml     # default paper protocol
  harness.py          # VideoMemoryHarness
  inner_loop.py / meta_harness.py
  agents/             # paper systems + seed sketches
manifests/            # 83/20 video IDs + 1232-QA split
paper/                # frozen scores, McNemar, 15-row search table
scripts/run_smoke.sh
```

## Naming

- **VL-Harness-RSI** — video-language harness + recursive self-improvement
- Distinct from Homer’s “LV-Harness” (prompt-level skills, not code-level RSI)

## Related work

- [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) — text harness RSI (outer loop; MIT, Copyright 2026 Yoonho Lee)
- AKS: Tang et al., CVPR 2025, https://github.com/ncTimTang/AKS (upstream publishes no SPDX license; our `agents/aks.py` is a protocol-matched reimplementation)

## License

- **Code** (original): [Apache-2.0](LICENSE)
- **Meta-Harness-derived loop files**: MIT; copyright notice in [NOTICE](NOTICE)
- **`manifests/`** (LVBench-derived split and video ids): [CC-BY-NC-SA-4.0](manifests/README.md), not Apache-2.0
- **Benchmarks and model weights**: not shipped; terms in [DATASETS.md](DATASETS.md)

Commercial use of the code does not grant commercial use of LVBench, Video-MME, or MLVU.
