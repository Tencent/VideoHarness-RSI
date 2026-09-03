# VL-Harness-RSI

**Code-level Recursive Self-Improvement (RSI) for long-video VLM harnesses.**

Freeze a vision-language model. Evolve the Python harness around it — what to
store, how to retrieve, which frames to show — with a coding-agent proposer.

Default protocol: **K=40**, LVBench **val 350 / held-out 882**, seed **42**,
temperature **0**. Config: `vl_harness/config_k40.yaml`.

This is a **1,232-QA subset** of LVBench (83 locally available videos). It is
not the full LVBench release. Videos are not redistributed. LVBench annotations
and `manifests/` are CC-BY-NC-SA-4.0 (academic / non-commercial); see
[`DATASETS.md`](DATASETS.md) and [`NOTICE`](NOTICE).

```text
Frozen VLM  +  evolvable VideoMemoryHarness  →  accuracy (optional cost Pareto)
```

## This repository vs the supplement

| | This git repo | Supplementary pack |
|---|---|---|
| What | Runnable code, frozen score JSON, split manifests | Per-question dumps, packed-context logs, table-only search trajectories |
| How to get it | `git clone` | Separate download (Zenodo / arXiv ancillary). **Not in this tree.** |
| Size | Small | ~25 MB compressed; do not commit it here |

After unpacking the supplement, start at `dumps/CONTEXTS.md`.
`val_contexts.jsonl` is the complete context-audit source;
`pack_index.jsonl` is a truncated index and cannot reconstruct full prompts.

Supplement DOI: *TODO — add after Zenodo publish.*

Re-score a frozen harness from this repo; you do **not** need the outer
proposer, and proposer sampling is not bit-reproducible.

## What is this?

The VLM parameters stay frozen (default **Qwen3-VL-8B-Instruct**). Search
rewrites the read-path around the video: ingest, retrieve, keyframe selection,
and prompting. Each candidate is a Python class:

```python
class MyHarness(VideoMemoryHarness):
    def build_memory(self, video):
        ...
    def answer_question(self, memory, question, options):
        ...
```

The loop discovers the file, evaluates it, records accuracy and visual-token
cost, and proposes the next generation.

## Reported scores (Qwen3-VL-8B-Instruct)

Frozen inner VLM. CLIP `clip-ViT-B-32` for AKS / Ledger / AKS-90.
Exact counts: [`paper/scores.json`](paper/scores.json).

AKS held-out is the **411/882** paired re-eval used with CardinalityLedger.
Do not mix with an older unpaired 410/882 dump.

### LVBench subset (K=40)

| Harness | Dev 350 | Held-out 882 |
|---|---:|---:|
| Uniform `pilot_uniform_k` | 36.0 search / 36.3 table | 36.3 (320/882) |
| AKS `aks` | 49.7 (174/350) | **46.6 (411/882)** |
| WeakFT `stated_time_address_decode_iter9` | 50.9 (178/350) | 49.0 (432/882) |
| AKS-90 (same `aks.py`, `--frame-budget 90`) | 53.1 (186/350) | 49.2 (434/882) |
| **CardinalityLedger** `cardinality_ledger` | **58.3 (204/350)** | **54.8 (483/882)** |

- **WeakFT** is the uniform-seeded endpoint (search iter 9).
- **CardinalityLedger** is the AKS-seeded endpoint (search iter 7).
- AKS-90 is a matched cumulative visual-token control, not a searched program.
- Uniform search val is 126/350; the paper table uses a later 127/350 re-eval.

Paired McNemar on held-out 882, AKS → CardinalityLedger:
**63 / 135**, exact two-sided *p* = 3.39×10⁻⁷.
See [`paper/mcnemar.json`](paper/mcnemar.json) and [`paper/bootstrap.json`](paper/bootstrap.json).

AKS here is a matched-K selector (CLIP-B/32, 320-frame @ 2 fps pool). It is
**not** a reproduction of the AKS paper's BLIP-ITM / 1 fps tables. See the
header of `vl_harness/agents/aks.py`.

### Transfer (frozen programs, official tests)

| Benchmark | n | Uniform | AKS | CardinalityLedger |
|---|---:|---:|---:|---:|
| Video-MME | 2700 | 59.9 | 66.9 (1805) | **67.5 (1823)** |
| MLVU | 2174 | 63.2 | 72.0 (1566) | **75.0 (1631)** |

Prediction dumps for AKS and Ledger transfer live in the supplement
(`dumps/aks/video_mme.json`, `dumps/cardinality_ledger/mlvu.json`, …).

Hybrid / timestamped-AKS were earlier table endpoints; they are not the
current main-table programs. Historical McNemar: `paper/mcnemar_legacy_hybrid_timestamped.json`.

## What is shipped here

Runnable code:

- `VideoMemoryHarness` ingest-once / answer-many interface
- Floor: `agents/pilot_uniform_k.py`
- AKS: `agents/aks.py` (AKS-90 = same file, `--frame-budget 90`)
- Uniform-seeded endpoint: `agents/stated_time_address_decode_iter9.py`
- AKS-seeded endpoint: `agents/cardinality_ledger.py`
- Search loop: `meta_harness.py`, `inner_loop.py`, `stats.py`
- Smoke: `scripts/run_smoke.sh`
- Frozen aggregates: `paper/scores.json`, `paper/mcnemar.json`

Offline checks (need the supplement unpacked as `--archive`):

```bash
python3 scripts/verify_main_scores.py --archive /path/to/videoharness-rsi-supplement
python3 scripts/verify_mcnemar.py --archive /path/to/videoharness-rsi-supplement
```

Not in this git tree (on purpose):

- Per-question dumps and `val_contexts.jsonl` (supplement)
- Full 10-step trajectory sources beyond the endpoints above (supplement `search/`)
- Videos, weights, API keys, pixels

## Quick start (no GPU)

```bash
git clone https://github.com/X-G-Y/vl-harness-rsi.git
cd vl-harness-rsi
python -m venv .venv && source .venv/bin/activate
pip install -e ./vl_harness   # or: uv sync --project vl_harness

bash scripts/run_smoke.sh
```

`run_smoke.sh` runs an offline pipeline on `mock_niah` with `StubVLM`.

## Reproduce paper scores

See [`REPRODUCE.md`](REPRODUCE.md). You need:

1. The 83 LVBench videos in `manifests/lvbench_videos.json` (download yourself).
2. Frozen VLM at `http://127.0.0.1:8080/v1`.
3. CLIP at `http://127.0.0.1:8181` for AKS / Ledger / AKS-90.

You do not need a proposer to re-score the frozen harnesses.

## Evolution loop

```bash
source setenv.sh
cd vl_harness
PYTHONPATH=.. python -m vl_harness.meta_harness --config config_k40.yaml
```

Requires a coding-agent CLI or LiteLLM proposer. The outer loop is derived from
[Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) (MIT; see
`NOTICE`).

## Layout

```text
vl-harness-rsi/
├── README.md / DATASETS.md / REPRODUCE.md / NOTICE / LICENSE
├── setenv.example.sh
├── manifests/          # 83 video IDs + 1232-QA split (CC-BY-NC-SA-4.0)
├── scripts/            # smoke + dump verifiers (verifiers need the supplement)
├── paper/              # frozen scores, McNemar, bootstrap, search tables
└── vl_harness/         # package + agents/
```

## License

- Code (original): Apache-2.0
- Meta-Harness-derived loop files: MIT; copyright in `NOTICE`
- `manifests/` (LVBench-derived): CC-BY-NC-SA-4.0, not Apache-2.0
- Benchmarks and weights: not shipped; terms in `DATASETS.md`

Commercial use of the code does not grant commercial use of LVBench,
Video-MME, or MLVU.

Distinct from Homer's "LV-Harness" (prompt-level skills, not code-level RSI).
