# Datasets

Set `VL_HARNESS_DATA` (see `setenv.example.sh`). Layout:

```text
$VL_HARNESS_DATA/
  lvbench/
  video_mme/
  mvbench/
  mlvu/
```

Apache-2.0 on this repository covers **code only**. Benchmarks keep their own
terms. This repo does **not** redistribute videos, file checksums, download-failure
logs, or model weights. Downloaded copies belong under `$VL_HARNESS_DATA` and
must stay out of git.

## Terms (read before downloading)

| Benchmark | Host | License / terms | What this repo ships | What you fetch locally |
|---|---|---|---|---|
| LVBench | [THUDM/LVBench](https://huggingface.co/datasets/THUDM/LVBench) | **CC-BY-NC-SA-4.0**. Academic / non-commercial only. ShareAlike on adaptations. LVBench does not own the raw videos. | `manifests/` (ids + 1,232-QA split, also CC-BY-NC-SA-4.0). See `manifests/README.md`. | `video_info.meta.jsonl` + mp4s |
| Video-MME | [lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) | Academic research only. Commercial use prohibited. **No distribute / publish / copy / modify** without prior approval. Video copyright stays with owners. | Downloader only | annotations + videos (w/o subtitles) |
| MLVU | [MLVU/MVLU](https://huggingface.co/datasets/MLVU/MVLU) | **CC-BY-NC-SA-4.0**. Research only; gated HF terms. MLVU does not own raw-video copyright. | Downloader only | annotations + videos |
| MVBench | [OpenGVLab/MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) | MIT on the HF dataset; gated access may still apply. 320 NTU RGB+D videos are **not** redistributed by MVBench. | Downloader only | annotations + videos (NTU skipped until you add them) |

Qwen3-VL and CLIP weights are also fetched by you and remain under their own licenses.

Reported LVBench numbers use a **1,232-QA subset**, not the full benchmark:

- 83 accessible video IDs and 20 inaccessible IDs: `manifests/lvbench_videos.json`
- QA index + val 350 / held-out 882: `manifests/lvbench_split_seed42.json`
- Recorded against a ModelScope snapshot dated 2026-08-08

The loader only builds an episode if the local mp4 exists. A different local video set yields a different pool and incomparable scores.

## Download helpers

From the repo root:

```bash
source setenv.sh
python scripts/download_lvbench.py
python scripts/download_video_mme.py
python scripts/download_mvbench.py
# MLVU may need HF_TOKEN for gated assets:
# HF_TOKEN=hf_xxx python scripts/download_mlvu.py
```

**LVBench videos:** prefer ModelScope (`AI-ModelScope/LVBench`), which is the
default `--source modelscope`. That snapshot is what the 83-id manifest was
recorded against (~83 of 103 videos; the rest are copyright-removed).

```bash
python scripts/download_lvbench.py --download-videos --max-videos 0
```

`--source youtube` uses yt-dlp. YouTube’s terms of service generally forbid
downloading; use it only if you have the right to fetch those videos. Prefer
matching the 83 IDs in the manifest.

Offline plumbing without videos:

```bash
PYTHONPATH=. python -m vl_harness.inner_loop \
  --memory vl_harness/agents/pilot_uniform_k.py \
  --dataset mock_niah --model stub --mode offline \
  --num-train 10 --num-val 40 --num-test 40 \
  --val-output /tmp/vlh/val.json --log /tmp/vlh/log.jsonl
```

Do not commit raw videos or evaluation logs into this repository.
