VL-Harness-RSI
Copyright 2026 VL-Harness-RSI contributors

This product is licensed under the Apache License, Version 2.0 (see LICENSE),
except where a file or directory states otherwise.

Apache-2.0 covers original source in this repository. It does not relicense
third-party code, benchmark annotations, videos, or model weights. Dataset
terms are in DATASETS.md. Files under manifests/ are CC-BY-NC-SA-4.0
derivatives of LVBench, not Apache-2.0.

==============================================================================
Meta-Harness
------------------------------------------------------------------------------
Portions of the evolution loop are derived from Meta-Harness
(https://github.com/stanford-iris-lab/meta-harness), including the structure
of claude_wrapper.py, inner_loop.py, meta_harness.py, litellm_proposer.py,
and the proposer SKILL layout under vl_harness/.claude/skills/.

MIT License

Copyright (c) 2026 Yoonho Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

==============================================================================
AKS (Adaptive Keyframe Sampling)
------------------------------------------------------------------------------
vl_harness/agents/aks.py is a protocol-matched reimplementation of the selector
described in Tang et al., CVPR 2025, Adaptive Keyframe Sampling for Long Video
Understanding, with the official repository https://github.com/ncTimTang/AKS
as the algorithm reference (frame_select.py / feature_extract.py).

As of 2026-08-18 that repository publishes no SPDX license (GitHub
license:null). This file does not grant rights in the official AKS code or
models. Use of the official AKS repository remains subject to its authors.

==============================================================================
Datasets
------------------------------------------------------------------------------
This repository does not redistribute raw videos or model weights.

manifests/lvbench_videos.json and manifests/lvbench_split_seed42.json are
derived from THUDM/LVBench (CC-BY-NC-SA-4.0). They are licensed under
CC-BY-NC-SA-4.0, not Apache-2.0. Academic / non-commercial use only;
ShareAlike applies to adaptations.

Downloaders for Video-MME, MLVU, and MVBench fetch data from the official
hosts into $VL_HARNESS_DATA. Those copies stay local and must not be committed.
See DATASETS.md for per-benchmark terms.

==============================================================================
Models used by the paper protocol (not shipped)
------------------------------------------------------------------------------
Qwen3-VL-8B-Instruct and clip-ViT-B-32 are downloaded by the user and remain
under their own licenses. This repository does not redistribute weights.
