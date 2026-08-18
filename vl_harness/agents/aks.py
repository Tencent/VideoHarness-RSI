"""Matched-budget AKS baseline (Tang et al., CVPR 2025).

Official code: https://github.com/ncTimTang/AKS
Paper: *Adaptive Keyframe Sampling for Long Video Understanding*.
Algorithm: ``frame_select.py`` (recursive mean/std split) + CLIP/BLIP
relevance scores from ``feature_extract.py``.

Pipeline, matching the repo:

1. Score every ingest-pool frame by text–image similarity to the question.
2. Min-max normalize scores. Recursively bisect a segment unless its top-K
   mean exceeds the segment mean by ``t1`` (peak found) or depth hits
   ``all_depth``.
3. Allocate ``K / 2**depth`` frames to each surviving segment, take the
   highest-scoring indices in that segment, then sort temporally.

This is a plug-and-play *selector* around the frozen VLM: one answer-time
request of exactly ``frame_budget()`` frames. It is the published harness
that fits the Harness-VL protocol (frozen backbone, matched K). It is **not**
a LongVideoBench/Video-MME paper-score reproduction.

Component mapping:

| Official                                      | Here                                      |
|-----------------------------------------------|-------------------------------------------|
| BLIP-ITM (main) / CLIP-B/32 cosine (alt)      | harness CLIP ``embed_images`` × ``embed_texts`` (clip-ViT-B-32) |
| 1 fps over the whole video                    | framework ingest pool (2 fps, cap 320)    |
| ``meanstd`` + ``t1=0.8``, ``t2=-100``, depth 5 | same defaults                             |
| ``max_num_frames`` (paper eval often 64)      | ``frame_budget()``                        |
| LLaVA-Video / Qwen2-VL answerer               | frozen Qwen3-VL-8B-Instruct               |

Declared deviations (none flatter this baseline):

1. Relevance scorer is CLIP-B/32, an official extractor in ``feature_extract.py``,
   not the BLIP-ITM model used in the paper's main tables.
2. Candidate pool is 320 @ 2 fps, not 1 fps over hour-long videos.
3. After the official integer allocation (which often undershoots K), unused
   frames are filled by remaining CLIP score so the arm actually spends K.
   Empty half-segments are not recursed (official ``meanstd`` can emit empty
   lists and crash ``np.mean``).
4. Answer template is the Qwen3-VL instruct protocol, so Δ vs instruct-uniform
   is selection, not prompt.

License: the official AKS repository published no SPDX license as of 2026-08-18
(GitHub ``license:null``). This file is a protocol-matched reimplementation for
the frozen-VLM / matched-K setting. It does not relicense official AKS code or
models. See the repository ``NOTICE``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream

AGENT_NAME = "aks"
T1 = 0.8
T2 = -100.0
ALL_DEPTH = 5
PAPER_INSTRUCT_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    "Question: {question}\n{options}"
)


def _as_list(score) -> list[float]:
    return [float(x) for x in np.asarray(score).tolist()]


def meanstd(
    dic_scores: list[dict[str, Any]],
    n: int,
    fns: list[list[int]],
    t1: float,
    t2: float,
    all_depth: int,
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Official recursive bisect. Skip empty / singleton splits (crash guard)."""
    split_scores: list[dict[str, Any]] = []
    split_fn: list[list[int]] = []
    no_split_scores: list[dict[str, Any]] = []
    no_split_fn: list[list[int]] = []
    for dic_score, fn in zip(dic_scores, fns):
        score = _as_list(dic_score["score"])
        depth = int(dic_score["depth"])
        if not score or not fn:
            continue
        mean = float(np.mean(score))
        std = float(np.std(score))
        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = float(np.mean(top_score)) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth and len(score) >= 2:
            mid = len(score) // 2
            split_scores.append({"score": score[:mid], "depth": depth + 1})
            split_fn.append(fn[:mid])
            split_scores.append({"score": score[mid:], "depth": depth + 1})
            split_fn.append(fn[mid:])
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if split_scores:
        rec_s, rec_f = meanstd(split_scores, n, split_fn, t1, t2, all_depth)
    else:
        rec_s, rec_f = [], []
    return no_split_scores + rec_s, no_split_fn + rec_f


def aks_select(
    scores: np.ndarray,
    k: int,
    t1: float = T1,
    t2: float = T2,
    all_depth: int = ALL_DEPTH,
) -> list[int]:
    """Return ``k`` pool indices in temporal order (official AKS + fill-to-K)."""
    raw = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(raw.size)
    if n == 0:
        return []
    k = min(int(k), n)
    if n <= k:
        return list(range(n))
    span = float(np.max(raw) - np.min(raw))
    if span <= 0:
        normalized = np.zeros(n, dtype=np.float64)
    else:
        normalized = (raw - np.min(raw)) / span
    fn = list(range(n))
    segs, seg_fns = meanstd(
        [{"score": _as_list(normalized), "depth": 0}],
        k,
        [fn],
        t1,
        t2,
        all_depth,
    )
    chosen: list[int] = []
    for seg, idxs in zip(segs, seg_fns):
        seg_score = _as_list(seg["score"])
        f_num = int(k / 2 ** int(seg["depth"]))
        if f_num <= 0 or not idxs:
            continue
        topk = heapq.nlargest(f_num, range(len(seg_score)), seg_score.__getitem__)
        chosen.extend(int(idxs[t]) for t in topk if 0 <= t < len(idxs))
    uniq: list[int] = []
    seen: set[int] = set()
    for i in chosen:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            uniq.append(i)
    if len(uniq) > k:
        uniq = sorted(uniq, key=lambda i: float(raw[i]), reverse=True)[:k]
    elif len(uniq) < k:
        rest = np.argsort(-raw)
        for i in rest:
            ii = int(i)
            if ii in seen:
                continue
            seen.add(ii)
            uniq.append(ii)
            if len(uniq) >= k:
                break
    return sorted(uniq)[:k]


@dataclass
class _Mem:
    frames: list[Frame]
    img_emb: Any
    duration: float | None


class AdaptiveKeyframeSampling(VideoMemoryHarness):
    """CLIP-scored AKS selector; answer-time spend is exactly K."""

    FPS = 2.0
    MAX_INGEST_FRAMES = 320

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_INGEST_FRAMES)
        images = [
            f.image if f.image is not None else (f.caption or "") for f in frames
        ]
        img_emb = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return _Mem(frames=frames, img_emb=img_emb, duration=video.duration)

    def answer_question(
        self, memory: _Mem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        k = self.frame_budget()
        n = len(memory.frames)
        if n == 0:
            return "?", {"error": "no frames", "strategy": AGENT_NAME}

        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        mat = np.asarray(memory.img_emb, dtype=np.float32)
        scores = mat @ qv.reshape(-1)
        idxs = aks_select(scores, k)
        chosen = [memory.frames[i] for i in idxs]
        chosen = self.take_answer_frames(chosen, k)

        parts = self.render_frames(chosen)
        parts.append(
            {
                "type": "text",
                "text": PAPER_INSTRUCT_PROMPT.format(
                    question=question, options=format_options(options)
                ),
            }
        )
        resp = self.ask_vlm(parts)
        letter = normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        )
        n_segments_est = int(
            np.clip(np.round(np.log2(max(n / max(k, 1), 1))), 0, ALL_DEPTH)
        )
        score_span = float(np.max(scores) - np.min(scores)) if n else 0.0
        return letter, {
            "strategy": AGENT_NAME,
            "sampled": len(chosen),
            "pool": n,
            "budget": k,
            "n_segments_est": n_segments_est,
            "score_span": score_span,
            "raw": (resp or "")[:200],
        }
