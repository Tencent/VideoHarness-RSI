"""Candidate: AKS with timestamp annotations on rendered frames (exploitation).

Hypothesis
----------
The champion (aks_anchor_bypass_iter2) shows K=40 AKS-selected frames with NO
temporal context in the unanchored branch. The VLM cannot distinguish frame
ordering or locate moments within the sequence. Iter3's temporal_context_answer
(-6pp) failed because it changed BOTH presentation (timestamps) AND answering
(reasoning-eliciting prompt). The 8B model is sensitive to prompt verbosity.

Here we add ONLY per-frame timestamp labels ("[45.0s]") via the built-in
render_frames(timestamps=True) mechanism—exactly what the ANCHOR branch already
uses successfully (65% vs 48% on anchored questions). The prompt remains the
proven terse "respond with only the letter" template. This isolates the effect
of temporal context from prompt confusion.

Axes: F (packing — frame ordering/annotation within a fixed selection).

Uses ``aks_select`` from ``aks.py``. Same upstream-license caveat as that file
(official AKS repo has no SPDX license). See repository ``NOTICE``.
"""

from __future__ import annotations

import re
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
from ..vlm import frame_token_cost
from .aks import aks_select

AGENT_NAME = "timestamped_aks_iter5"

FPS = 2.0
MAX_INGEST_FRAMES = 320
ANCHOR_PAD_S = 6.0
ANCHOR_SHARE = 0.75

_HMS = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")
_MS = re.compile(r"\b(\d{1,2}):(\d{2})\b")

ANCHOR_PROMPT = (
    "Select the best answer to the following multiple-choice question based on "
    "the video.\n"
    "The question refers to the moment around {desc}. You are shown dense frames "
    "from that moment at full resolution, plus a few frames from the rest of the "
    "video for context.\n"
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    "Question: {question}\n{options}"
)
AKS_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    "Question: {question}\n{options}"
)


def _clock_times(text: str) -> list[float]:
    out: list[float] = []
    spans: list[tuple[int, int]] = []
    for m in _HMS.finditer(text):
        h, mi, s = (int(g) for g in m.groups())
        out.append(float(h * 3600 + mi * 60 + s))
        spans.append(m.span())
    for m in _MS.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        mi, s = (int(g) for g in m.groups())
        if s < 60:
            out.append(float(mi * 60 + s))
    return out


def _anchor_window(question: str, duration: float | None) -> tuple[float, float] | None:
    ts = _clock_times(question)
    if duration and duration > 0:
        ts = [t for t in ts if t <= duration + 1.0]
    if not ts:
        return None
    lo, hi = min(ts), max(ts)
    return (max(0.0, lo - ANCHOR_PAD_S), hi + ANCHOR_PAD_S)


@dataclass
class _Mem:
    video: VideoStream
    frames: list[Frame]
    img_emb: Any
    duration: float | None


class TimestampedAks(VideoMemoryHarness):
    """AKS with per-frame timestamp annotations in the unanchored branch."""

    FPS = FPS
    MAX_INGEST_FRAMES = MAX_INGEST_FRAMES

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_INGEST_FRAMES)
        images = [
            f.image if f.image is not None else (f.caption or "") for f in frames
        ]
        img_emb = self.embed_images(images)
        vtok = sum(frame_token_cost(w, h) for w, h in (f.size for f in frames))
        self.account_ingest(video.video_id, len(frames), vtok)
        return _Mem(video=video, frames=frames, img_emb=img_emb, duration=video.duration)

    def _anchored_answer(self, memory: _Mem, question: str, options: list[str],
                         window: tuple[float, float], budget: int):
        t0, t1 = window
        n_dense = max(1, int(round(budget * ANCHOR_SHARE)))
        n_ctx = max(1, budget - n_dense)

        dense = memory.video.sample_time_range(t0, t1, n_dense)
        if not dense:
            dense = sorted(
                memory.frames, key=lambda f: abs(f.timestamp - (t0 + t1) / 2)
            )[:n_dense]

        if n_ctx and memory.frames:
            step = max(1, len(memory.frames) // n_ctx)
            ctx = [memory.frames[i] for i in range(0, len(memory.frames), step)][:n_ctx]
            ctx = [f for f in ctx if not (t0 <= f.timestamp <= t1)]
        else:
            ctx = []

        all_frames = sorted(
            {id(f): f for f in dense + ctx}.values(),
            key=lambda f: f.timestamp,
        )
        all_frames = self.take_answer_frames(all_frames, budget)

        parts: list[dict[str, Any]] = []
        for fr in all_frames:
            parts.append({"type": "text", "text": f"[{fr.timestamp:.0f}s]"})
            parts.extend(self.render_frames([fr]))

        desc = f"{int(t0 + ANCHOR_PAD_S) // 60:02d}:{int(t0 + ANCHOR_PAD_S) % 60:02d}"
        if t1 - t0 > 2 * ANCHOR_PAD_S + 1:
            desc += f"-{int(t1 - ANCHOR_PAD_S) // 60:02d}:{int(t1 - ANCHOR_PAD_S) % 60:02d}"

        parts.append({"type": "text", "text": ANCHOR_PROMPT.format(
            desc=desc, question=question, options=format_options(options))})
        return parts, {"branch": "anchor", "window": [t0, t1], "dense": len(dense)}

    def _aks_answer(self, memory: _Mem, question: str, options: list[str], budget: int):
        n = len(memory.frames)
        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        mat = np.asarray(memory.img_emb, dtype=np.float32)
        scores = mat @ qv.reshape(-1)
        idxs = aks_select(scores, budget)
        chosen = [memory.frames[i] for i in idxs]
        chosen = self.take_answer_frames(chosen, budget)

        # Key difference: render with timestamps=True
        parts = self.render_frames(chosen, timestamps=True)
        parts.append({"type": "text", "text": AKS_PROMPT.format(
            question=question, options=format_options(options))})
        return parts, {"branch": "aks_ts", "pool": n, "selected": len(chosen)}

    def answer_question(
        self, memory: _Mem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        if not memory.frames:
            return "?", {"error": "no frames", "strategy": AGENT_NAME}
        budget = self.frame_budget()
        window = _anchor_window(question, memory.duration)
        if window:
            parts, meta = self._anchored_answer(memory, question, options, window, budget)
        else:
            parts, meta = self._aks_answer(memory, question, options, budget)
        resp = self.ask_vlm(parts)
        letter = normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        )
        meta.update({"strategy": AGENT_NAME, "raw": (resp or "")[:200]})
        return letter, meta
