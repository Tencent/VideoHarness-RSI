"""Stated-interval frame addressing spliced onto the AKS selector.

Parent: ``agents/aks.py`` (frontier). The parent chooses every shown frame by
CLIP question-image similarity over one question-independent ingest pool. When
a question names *where* to look ("what happens from 37:30-38:05?"), that
address is a far more precise localiser than any similarity score -- and it is
known before a frame is decoded. The parent cannot use it: its memory holds a
fixed pool and no video handle, so an instant between two pool points is
unreachable no matter how the ranker behaves.

Mechanism (axis D -- retrieval modality router; the route is chosen by a
property of the question text rather than by an embedding score):

1. Parse elapsed-time references out of the question. A range bounds the
   interval; a single point becomes a small symmetric window.
2. Guard: no reference, or a reference past the video duration (an on-screen
   wall clock rather than an offset), routes to the parent unchanged -- same
   pool, same scorer, same allocation, same frame count.
3. Otherwise split the answer budget in two tiers that share the parent's
   per-request cap:
   - a *stated-interval* tier, decoded fresh from the video inside the named
     interval at native spacing, which reaches instants the pool skipped;
   - a *skeleton* tier, the parent's own AKS selection, kept so questions whose
     evidence is broader than the stated span keep the global spread the parent
     already answers correctly.
   Frames are merged, de-duplicated by frame index, and shown chronologically
   with timestamps so the model can bind the named time to what it sees.

The interval tier is additive within the cap rather than a replacement: a
prior candidate that let a similarity score aim the same re-decode regressed,
because it evicted global coverage and aimed at the wrong minute. Here the
aiming signal is the question itself and the skeleton always survives.
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
from .aks import ALL_DEPTH, aks_select

AGENT_NAME = "stated_interval_router"

# Elapsed-time reference: MM:SS or HH:MM:SS, not embedded in a longer number.
_TIMECODE = re.compile(r"(?<!\d)(\d{1,2}):([0-5]\d)(?::([0-5]\d))?(?!\d)")

# A bare point reference is widened to this half-width before decoding, so a
# single named instant still yields a short observable stretch of video.
POINT_PAD_S = 6.0
# Interval tier share of the per-request budget. The remainder always goes to
# the whole-video skeleton, so global coverage cannot be evicted entirely.
INTERVAL_SHARE = 0.5
# A stated span wider than this is already coarse enough for the parent's grid;
# treat it as a hint and let the skeleton carry most of the budget.
WIDE_SPAN_S = 300.0

PAPER_INSTRUCT_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    "Question: {question}\n{options}"
)


def parse_stated_interval(
    question: str, duration: float | None
) -> tuple[float, float] | None:
    """Return the elapsed-time interval the question names, or None.

    Two or more references bound the interval directly; one becomes a symmetric
    window. A reference beyond the video's duration is rejected -- that is a
    clock shown *inside* the scene, not an offset into the file.
    """
    seconds: list[float] = []
    for m in _TIMECODE.finditer(question):
        h_or_m, mins, secs = m.groups()
        if secs is None:
            total = int(h_or_m) * 60 + int(mins)
        else:
            total = int(h_or_m) * 3600 + int(mins) * 60 + int(secs)
        seconds.append(float(total))
    if not seconds:
        return None
    lo, hi = min(seconds), max(seconds)
    if duration and duration > 0:
        # Reject references that cannot be positions in this video.
        if lo > duration:
            return None
        hi = min(hi, duration)
    if hi - lo < 1e-6:
        lo, hi = lo - POINT_PAD_S, lo + POINT_PAD_S
    lo = max(0.0, lo)
    if duration and duration > 0:
        hi = min(hi, duration)
    if hi <= lo:
        return None
    return lo, hi


@dataclass
class _Mem:
    frames: list[Frame]
    img_emb: Any
    duration: float | None
    video: VideoStream | None


class StatedIntervalRouter(VideoMemoryHarness):
    """AKS selection, plus a question-addressed interval tier when one exists."""

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
        # The stream is retained so answer time can decode inside an interval the
        # pool never covered. `_close_memory` releases `memory.video`'s reader on
        # eviction and the reader re-opens lazily, so this holds no fd hostage.
        return _Mem(
            frames=frames,
            img_emb=img_emb,
            duration=video.duration,
            video=video,
        )

    def _interval_frames(
        self, memory: _Mem, lo: float, hi: float, want: int
    ) -> list[Frame]:
        """Decode up to ``want`` frames inside [lo, hi], finer than the pool."""
        if want <= 0 or memory.video is None:
            return []
        try:
            return memory.video.sample_time_range(lo, hi, want)
        except Exception:
            # Decode is best-effort; the skeleton tier still answers.
            return []

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

        span = parse_stated_interval(question, memory.duration)

        if span is None:
            # No temporal address: the parent's exact path.
            chosen = [memory.frames[i] for i in aks_select(scores, k)]
            chosen = self.take_answer_frames(chosen, k)
            parts = self.render_frames(chosen)
            route = "parent"
            n_interval = 0
        else:
            lo, hi = span
            # A wide stated span is already addressable by the parent's grid, so
            # it only earns a small share; a narrow one earns the full share.
            share = INTERVAL_SHARE if (hi - lo) <= WIDE_SPAN_S else 0.25
            want_interval = max(1, int(round(k * share)))
            interval = self._interval_frames(memory, lo, hi, want_interval)

            # Skeleton keeps whole-video coverage using the parent's selector.
            skeleton_k = max(1, k - len(interval))
            skeleton = [memory.frames[i] for i in aks_select(scores, skeleton_k)]

            merged: dict[int, Frame] = {}
            for fr in interval:
                merged[int(fr.index)] = fr
            n_interval = len(merged)
            for fr in skeleton:
                merged.setdefault(int(fr.index), fr)

            chosen = [merged[i] for i in sorted(merged)]
            chosen = self.take_answer_frames(chosen, k)
            # Timestamps let the model bind the named time to the frames shown.
            parts = self.render_frames(chosen, timestamps=True)
            route = "interval"

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
        return letter, {
            "strategy": AGENT_NAME,
            "sampled": len(chosen),
            "pool": n,
            "budget": k,
            "route": route,
            "stated_interval": (
                [round(span[0], 1), round(span[1], 1)] if span else None
            ),
            "interval_frames": n_interval,
            "n_segments_est": n_segments_est,
            "score_span": float(np.max(scores) - np.min(scores)) if n else 0.0,
            "raw": (resp or "")[:200],
        }
