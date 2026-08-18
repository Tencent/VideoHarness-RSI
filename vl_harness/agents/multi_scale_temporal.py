"""Candidate: multi-scale temporal framing (overview + focus).

Hypothesis: Long videos have multi-scale structure that a single uniform sampling
cannot capture. Homer-style hierarchical perceptual memory suggests the VLM
benefits from BOTH global context (what happens across the entire video) AND
local detail (fine-grained frames from relevant moments). By showing two scales
simultaneously — 8 sparse overview frames for global temporal orientation plus
10-12 densely-sampled frames from the most relevant segment identified via CLIP —
the VLM can locate the answer temporally (overview) and verify it visually
(focus). This is fundamentally different from temporal_grounding (which requires
explicit timestamps) because embedding-based focus works for ALL question types.

Axes: A (two-scale ingestion: sparse overview + dense pool) + C (hierarchical
granularity: global overview level + local focus level) + F (budget packing:
~20 frames split across two scales instead of 32 uniform).
"""

from typing import Any

import numpy as np

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream


PROMPT = (
    "You are answering a multiple-choice question about a long video "
    "({dur:.0f}s duration).\n\n"
    "You are shown frames at two scales:\n"
    "1. OVERVIEW: {n_overview} frames evenly spaced across the entire video "
    "(for temporal context)\n"
    "2. FOCUS: {n_focus} frames from the segment most relevant to the question "
    "({focus_start:.0f}s–{focus_end:.0f}s)\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Use the overview for global context and the focus frames for detail. "
    "Think step by step, then end with: ANSWER: <letter>\n"
)

TAIL = "\n\nAfter your reasoning, write your final answer as: ANSWER: <letter>"


class MultiScaleTemporal(VideoMemoryHarness):
    """Two-scale viewing: sparse global overview + dense local focus."""

    NUM_OVERVIEW = 8
    NUM_POOL = 340
    FOCUS_FRAMES = 12
    FOCUS_WINDOW_RATIO = 0.15

    def build_memory(self, video: VideoStream) -> Any:
        all_frames = video.sample_uniform(self.NUM_POOL)
        images = [f.image if f.image is not None else f.caption for f in all_frames]
        img_emb = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in all_frames
        )
        self.account_ingest(video.video_id, len(all_frames), vtok)

        overview_step = max(1, len(all_frames) // self.NUM_OVERVIEW)
        overview_idxs = list(range(0, len(all_frames), overview_step))[:self.NUM_OVERVIEW]
        overview = [all_frames[i] for i in overview_idxs]

        return {
            "all_frames": all_frames,
            "img_emb": img_emb,
            "overview": overview,
            "overview_idxs": set(overview_idxs),
            "duration": video.duration,
        }

    def _find_focus_segment(self, query_vec, frames: list[Frame], img_emb) -> list[int]:
        """Find the best contiguous temporal segment using a sliding relevance window."""
        n = len(frames)
        if n <= self.FOCUS_FRAMES:
            return list(range(n))

        sims = np.asarray(img_emb) @ np.asarray(query_vec).reshape(-1)

        window_size = max(self.FOCUS_FRAMES, int(n * self.FOCUS_WINDOW_RATIO))
        window_size = min(window_size, n)

        best_score = -float("inf")
        best_start = 0
        current_sum = float(np.sum(sims[:window_size]))

        if current_sum > best_score:
            best_score = current_sum
            best_start = 0

        for start in range(1, n - window_size + 1):
            current_sum += sims[start + window_size - 1] - sims[start - 1]
            if current_sum > best_score:
                best_score = current_sum
                best_start = start

        window_idxs = list(range(best_start, best_start + window_size))
        window_sims = sims[window_idxs]
        top_in_window = np.argsort(-window_sims)[:self.FOCUS_FRAMES]
        return sorted([window_idxs[i] for i in top_in_window])

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        all_frames: list[Frame] = memory["all_frames"]
        img_emb = memory["img_emb"]
        overview: list[Frame] = memory["overview"]
        overview_idxs: set = memory["overview_idxs"]
        duration: float = memory["duration"]

        qv = self.embed_texts([question])[0]

        focus_idxs = self._find_focus_segment(qv, all_frames, img_emb)
        focus_frames = [all_frames[i] for i in focus_idxs if i not in overview_idxs]
        if not focus_frames:
            focus_frames = [all_frames[i] for i in focus_idxs[:self.FOCUS_FRAMES]]

        focus_start = focus_frames[0].timestamp if focus_frames else 0.0
        focus_end = focus_frames[-1].timestamp if focus_frames else duration

        parts = [
            {
                "type": "text",
                "text": PROMPT.format(
                    dur=duration,
                    n_overview=len(overview),
                    n_focus=len(focus_frames),
                    focus_start=focus_start,
                    focus_end=focus_end,
                    question=question,
                    options=format_options(options),
                ),
            }
        ]

        parts.append({"type": "text", "text": "\n--- OVERVIEW (full video) ---"})
        parts += self.render_frames(overview)
        parts.append({"type": "text", "text": "\n--- FOCUS (relevant segment) ---"})
        parts += self.render_frames(focus_frames)
        parts.append({"type": "text", "text": TAIL})

        resp = self.ask_vlm(parts)

        import re
        m = re.search(r"ANSWER\s*:\s*\(?([A-Za-z])\)?", resp, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            allowed = {chr(ord("A") + i) for i in range(len(options))}
            if letter in allowed:
                return letter, {
                    "focus_range": (focus_start, focus_end),
                    "n_overview": len(overview),
                    "n_focus": len(focus_frames),
                    "raw": resp[:200],
                }
        letter = normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        )
        return letter, {
            "focus_range": (focus_start, focus_end),
            "n_overview": len(overview),
            "n_focus": len(focus_frames),
            "raw": resp[:200],
        }
