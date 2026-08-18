"""Candidate: hybrid VLM-navigation + embedding retrieval for frame selection.

Hypothesis: The champion uses ONLY VLM text navigation to select verification
frames, while prior embedding-based systems (hybrid_temporal_retrieve_iter1)
used ONLY vector similarity. Each has complementary strengths: VLM navigation
excels at reasoning-heavy questions ("what happens after X", "why does Y")
because it can follow causal chains in text, while embedding retrieval excels
at appearance questions ("what color", "who is", "where is") because
text-to-image similarity directly matches visual concepts. By COMBINING both
signals — using VLM navigation to identify temporal ranges AND embedding
similarity to prioritize frames within and around those ranges — we cover both
question types with a single mechanism.

Key mechanism: After VLM navigation identifies ranges, we compute embedding
similarity (question -> frame image embeddings) for ALL frames, then use a
WEIGHTED score that combines (a) being inside a VLM-navigated range and
(b) embedding similarity. This selects the 40 most relevant frames using both
reasoning-based and similarity-based signals simultaneously.

Axes: E (hybrid retrieval combining VLM navigation with embedding kNN) +
D (implicit routing — the weighting between the two signals adapts to the
question: reasoning questions benefit more from navigation, appearance questions
from embeddings).
"""

import json
import re
from typing import Any

import numpy as np

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream


NAVIGATE_PROMPT = (
    "You are analyzing a long video ({dur:.0f}s, {n} sampled moments) to "
    "answer a question. Below is a timestamped timeline of descriptions:\n\n"
    "{timeline}\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Identify 1-4 time ranges (as start_index-end_index from the list above) "
    "that contain the information needed to answer. Also give your tentative "
    "answer based on the descriptions.\n\n"
    '{{"ranges": [[start_idx, end_idx], ...], '
    '"reasoning": "...", "tentative_answer": "<letter>"}}'
)

ANSWER_PROMPT = (
    "You are answering a question about a long video. The {k} frames below "
    "were selected using both text-based reasoning and visual similarity to "
    "the question.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Please select the best answer from the options above and directly "
    "provide the letter representing your choice without giving any "
    "explanation."
)


class EmbedNavigateHybridIter2(VideoMemoryHarness):
    """Hybrid frame selection: VLM navigation + embedding similarity."""

    FPS = 2.0
    MAX_INGEST_FRAMES = 320
    MAX_TIMELINE_CAPTIONS = 160
    NAV_WEIGHT = 0.6
    EMBED_WEIGHT = 0.4

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_INGEST_FRAMES)
        captions = self.caption_frames(frames)
        images = [f.image if f.image is not None else (f.caption or "") for f in frames]
        img_emb = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return {
            "frames": frames,
            "captions": captions,
            "img_emb": img_emb,
            "duration": video.duration,
        }

    def _build_timeline(self, frames: list[Frame], captions: list[str]) -> str:
        n = len(frames)
        if n <= self.MAX_TIMELINE_CAPTIONS:
            return "\n".join(
                f"[{i}] {frames[i].timestamp:.1f}s: {captions[i]}"
                for i in range(n)
            )
        step = n / self.MAX_TIMELINE_CAPTIONS
        selected = [int(i * step) for i in range(self.MAX_TIMELINE_CAPTIONS)]
        return "\n".join(
            f"[{idx}] {frames[idx].timestamp:.1f}s: {captions[idx]}"
            for idx in selected
        )

    def _parse_ranges(self, resp: str, n: int) -> list[tuple[int, int]]:
        raw = extract_json_field(resp, "ranges")
        if raw:
            try:
                ranges = (
                    json.loads(raw) if raw.startswith("[") else json.loads(f"[{raw}]")
                )
                result = []
                for r in ranges:
                    if isinstance(r, list) and len(r) == 2:
                        s, e = int(r[0]), int(r[1])
                        s = max(0, min(s, n - 1))
                        e = max(s, min(e, n - 1))
                        result.append((s, e))
                if result:
                    return result
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        nums = re.findall(r"\[(\d+)\s*[-,]\s*(\d+)\]", resp)
        if nums:
            result = []
            for s, e in nums[:4]:
                s, e = int(s), int(e)
                s = max(0, min(s, n - 1))
                e = max(s, min(e, n - 1))
                result.append((s, e))
            return result
        single_nums = re.findall(r"\[(\d+)\]", resp)
        if single_nums:
            return [
                (max(0, int(x) - 2), min(n - 1, int(x) + 2))
                for x in single_nums[:4]
            ]
        return []

    def _compute_hybrid_scores(
        self,
        n_frames: int,
        ranges: list[tuple[int, int]],
        question: str,
        img_emb: Any,
    ) -> np.ndarray:
        """Combine navigation-based and embedding-based relevance scores."""
        nav_scores = np.zeros(n_frames)
        if ranges:
            for s, e in ranges:
                nav_scores[s : e + 1] = 1.0
            margin = 5
            for s, e in ranges:
                for offset in range(1, margin + 1):
                    decay = 1.0 - offset / (margin + 1)
                    if s - offset >= 0:
                        nav_scores[s - offset] = max(nav_scores[s - offset], decay * 0.5)
                    if e + offset < n_frames:
                        nav_scores[e + offset] = max(nav_scores[e + offset], decay * 0.5)
        else:
            nav_scores[:] = 1.0 / n_frames

        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        embed_sims = np.asarray(img_emb, dtype=np.float32) @ qv.reshape(-1)
        embed_min = embed_sims.min()
        embed_max = embed_sims.max()
        if embed_max - embed_min > 1e-6:
            embed_scores = (embed_sims - embed_min) / (embed_max - embed_min)
        else:
            embed_scores = np.ones(n_frames) * 0.5

        hybrid = self.NAV_WEIGHT * nav_scores + self.EMBED_WEIGHT * embed_scores
        return hybrid

    def _select_frames_hybrid(
        self,
        frames: list[Frame],
        scores: np.ndarray,
        budget: int,
    ) -> list[Frame]:
        """Select top-budget frames by hybrid score, maintaining temporal order."""
        top_indices = list(np.argsort(-scores)[:budget])
        top_indices.sort()
        return [frames[i] for i in top_indices]

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        frames: list[Frame] = memory["frames"]
        captions: list[str] = memory["captions"]
        img_emb = memory["img_emb"]
        duration: float = memory["duration"]
        budget = self.frame_budget()

        timeline = self._build_timeline(frames, captions)

        nav_parts = [
            {
                "type": "text",
                "text": NAVIGATE_PROMPT.format(
                    dur=duration,
                    n=len(frames),
                    timeline=timeline,
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        nav_resp = self.ask_vlm(nav_parts)

        ranges = self._parse_ranges(nav_resp, len(frames))
        tentative = extract_json_field(nav_resp, "tentative_answer") or "?"

        hybrid_scores = self._compute_hybrid_scores(
            len(frames), ranges, question, img_emb
        )
        selected_frames = self._select_frames_hybrid(frames, hybrid_scores, budget)

        answer_parts = [
            {
                "type": "text",
                "text": ANSWER_PROMPT.format(
                    k=len(selected_frames),
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        answer_parts += self.render_frames(selected_frames)
        answer_resp = self.ask_vlm(answer_parts)
        final = extract_json_field(answer_resp, "final_answer") or answer_resp
        letter = normalize_choice(final, options)

        if letter == "?":
            letter = normalize_choice(tentative, options)

        return letter, {
            "tentative": tentative,
            "ranges": ranges,
            "num_selected_frames": len(selected_frames),
            "raw": answer_resp[:200],
        }
