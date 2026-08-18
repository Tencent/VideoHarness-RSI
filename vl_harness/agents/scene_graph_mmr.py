"""Candidate: scene-level retrieval with MMR diversity selection.

Hypothesis: Frame-level embedding retrieval often returns near-duplicate frames
from a single moment (high redundancy, low coverage). By first segmenting the
video into scenes based on temporal gaps, retrieving at the SCENE level, then
using Maximal Marginal Relevance (MMR) to select diverse frames within retrieved
scenes, we get both relevance AND coverage of the video's temporal span. The VLM
sees frames from multiple relevant scenes instead of 6 near-identical frames.

Axes: C (scene-level granularity + temporal segmentation) + D (text→text scene
retrieval routing) + E (MMR diversity-aware retrieval within scenes).
"""

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


PROMPT = (
    "You answer a multiple-choice question about a long video ({dur:.0f}s). "
    "The video has been segmented into scenes. You are shown frames from the "
    "{n_scenes} most relevant scenes, selected for both relevance and diversity.\n\n"
    "Scene context:\n{scene_context}\n\n"
    "Question: {question}\n\nOptions:\n{options}\n"
)
INSTRUCTION = '\nRespond in JSON: {"reasoning": "...", "final_answer": "<letter>"}'


@dataclass
class _Scene:
    start_idx: int
    end_idx: int
    frames: list[Frame]
    captions: list[str]
    summary: str
    time_range: tuple[float, float]


@dataclass
class _SceneMem:
    scenes: list[_Scene]
    scene_embeddings: Any
    all_frames: list[Frame]
    all_captions: list[str]
    duration: float


class SceneGraphMMR(VideoMemoryHarness):
    """Scene-level retrieval + MMR frame diversity."""

    NUM_INGEST = 340
    MAX_SCENE_LEN = 8
    MIN_SCENE_LEN = 3
    TOP_SCENES = 3
    FRAMES_PER_SCENE = 3
    MMR_LAMBDA = 0.6

    def build_memory(self, video: VideoStream) -> Any:
        frames = video.sample_uniform(self.NUM_INGEST)
        captions = [self.caption_frame(f) for f in frames]
        timestamps = [f.timestamp for f in frames]

        scenes = self._segment_scenes(frames, captions, timestamps)

        scene_texts = [s.summary for s in scenes]
        scene_embeddings = self.embed_texts(scene_texts) if scene_texts else None

        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)

        return _SceneMem(
            scenes=scenes,
            scene_embeddings=scene_embeddings,
            all_frames=frames,
            all_captions=captions,
            duration=video.duration,
        )

    def _segment_scenes(
        self, frames: list[Frame], captions: list[str], timestamps: list[float]
    ) -> list[_Scene]:
        n = len(frames)
        if n == 0:
            return []

        scenes: list[_Scene] = []
        current_start = 0

        for i in range(1, n):
            scene_len = i - current_start
            if scene_len >= self.MAX_SCENE_LEN:
                scenes.append(self._make_scene(current_start, i, frames, captions))
                current_start = i
                continue
            if len(timestamps) > 1:
                gaps = np.diff(timestamps)
                median_gap = float(np.median(gaps)) if len(gaps) > 0 else 1.0
                gap = timestamps[i] - timestamps[i - 1]
                if gap > median_gap * 2.0 and scene_len >= self.MIN_SCENE_LEN:
                    scenes.append(self._make_scene(current_start, i, frames, captions))
                    current_start = i

        if current_start < n:
            scenes.append(self._make_scene(current_start, n, frames, captions))

        return scenes

    def _make_scene(
        self, start: int, end: int, frames: list[Frame], captions: list[str]
    ) -> _Scene:
        scene_frames = frames[start:end]
        scene_caps = captions[start:end]
        summary = " | ".join(scene_caps[:4])
        t_start = scene_frames[0].timestamp
        t_end = scene_frames[-1].timestamp
        return _Scene(
            start_idx=start,
            end_idx=end,
            frames=scene_frames,
            captions=scene_caps,
            summary=summary,
            time_range=(t_start, t_end),
        )

    def _mmr_select(
        self, frame_captions: list[str], query_vec, k: int
    ) -> list[int]:
        if len(frame_captions) <= k:
            return list(range(len(frame_captions)))

        embeddings = np.asarray(self.embed_texts(frame_captions))
        query_vec = np.asarray(query_vec).reshape(-1)
        sims = embeddings @ query_vec

        selected = [int(np.argmax(sims))]

        for _ in range(k - 1):
            remaining = [i for i in range(len(frame_captions)) if i not in selected]
            if not remaining:
                break

            best_score = -float("inf")
            best_idx = remaining[0]
            for i in remaining:
                relevance = float(sims[i])
                redundancy = max(
                    float(embeddings[i] @ embeddings[j]) for j in selected
                )
                score = self.MMR_LAMBDA * relevance - (1 - self.MMR_LAMBDA) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = i
            selected.append(best_idx)

        return sorted(selected)

    def answer_question(
        self, memory: _SceneMem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        qv = self.embed_texts([question])[0]

        scene_idxs = self.topk_indices(qv, memory.scene_embeddings, self.TOP_SCENES)

        selected_frames: list[Frame] = []
        scene_context_lines: list[str] = []

        for si in scene_idxs:
            scene = memory.scenes[si]
            t0, t1 = scene.time_range
            scene_context_lines.append(
                f"Scene ({t0:.0f}s–{t1:.0f}s): {scene.summary[:150]}"
            )
            local_idxs = self._mmr_select(scene.captions, qv, self.FRAMES_PER_SCENE)
            selected_frames.extend(scene.frames[i] for i in local_idxs)

        selected_frames = sorted(selected_frames, key=lambda f: f.timestamp)
        scene_context = "\n".join(scene_context_lines)

        parts = [
            {
                "type": "text",
                "text": PROMPT.format(
                    dur=memory.duration,
                    n_scenes=len(scene_idxs),
                    scene_context=scene_context,
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        parts.append({"type": "text", "text": "\nRelevant frames:"})
        parts += self.render_frames(selected_frames)
        parts.append({"type": "text", "text": INSTRUCTION})

        resp = self.ask_vlm(parts)
        letter = normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        )
        return letter, {
            "scenes_retrieved": [int(i) for i in scene_idxs],
            "num_frames_shown": len(selected_frames),
            "raw": resp[:200],
        }
