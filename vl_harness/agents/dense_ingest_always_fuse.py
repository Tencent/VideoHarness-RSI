"""Candidate: dense ingest with always-fuse multimodal retrieval.

Hypothesis: The hybrid_router_rag uses a brittle keyword-based router to decide
whether to show captions OR frames. But questions often benefit from BOTH
modalities simultaneously — textual context provides narrative understanding while
visual frames provide fine-grained detail. Instead of routing to one modality, we
ALWAYS retrieve and present BOTH: text→text retrieval finds semantically relevant
captions (which provide temporal context), and text→image retrieval finds visually
relevant frames (which provide appearance details). Both are presented together in
chronological order, giving the VLM a rich multimodal "story" view of the relevant
portions of the video. Combined with denser 64-frame ingest for better temporal
resolution and temporal deduplication to ensure diversity.

Axes: A (dense 64-frame ingest for better temporal coverage) + D (always-fuse:
no routing, both modalities always present — fundamentally different from
keyword routing) + F (chronological evidence packing with temporal dedup to
maximize information density).
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
    "You answer a multiple-choice question about a long video ({dur:.0f}s). "
    "Below are retrieved textual descriptions and visual frames from the most "
    "relevant moments, presented in chronological order.\n\n"
    "Retrieved descriptions (providing narrative context):\n{context}\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "The following frames show the visual details from these moments:\n"
)
INSTRUCTION = '\nRespond in JSON: {"reasoning": "...", "final_answer": "<letter>"}'


class DenseIngestAlwaysFuse(VideoMemoryHarness):
    """Dense ingest + always-fuse text and image retrieval."""

    NUM_INGEST = 340
    TOP_K_TEXT = 8
    TOP_K_IMG = 8
    MAX_SHOW_FRAMES = 10
    DEDUP_WINDOW_SEC = 15.0

    def build_memory(self, video: VideoStream) -> Any:
        frames = video.sample_uniform(self.NUM_INGEST)
        captions = [self.caption_frame(f) for f in frames]
        text_emb = self.embed_texts(captions)
        images = [f.image if f.image is not None else f.caption for f in frames]
        img_emb = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return {
            "frames": frames,
            "captions": captions,
            "text_emb": text_emb,
            "img_emb": img_emb,
            "duration": video.duration,
        }

    def _fused_retrieval(
        self, query_vec, memory: dict
    ) -> tuple[list[int], list[int]]:
        """Retrieve by both modalities, merge with temporal dedup."""
        text_emb = np.asarray(memory["text_emb"])
        img_emb = np.asarray(memory["img_emb"])
        frames = memory["frames"]
        q = np.asarray(query_vec).reshape(-1)

        text_sims = text_emb @ q
        img_sims = img_emb @ q

        text_top = list(np.argsort(-text_sims)[: self.TOP_K_TEXT])
        img_top = list(np.argsort(-img_sims)[: self.TOP_K_IMG])

        all_candidates = list(set(text_top + img_top))
        scores = {
            idx: max(float(text_sims[idx]), float(img_sims[idx]))
            for idx in all_candidates
        }
        sorted_candidates = sorted(all_candidates, key=lambda x: -scores[x])

        selected: list[int] = []
        for idx in sorted_candidates:
            t = frames[idx].timestamp
            if not any(
                abs(frames[s].timestamp - t) < self.DEDUP_WINDOW_SEC
                for s in selected
            ):
                selected.append(idx)
            if len(selected) >= self.MAX_SHOW_FRAMES:
                break

        selected.sort(key=lambda i: frames[i].timestamp)

        caption_indices = sorted(
            set(text_top) | set(i for i in selected),
            key=lambda i: frames[i].timestamp,
        )

        return caption_indices, selected

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        frames: list[Frame] = memory["frames"]
        captions: list[str] = memory["captions"]
        duration: float = memory["duration"]

        qv = self.embed_texts([question])[0]
        caption_indices, frame_indices = self._fused_retrieval(qv, memory)

        context = "\n".join(
            f"- {frames[i].timestamp:.1f}s: {captions[i]}"
            for i in caption_indices
        )
        shown_frames = [frames[i] for i in frame_indices]

        parts = [
            {
                "type": "text",
                "text": PROMPT.format(
                    dur=duration,
                    context=context,
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        parts += self.render_frames(shown_frames)
        parts.append({"type": "text", "text": INSTRUCTION})

        resp = self.ask_vlm(parts)
        letter = normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        )
        return letter, {
            "num_captions": len(caption_indices),
            "num_frames_shown": len(shown_frames),
            "frame_indices": frame_indices,
            "raw": resp[:200],
        }
