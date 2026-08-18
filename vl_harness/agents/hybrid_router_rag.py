"""Baseline B3: hybrid multimodal memory + cross-modal retrieval router.

Ingest: store BOTH textual captions (text embedding) and keyframes (visual
embedding). Answer: a lightweight router picks a retrieval modality from the
question type:
  - reasoning / summary questions ("why", "how", "overall") -> TEXT captions
  - appearance / localization questions ("what", "where", "color", "who") -> FUSION
    (retrieved captions + retrieved frames shown together)
  - default -> FUSION

This is the strong baseline that embodies the VL-Harness thesis (multimodal
memory + cross-modal routing) and serves as the evolution seed. The router
policy, retrieval counts, and packing are exactly the kind of logic a proposer
can rewrite.
"""

from dataclasses import dataclass
from typing import Any

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream

_REASONING_CUES = ("why", "how", "reason", "cause", "overall", "summar", "purpose")
_APPEARANCE_CUES = ("what", "where", "who", "color", "colour", "wearing", "many", "count")

PROMPT = (
    "You answer a multiple-choice question about a long video using a mix of "
    "retrieved textual descriptions and retrieved frames.\n\n"
    "Retrieved descriptions (timestamp: text):\n{context}\n\n"
    "Question: {question}\n\nOptions:\n{options}\n"
)
INSTRUCTION = '\nRespond in JSON: {"reasoning": "...", "final_answer": "<letter>"}'


@dataclass
class _HybridMem:
    frames: list[Frame]
    captions: list[str]
    timestamps: list[float]
    text_emb: Any
    img_emb: Any


class HybridRouterRAG(VideoMemoryHarness):
    FPS = 2.0
    NUM_INGEST_FRAMES = 320
    TOP_K_TEXT = 5
    TOP_K_IMG = 4

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.NUM_INGEST_FRAMES)
        captions = self.caption_frames(frames)
        timestamps = [f.timestamp for f in frames]
        text_emb = self.embed_texts(captions)
        images = [f.image if f.image is not None else f.caption for f in frames]
        img_emb = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return _HybridMem(frames, captions, timestamps, text_emb, img_emb)

    def route(self, question: str) -> str:
        q = question.lower()
        if any(c in q for c in _REASONING_CUES):
            return "text"
        if any(c in q for c in _APPEARANCE_CUES):
            return "fusion"
        return "fusion"

    def answer_question(
        self, memory: _HybridMem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        modality = self.route(question)
        qv = self.embed_texts([question])[0]

        text_idxs = self.topk_indices(qv, memory.text_emb, self.TOP_K_TEXT)
        context = "\n".join(
            f"- {memory.timestamps[i]:.1f}s: {memory.captions[i]}" for i in text_idxs
        )
        parts = [
            {
                "type": "text",
                "text": PROMPT.format(
                    context=context,
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        img_idxs: list[int] = []
        if modality == "fusion":
            img_idxs = self.topk_indices(qv, memory.img_emb, self.TOP_K_IMG)
            frames = [memory.frames[i] for i in img_idxs]
            parts.append({"type": "text", "text": "\nRelevant frames:"})
            parts += self.render_frames(frames)
        parts.append({"type": "text", "text": INSTRUCTION})

        resp = self.ask_vlm(parts)
        letter = normalize_choice(extract_json_field(resp, "final_answer") or resp, options)
        return letter, {
            "modality": modality,
            "text_retrieved": [int(i) for i in text_idxs],
            "img_retrieved": [int(i) for i in img_idxs],
        }
