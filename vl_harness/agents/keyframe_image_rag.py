"""Baseline B2: keyframe + image retrieval.

Ingest: sample keyframes, store each with a visual embedding. Answer: embed the
question text and retrieve the top-k *frames* by text->image similarity, then
show those frames (images) to the VLM. Pays visual tokens only for the few
retrieved frames -- the value of visual retrieval when captions would miss the
answer. (With the stub embedder cross-modal similarity is uninformative; this
harness is meant to shine with a real CLIP/SigLIP backend.)
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

HEADER = (
    "You answer a multiple-choice question about a long video. You are shown "
    "the {k} frames most relevant to the question, retrieved from across the "
    "whole video.\n\nQuestion: {question}\n\nOptions:\n{options}\n"
)
INSTRUCTION = '\nRespond in JSON: {"reasoning": "...", "final_answer": "<letter>"}'


@dataclass
class _ImgMem:
    frames: list[Frame]
    embeddings: Any  # (N, d)


class KeyframeImageRAG(VideoMemoryHarness):
    FPS = 2.0
    NUM_INGEST_FRAMES = 320
    TOP_K = 6

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.NUM_INGEST_FRAMES)
        # Embed frames visually. Mock frames have no pixels; the stub embedder
        # fingerprints their identity so the pipeline still runs.
        images = [f.image if f.image is not None else f.caption for f in frames]
        embeddings = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return _ImgMem(frames, embeddings)

    def answer_question(
        self, memory: _ImgMem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        qv = self.embed_texts([question])[0]
        idxs = self.topk_indices(qv, memory.embeddings, self.TOP_K)
        chosen = [memory.frames[i] for i in idxs]
        parts = [
            {
                "type": "text",
                "text": HEADER.format(
                    k=len(chosen), question=question, options=format_options(options)
                ),
            }
        ]
        parts += self.render_frames(chosen)
        parts.append({"type": "text", "text": INSTRUCTION})
        resp = self.ask_vlm(parts)
        letter = normalize_choice(extract_json_field(resp, "final_answer") or resp, options)
        return letter, {"retrieved": [int(i) for i in idxs]}
