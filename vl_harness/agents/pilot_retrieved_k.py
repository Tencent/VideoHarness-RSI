"""Pilot control B: show the K question-relevant frames, K = the sweep budget.

The other half of the matched pair (see ``pilot_uniform_k``). Ingest embeds a
large candidate pool at write time -- which the budget deliberately does not
constrain -- and answer time selects the K frames whose CLIP embedding is
closest to the question, then shows them in chronological order so the only
difference from the uniform arm is WHICH frames, not how many or in what order.

Paper table name: CLIP-kNN.
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
from .pilot_uniform_k import HEADER, INSTRUCTION


@dataclass
class _PoolMem:
    frames: list[Frame]
    img_emb: Any


class PilotRetrievedK(VideoMemoryHarness):
    """Question-conditioned selection at the sweep's frame budget."""

    FPS = 2.0
    POOL_FRAMES = 320

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.POOL_FRAMES)
        images = [
            f.image if f.image is not None else (f.caption or "") for f in frames
        ]
        img_emb = np.asarray(self.embed_images(images), dtype=np.float32)
        vtok = sum(max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames)
        self.account_ingest(video.video_id, len(frames), vtok)
        return _PoolMem(frames, img_emb)

    def answer_question(
        self, memory: _PoolMem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        if not memory.frames:
            return "?", {"error": "no frames"}

        k = min(self.frame_budget(), len(memory.frames))
        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        idxs = self.topk_indices(qv, memory.img_emb, k)
        frames = [memory.frames[i] for i in sorted(int(i) for i in idxs)]

        parts = [
            {
                "type": "text",
                "text": HEADER.format(
                    k=len(frames), question=question, options=format_options(options)
                ),
            }
        ]
        parts += self.render_frames(frames)
        parts.append({"type": "text", "text": INSTRUCTION})
        resp = self.ask_vlm(parts)
        letter = normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        )
        return letter, {
            "selection": "question_similarity",
            "num_frames_shown": len(frames),
            "pool_size": len(memory.frames),
            "budget": self.frame_budget(),
            "raw": resp[:200],
        }
