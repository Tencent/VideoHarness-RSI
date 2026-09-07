"""Pilot arm C: select K frames by caption similarity instead of image similarity.

Single variable versus ``pilot_retrieved_k``: the scoring signal. Same budget,
same neutral prompt, same chronological ordering, same answer parsing.

Paper table name: Caption-kNN. Caption neighbors are mapped back to their
frames for visual answering (not text-only RAG).
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
class _CapMem:
    frames: list[Frame]
    captions: list[str]
    cap_emb: Any


class PilotRetrievedKCaption(VideoMemoryHarness):
    """Question-to-caption selection at the sweep's frame budget."""

    FPS = 2.0
    POOL_FRAMES = 320

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.POOL_FRAMES)
        if not frames:
            return _CapMem([], [], None)
        captions = self.caption_frames(frames)
        cap_emb = np.asarray(self.embed_texts(captions), dtype=np.float32)
        vtok = sum(max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames)
        self.account_ingest(video.video_id, len(frames), vtok)
        return _CapMem(frames, captions, cap_emb)

    def answer_question(
        self, memory: _CapMem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        if not memory.frames:
            return "?", {"error": "no frames"}
        k = min(self.frame_budget(), len(memory.frames))
        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        idxs = self.topk_indices(qv, memory.cap_emb, k)
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
            "selection": "caption_similarity",
            "num_frames_shown": len(frames),
            "pool_size": len(memory.frames),
            "budget": self.frame_budget(),
            "raw": (resp or "")[:200],
        }
