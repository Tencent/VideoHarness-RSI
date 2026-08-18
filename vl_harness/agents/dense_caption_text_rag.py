"""Baseline B1: dense caption + text retrieval (the pure-text control).

Ingest: sample frames densely, caption each, store (timestamp, caption) with a
text embedding. Answer: embed the question, retrieve the top-k captions, and
answer from TEXT ONLY (no images shown -> ~0 visual tokens). This is the
"text harness" reference: cheap, but blind to anything captions miss.
"""

from dataclasses import dataclass
from typing import Any

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import VideoStream

PROMPT = (
    "You answer a multiple-choice question about a long video using retrieved "
    "textual descriptions of moments in the video.\n\n"
    "Retrieved moments (timestamp: description):\n{context}\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    'Respond in JSON: {{"reasoning": "...", "final_answer": "<letter>"}}'
)


@dataclass
class _TextMem:
    captions: list[str]
    timestamps: list[float]
    embeddings: Any  # (N, d)


CAPTION_PROMPT = "Write ONE short factual sentence describing the frame."


class DenseCaptionTextRAG(VideoMemoryHarness):
    FPS = 2.0
    NUM_INGEST_FRAMES = 320
    TOP_K = 5

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.NUM_INGEST_FRAMES)
        captions = self.caption_frames(frames, prompt=CAPTION_PROMPT)
        timestamps = [f.timestamp for f in frames]
        embeddings = self.embed_texts(captions)
        # Write-time cost: captioning read every ingested frame once.
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return _TextMem(captions, timestamps, embeddings)

    def answer_question(
        self, memory: _TextMem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        qv = self.embed_texts([question])[0]
        idxs = self.topk_indices(qv, memory.embeddings, self.TOP_K)
        context = "\n".join(
            f"- {memory.timestamps[i]:.1f}s: {memory.captions[i]}" for i in idxs
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
        resp = self.ask_vlm(parts)  # text-only: no visual tokens
        letter = normalize_choice(extract_json_field(resp, "final_answer") or resp, options)
        return letter, {"retrieved": [int(i) for i in idxs]}
