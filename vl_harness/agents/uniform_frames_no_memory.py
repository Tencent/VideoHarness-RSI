"""Baseline B0: 2-fps frame sampling, no memory, no retrieval.

Sampling follows the Qwen3-VL report's video-evaluation rate (2 fps for all
non-Charades-STA benchmarks). The 340-frame cap is the single-H20 TP=1 vLLM
context budget; for longer videos it reduces the effective rate rather than
sending a prompt that cannot fit in the engine's 80K-token context.
"""

from typing import Any

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import VideoStream

HEADER = (
    "Answer the multiple-choice question about the video. "
    "You are shown {k} frames sampled uniformly across the video.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n"
)

INSTRUCTION = (
    "\nPlease select the best answer from the options above and directly "
    "provide the letter representing your choice without giving any "
    "explanation."
)


class UniformFramesNoMemory(VideoMemoryHarness):
    """No-memory lower bound using the paper's 2-fps target rate."""

    FPS = 2.0
    MAX_FRAMES = 320

    def build_memory(self, video: VideoStream) -> Any:
        # No preprocessing: keep the stream and sample lazily at answer time.
        return video

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        video: VideoStream = memory
        # Opening the reader populates duration for real videos.
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_FRAMES)
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
        answer = extract_json_field(resp, "final_answer") or resp
        letter = normalize_choice(answer, options)
        return letter, {
            "raw": resp[:200],
            "sampled": len(frames),
            "fps": self.FPS,
            "duration_s": round(video.duration, 1) if video.duration else None,
        }
