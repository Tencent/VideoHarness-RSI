"""Pilot control A: show K uniformly-spaced frames, where K is the sweep budget.

Half of the matched pair that tests the premise behind the frame-budget sweep:
at a FIXED answer-time budget of K frames, does question-conditioned selection
beat uniform selection? This arm selects uniformly; ``pilot_retrieved_k`` selects
by question-frame similarity. Everything else -- prompt wording, frame count,
rendering order, answer parsing -- is identical between the two, so the accuracy
gap is attributable to the selection policy alone.

Distinct from ``uniform_frames_no_memory`` only in prompt wording: that baseline
tells the model its frames are "sampled uniformly", which would be false for the
retrieval arm, so both pilot arms use a neutral phrasing instead.
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
    "You are shown {k} frames from the video, in chronological order.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n"
)

INSTRUCTION = (
    "\nPlease select the best answer from the options above and directly "
    "provide the letter representing your choice without giving any "
    "explanation."
)


class PilotUniformK(VideoMemoryHarness):
    """Uniform selection at the sweep's frame budget."""

    FPS = 2.0
    MAX_FRAMES = 320

    def build_memory(self, video: VideoStream) -> Any:
        return video

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        video: VideoStream = memory
        # Clamped to frame_budget() at answer time, so this is uniform-K.
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
        letter = normalize_choice(extract_json_field(resp, "final_answer") or resp, options)
        return letter, {
            "selection": "uniform",
            "num_frames_shown": len(frames),
            "budget": self.frame_budget(),
            "raw": resp[:200],
        }
