"""Qwen3-VL report protocol for LVBench, INSTRUCT template (Appendix, p37).

``uniform_frames_qwen3_paper_prompt`` transcribes the report's *thinking* template
-- it ends with "Please reason step-by-step ..." -- but we evaluate
Qwen3-VL-8B-Instruct with thinking disabled. The model duly reasons, and the
answer letter is either buried or truncated: that harness parse-fails on 31% of
questions and scores 44.7%, below the plain no-memory baseline. This file uses
the two-line template the report specifies for instruct models on
MVBench | VideoMME | MLVU | LVBench, so the published protocol is reproduced with
the template that matches the model variant being run.

Remaining, unavoidable deviation from the report's numbers: the report caps each
video at 2,048 frames and 224K video tokens (<=640 tokens/frame, 2 fps). A single
H20 at TP=1 gives ~84K tokens of context, so this harness sees ~320 frames and
~83K visual tokens -- roughly 2.7x less visual information in total. The report's
58.0 for Qwen3-VL-8B-Instruct is therefore an upper-budget reference point, not a
number this configuration should be expected to match.
"""

from typing import Any

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import VideoStream

# Verbatim from the report's appendix, "MVBench | VideoMME | MLVU | LVBench -
# For instruct models".
PAPER_INSTRUCT_PROMPT = """Select the best answer to the following multiple-choice question based on the video.
Respond with only the letter (A, B, C, or D) of the correct option.
Question: {question}
{options}"""


class UniformFramesQwen3InstructPrompt(VideoMemoryHarness):
    """Report protocol with the instruct-model answer template."""

    FPS = 2.0
    MAX_FRAMES = 320

    def build_memory(self, video: VideoStream) -> Any:
        return video

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        video: VideoStream = memory
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_FRAMES)
        # The report places the video before the textual question.
        parts = self.render_frames(frames)
        parts.append(
            {
                "type": "text",
                "text": PAPER_INSTRUCT_PROMPT.format(
                    question=question, options=format_options(options)
                ),
            }
        )
        resp = self.ask_vlm(parts)
        letter = normalize_choice(extract_json_field(resp, "final_answer") or resp, options)
        return letter, {
            "raw": (resp or "")[:200],
            "sampled": len(frames),
            "fps": self.FPS,
            "budget": self.frame_budget(),
            "prompt_profile": "qwen3vl_report_instruct",
        }
