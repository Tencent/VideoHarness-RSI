"""Candidate: two-pass caption-then-look verification answering.

Hypothesis: The baseline shows 32 random frames and hopes the answer is visible.
But for long videos, the VLM can reason about WHAT to look at from text alone.
Pass 1 shows all frame captions (text-only, zero visual tokens) and asks the VLM
to identify relevant moments + give a tentative answer. Pass 2 shows the ACTUAL
frames from those moments, letting the VLM visually verify/correct its answer.
This concentrates visual tokens on the frames that matter, selected by the VLM's
own reasoning rather than embedding similarity.

Axes: B (mixed text+image representation) + G (two-pass verify-and-correct
answering, inspired by Homer's verify-and-correct loop).
"""

import json
import re
from typing import Any

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream


PASS1_PROMPT = (
    "You are analyzing a long video ({dur:.0f}s) to answer a question. "
    "Below are timestamped descriptions of {n} moments sampled from the video.\n\n"
    "Moments:\n{moments}\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Identify which 3-6 moments (by their index number) are most relevant, "
    "then give your tentative answer based on the descriptions alone.\n\n"
    'Respond in JSON: {{"relevant_indices": [<int>, ...], '
    '"reasoning": "...", "tentative_answer": "<letter>"}}'
)

PASS2_PROMPT = (
    "You previously analyzed text descriptions of a video and tentatively "
    "answered a question. Now look at the actual frames from the moments you "
    "identified as relevant, and verify or correct your answer.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Your tentative answer was: {tentative}\n"
    "Your reasoning was: {reasoning}\n\n"
    "Examine the frames below carefully. Do they confirm or change your answer?\n"
)

INSTRUCTION = '\nRespond in JSON: {"reasoning": "...", "final_answer": "<letter>"}'


class CaptionThenLook(VideoMemoryHarness):
    """Two-pass: text reasoning → visual verification."""

    FPS = 2.0
    NUM_INGEST = 340
    MAX_VERIFY_FRAMES = 8

    def build_memory(self, video: VideoStream) -> Any:
        frames = video.sample_uniform(self.NUM_INGEST)
        captions = [self.caption_frame(f) for f in frames]
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return {
            "frames": frames,
            "captions": captions,
            "duration": video.duration,
        }

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        frames: list[Frame] = memory["frames"]
        captions: list[str] = memory["captions"]
        duration: float = memory["duration"]

        moments = "\n".join(
            f"  [{i}] {frames[i].timestamp:.1f}s: {captions[i]}"
            for i in range(len(frames))
        )

        pass1_parts = [
            {
                "type": "text",
                "text": PASS1_PROMPT.format(
                    dur=duration,
                    n=len(frames),
                    moments=moments,
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        pass1_resp = self.ask_vlm(pass1_parts)

        relevant_indices = self._parse_indices(pass1_resp, len(frames))
        tentative = extract_json_field(pass1_resp, "tentative_answer") or "?"
        reasoning = extract_json_field(pass1_resp, "reasoning") or ""

        if not relevant_indices:
            relevant_indices = list(range(min(6, len(frames))))

        verify_frames = [frames[i] for i in relevant_indices[:self.MAX_VERIFY_FRAMES]]

        pass2_parts = [
            {
                "type": "text",
                "text": PASS2_PROMPT.format(
                    question=question,
                    options=format_options(options),
                    tentative=tentative,
                    reasoning=reasoning[:300],
                ),
            }
        ]
        pass2_parts += self.render_frames(verify_frames)
        pass2_parts.append({"type": "text", "text": INSTRUCTION})

        pass2_resp = self.ask_vlm(pass2_parts)
        final = extract_json_field(pass2_resp, "final_answer") or pass2_resp
        letter = normalize_choice(final, options)

        if letter == "?":
            letter = normalize_choice(tentative, options)

        return letter, {
            "tentative": tentative,
            "verified_indices": relevant_indices[:self.MAX_VERIFY_FRAMES],
            "num_verify_frames": len(verify_frames),
            "raw_pass2": pass2_resp[:200],
        }

    def _parse_indices(self, resp: str, n: int) -> list[int]:
        raw = extract_json_field(resp, "relevant_indices")
        if raw:
            try:
                indices = json.loads(raw) if raw.startswith("[") else json.loads(f"[{raw}]")
                return [i for i in indices if isinstance(i, int) and 0 <= i < n]
            except (json.JSONDecodeError, TypeError):
                pass
        numbers = re.findall(r"\[(\d+)\]", resp)
        if numbers:
            return [int(x) for x in numbers if int(x) < n][:8]
        numbers = re.findall(r"\b(\d{1,2})\b", raw or "")
        return [int(x) for x in numbers if int(x) < n][:8]
