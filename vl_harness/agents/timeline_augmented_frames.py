"""Candidate: timeline-augmented full-frame viewing.

Hypothesis: The champion (52%) shows 320 frames but the VLM must visually scan
ALL of them linearly without structural guidance. This is particularly costly for
long videos (30-60+ min) where the VLM needs to locate specific moments among
hundreds of near-identical frames. By building a SPARSE text timeline at ingest
(captioning every 10th frame = 32 anchor captions) and prepending it to the
visual frames at answer time, we give the VLM a "table of contents" that lets it
quickly identify WHICH frames to focus on. The visual frame budget stays at 320
(same cost as champion), but the VLM gets navigational context that helps it
find the needle in the haystack.

The mechanism change vs champion: adds a TEXT NAVIGATION LAYER on top of the same
visual coverage. Unlike dense_caption_text_rag (text-only), we KEEP all 320 visual
frames. Unlike hybrid_router_rag (text+few images), we show the FULL visual set.
The sparse timeline is cheap (~32 captions, <1K tokens of text) but provides
structural scaffolding for visual navigation.

Axes: B (mixed text+image representation in single pass) +
C (explicit temporal hierarchy via anchored text timeline).
"""

import re
from typing import Any

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream

CAPTION_BATCH_PROMPT = (
    "You are given {n} video frames in chronological order. "
    "Write ONE short factual caption (a single sentence) for EACH frame, "
    "describing exactly what is visible in that frame. "
    "Output ONLY a numbered list:\n"
    "1. <caption for frame 1>\n"
    "...\n"
    "{n}. <caption for frame {n}>"
)

_NUM_RE = re.compile(r"^\s*(\d+)[.):]\s*(.+)$")

PROMPT_WITH_TIMELINE = (
    "Answer the multiple-choice question about the video.\n\n"
    "VIDEO TIMELINE (key moments):\n{timeline}\n\n"
    "You are also shown {k} frames sampled uniformly across the video "
    "(covering the full duration). Use the timeline above to help locate "
    "the relevant frames.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n"
    "\nPlease select the best answer from the options above and directly "
    "provide the letter representing your choice without giving any "
    "explanation."
)

PROMPT_NO_TIMELINE = (
    "Answer the multiple-choice question about the video. "
    "You are shown {k} frames sampled uniformly across the video.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n"
    "\nPlease select the best answer from the options above and directly "
    "provide the letter representing your choice without giving any "
    "explanation."
)


class TimelineAugmentedFrames(VideoMemoryHarness):
    """Champion-equivalent visual coverage + sparse text timeline for navigation."""

    MAX_FRAMES = 320
    ANCHOR_INTERVAL = 10
    CAPTION_BATCH_SIZE = 4

    def _caption_one(self, frame: Frame) -> str:
        if frame.caption is not None:
            return frame.caption
        parts = [
            {"type": "text", "text": "Describe this video frame in one factual sentence."},
            {"type": "image", "image": frame.image, "size": frame.size},
        ]
        resp = self._vlm(parts, enable_thinking=False, max_tokens=128)
        return resp.strip()

    def _caption_batch(self, frames: list[Frame]) -> list[str]:
        if not frames:
            return []
        if len(frames) == 1:
            return [self._caption_one(frames[0])]
        if any(getattr(f, "caption", None) is not None for f in frames):
            return [self._caption_one(f) for f in frames]
        parts: list[dict] = [
            {"type": "text", "text": CAPTION_BATCH_PROMPT.format(n=len(frames))}
        ]
        for f in frames:
            parts.append({"type": "image", "image": f.image, "size": f.size})
        resp = self._vlm(parts, enable_thinking=False, max_tokens=max(512, len(frames) * 128))
        parsed = self._parse_batch(resp, len(frames))
        if parsed:
            return parsed
        return [self._caption_one(f) for f in frames]

    def _parse_batch(self, resp: str, n: int) -> list[str] | None:
        if not resp:
            return None
        found: dict[int, str] = {}
        for line in resp.splitlines():
            m = _NUM_RE.match(line)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= n and idx not in found:
                    found[idx] = m.group(2).strip()
        if len(found) == n:
            return [found[i] for i in range(1, n + 1)]
        return None

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_FRAMES)
        anchor_indices = list(range(0, len(frames), self.ANCHOR_INTERVAL))
        anchor_frames = [frames[i] for i in anchor_indices]

        captions: list[str] = []
        bs = self.CAPTION_BATCH_SIZE
        for i in range(0, len(anchor_frames), bs):
            captions.extend(self._caption_batch(anchor_frames[i : i + bs]))

        anchor_timestamps = [frames[i].timestamp for i in anchor_indices]

        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28)
            for f in anchor_frames
        )
        self.account_ingest(video.video_id, len(anchor_frames), vtok)

        return {
            "video": video,
            "anchor_timestamps": anchor_timestamps,
            "anchor_captions": captions,
        }

    def _build_timeline(self, timestamps: list[float], captions: list[str]) -> str:
        lines = []
        for ts, cap in zip(timestamps, captions):
            m = int(ts // 60)
            s = int(ts % 60)
            lines.append(f"[{m:02d}:{s:02d}] {cap}")
        return "\n".join(lines)

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        video: VideoStream = memory["video"]
        anchor_ts = memory["anchor_timestamps"]
        anchor_caps = memory["anchor_captions"]

        frames = self.sample_ingest_frames(video, max_frames=self.MAX_FRAMES)

        if anchor_caps:
            timeline = self._build_timeline(anchor_ts, anchor_caps)
            parts = self.render_frames(frames)
            parts.append(
                {
                    "type": "text",
                    "text": PROMPT_WITH_TIMELINE.format(
                        timeline=timeline,
                        k=len(frames),
                        question=question,
                        options=format_options(options),
                    ),
                }
            )
            has_timeline = True
        else:
            parts = self.render_frames(frames)
            parts.append(
                {
                    "type": "text",
                    "text": PROMPT_NO_TIMELINE.format(
                        k=len(frames),
                        question=question,
                        options=format_options(options),
                    ),
                }
            )
            has_timeline = False

        resp = self.ask_vlm(parts)
        answer = extract_json_field(resp, "final_answer") or resp
        letter = normalize_choice(answer, options)
        return letter, {
            "has_timeline": has_timeline,
            "num_anchors": len(anchor_caps),
            "num_frames": len(frames),
            "raw": resp[:200],
        }
