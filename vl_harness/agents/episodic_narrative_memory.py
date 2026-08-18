"""Candidate: episodic narrative memory with VLM-navigated two-pass answering.

Hypothesis: The baseline dumps 32 frames without structure, forcing the VLM to
visually scan everything linearly. LVBench questions often require locating a
specific event within a long narrative (e.g., "what happens after the second
hunting", "what does the maid do from 13:11-13:24"). By building an EVENT-LEVEL
narrative structure (not frame-level) where consecutive frames are grouped into
semantic events, we give the VLM a "table of contents" it can navigate. In pass 1,
the VLM reads only the event summaries (zero visual tokens) and identifies which
events are relevant. In pass 2, it examines the actual frames from those events.
This differs from caption_then_look (which lists individual frame captions) by
providing a higher-level semantic segmentation that better supports narrative
reasoning about event sequences and temporal relationships.

Axes: B (mixed text+image representation at different passes) + C (event-level
granularity with semantic segmentation) + G (two-pass VLM-navigated answering
with structured narrative overview).
"""

import json
import re
from typing import Any

import numpy as np

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream


PASS1_PROMPT = (
    "You are analyzing a long video ({dur:.0f}s) to answer a question. "
    "The video has been segmented into {n_events} events. Below is a structured "
    "narrative overview:\n\n{narrative}\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Which events (by number) contain information needed to answer this question? "
    "Select 2-4 events. Also give your tentative answer based on the descriptions.\n\n"
    'Respond in JSON: {{"relevant_events": [<int>, ...], '
    '"reasoning": "...", "tentative_answer": "<letter>"}}'
)

PASS2_PROMPT = (
    "You are verifying your answer about a long video by examining the actual "
    "frames from the relevant events.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Your tentative answer was: {tentative}\n"
    "Your reasoning: {reasoning}\n\n"
    "Examine the frames below from the relevant events. "
    "Confirm or correct your answer.\n"
)

INSTRUCTION = '\nRespond in JSON: {"reasoning": "...", "final_answer": "<letter>"}'


class EpisodicNarrativeMemory(VideoMemoryHarness):
    """Event-level narrative memory with two-pass VLM navigation."""

    NUM_INGEST = 340
    SIM_THRESHOLD = 0.65
    MIN_EVENT_FRAMES = 3
    MAX_EVENT_FRAMES = 8
    MAX_VERIFY_FRAMES = 10
    MAX_EVENTS_TO_SHOW = 4

    def build_memory(self, video: VideoStream) -> Any:
        frames = video.sample_uniform(self.NUM_INGEST)
        captions = [self.caption_frame(f) for f in frames]
        cap_emb = self.embed_texts(captions)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)

        events = self._segment_events(frames, captions, cap_emb)

        return {
            "frames": frames,
            "captions": captions,
            "events": events,
            "duration": video.duration,
        }

    def _segment_events(
        self, frames: list[Frame], captions: list[str], embeddings
    ) -> list[dict[str, Any]]:
        n = len(frames)
        emb = np.asarray(embeddings)
        events = []
        current_start = 0

        for i in range(1, n):
            seg_len = i - current_start
            sim = float(emb[i] @ emb[i - 1])

            if (sim < self.SIM_THRESHOLD and seg_len >= self.MIN_EVENT_FRAMES) or \
               seg_len >= self.MAX_EVENT_FRAMES:
                events.append(self._make_event(
                    current_start, i, frames, captions
                ))
                current_start = i

        if current_start < n:
            events.append(self._make_event(current_start, n, frames, captions))

        return events

    def _make_event(
        self, start: int, end: int, frames: list[Frame], captions: list[str]
    ) -> dict[str, Any]:
        event_caps = captions[start:end]
        t0 = frames[start].timestamp
        t1 = frames[end - 1].timestamp
        summary = " → ".join(event_caps[:3])
        if len(event_caps) > 3:
            summary += f" (+ {len(event_caps) - 3} more moments)"
        return {
            "start_idx": start,
            "end_idx": end,
            "time_range": (t0, t1),
            "summary": summary,
            "frame_indices": list(range(start, end)),
        }

    def _build_narrative(self, events: list[dict]) -> str:
        lines = []
        for i, ev in enumerate(events):
            t0, t1 = ev["time_range"]
            lines.append(f"[Event {i+1}] ({t0:.0f}s–{t1:.0f}s): {ev['summary']}")
        return "\n".join(lines)

    def _parse_event_indices(self, resp: str, n_events: int) -> list[int]:
        raw = extract_json_field(resp, "relevant_events")
        if raw:
            try:
                indices = json.loads(raw) if raw.startswith("[") else json.loads(f"[{raw}]")
                return [i - 1 for i in indices if isinstance(i, int) and 1 <= i <= n_events]
            except (json.JSONDecodeError, TypeError):
                pass
        numbers = re.findall(r"Event\s*(\d+)", resp)
        if numbers:
            return [int(x) - 1 for x in numbers if 1 <= int(x) <= n_events][:4]
        numbers = re.findall(r"\b(\d{1,2})\b", raw or "")
        return [int(x) - 1 for x in numbers if 1 <= int(x) <= n_events][:4]

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        frames: list[Frame] = memory["frames"]
        events: list[dict] = memory["events"]
        duration: float = memory["duration"]

        narrative = self._build_narrative(events)

        # Pass 1: VLM navigates the narrative (text-only)
        pass1_parts = [
            {
                "type": "text",
                "text": PASS1_PROMPT.format(
                    dur=duration,
                    n_events=len(events),
                    narrative=narrative,
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        pass1_resp = self.ask_vlm(pass1_parts)

        event_indices = self._parse_event_indices(pass1_resp, len(events))
        tentative = extract_json_field(pass1_resp, "tentative_answer") or "?"
        reasoning = extract_json_field(pass1_resp, "reasoning") or ""

        if not event_indices:
            event_indices = list(range(min(3, len(events))))

        event_indices = event_indices[:self.MAX_EVENTS_TO_SHOW]

        # Gather frames from selected events
        verify_frame_indices: list[int] = []
        for ei in event_indices:
            if 0 <= ei < len(events):
                verify_frame_indices.extend(events[ei]["frame_indices"])
        verify_frame_indices = verify_frame_indices[:self.MAX_VERIFY_FRAMES]
        verify_frames = [frames[i] for i in verify_frame_indices]

        # Pass 2: Visual verification
        pass2_parts = [
            {
                "type": "text",
                "text": PASS2_PROMPT.format(
                    question=question,
                    options=format_options(options),
                    tentative=tentative,
                    reasoning=reasoning[:400],
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
            "selected_events": [i + 1 for i in event_indices],
            "num_verify_frames": len(verify_frames),
            "raw_pass2": pass2_resp[:200],
        }
