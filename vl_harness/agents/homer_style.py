"""Reference implementation: Homer-style hierarchical memory + verify-and-correct.

Reimplementation of the harness described in *Homer* (arXiv 2607.02588) inside
the VL-Harness interface, so that a published hand-designed long-video harness
can be scored under the same protocol, backbone, split and visual budget as
searched harnesses.

Component mapping (paper -> this file), for reviewers checking faithfulness:

| Paper component                  | Here                                          |
|----------------------------------|-----------------------------------------------|
| Perceptual Buffer (keyframes)    | ``_Mem.frames`` + CLIP image embeddings        |
| EntityGraph                      | ``_Mem.entities``: entity -> frames it occurs  |
| EventGraph (temporal/causal)     | ``_Mem.events``: ordered segment events with   |
|                                  | prev/next links used by NEIGHBOR retrieval     |
| Retrieval mode EVENT             | ``_retrieve`` branch "EVENT"                   |
| Retrieval mode VIDEO             | ``_retrieve`` branch "VIDEO" (global spread)   |
| Retrieval mode NEIGHBOR          | ``_retrieve`` branch "NEIGHBOR" (temporal      |
|                                  | expansion around the best event)               |
| Retrieval mode KEYFRAME          | ``_retrieve`` branch "KEYFRAME" (image sim)    |
| verify-and-correct with re-look  | ``answer_question`` second pass over the       |
|                                  | keyframes supporting the tentative answer      |
| Frozen backbone                  | inherited: VL-Harness never trains the VLM     |

Deliberate deviations, declared so the comparison cannot be dismissed as an
unfair reimplementation:

1. The paper's cross-question self-evolving skill library is NOT implemented.
   It is a learning component, and this file is a static reference point; a
   harness that accumulates skills across questions is exactly what the search
   is allowed to propose, so building it into the baseline would confound the
   comparison this file exists to support.
2. Entity extraction is lexical rather than a dedicated extractor (same
   deviation as ``worldmm_style``, kept identical between the two so neither
   baseline is advantaged).
3. Answer-time frames are capped by ``frame_budget()``, and the verify pass
   spends from the same budget, so the total shown never exceeds the arm's
   budget. At the default (no budget) it behaves as originally specified.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..harness import (
    VideoMemoryHarness,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream
from .worldmm_style import _STOP

MODE_PROMPT = (
    "Choose how to search a long video's memory to answer a question.\n"
    "  EVENT    - find the period whose summary matches the question\n"
    "  VIDEO    - the question is about the video as a whole\n"
    "  NEIGHBOR - the question is about what happens just before/after something\n"
    "  KEYFRAME - the question is about a specific visual detail\n\n"
    "Question: {question}\n\nReply with exactly one word from the list above."
)

ANSWER_PROMPT = (
    "Answer the multiple-choice question about the video.\n\n{context}\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Please select the best answer from the options above and directly provide "
    "the letter representing your choice without giving any explanation."
)

VERIFY_PROMPT = (
    "You previously answered '{tentative}' to the question below. Here are the "
    "frames that most support that choice, shown again for verification.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "If the frames confirm your answer, repeat it. If they contradict it, give "
    "the better option instead. Reply with the letter only."
)

_NEIGHBOR_CUES = ("before", "after", "then", "next", "following", "prior", "earlier", "later")
_GLOBAL_CUES = ("overall", "summar", "main topic", "throughout", "how many times", "in total")


@dataclass
class _Mem:
    frames: list[Frame]
    captions: list[str]
    timestamps: list[float]
    img_emb: Any
    events: list[dict[str, Any]] = field(default_factory=list)
    event_emb: Any = None
    entities: dict[str, list[int]] = field(default_factory=dict)
    duration: float = 0.0


class HomerStyle(VideoMemoryHarness):
    """Hierarchical memory, four retrieval modes, verify-and-correct answering."""

    FPS = 2.0
    POOL_FRAMES = 320
    EVENT_WINDOW = 20
    NEIGHBOR_SPAN = 1        # events on each side of the best-matching event
    VERIFY_FRACTION = 0.25   # share of the budget reserved for the re-look pass

    # ── ingest: perceptual buffer + entity graph + event graph ──────────
    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.POOL_FRAMES)
        if not frames:
            return _Mem([], [], [], None)
        captions = self.caption_frames(frames)
        timestamps = [f.timestamp for f in frames]
        img_emb = np.asarray(self.embed_images([f.image for f in frames]), dtype=np.float32)

        events = []
        w = max(1, self.EVENT_WINDOW)
        for start in range(0, len(captions), w):
            end = min(start + w, len(captions))
            events.append(
                {
                    "idx": len(events),
                    "t_start": timestamps[start],
                    "t_end": timestamps[end - 1],
                    "frame_range": (start, end),
                    "text": (
                        f"[{timestamps[start]:.0f}-{timestamps[end - 1]:.0f}s] "
                        + " ".join(captions[start:end])
                    ),
                }
            )
        event_emb = np.asarray(
            self.embed_texts([e["text"] for e in events]), dtype=np.float32
        ) if events else None

        entities: dict[str, list[int]] = {}
        for i, cap in enumerate(captions):
            for tok in re.findall(r"[A-Za-z][A-Za-z\-']+", cap.lower()):
                if len(tok) < 4 or tok in _STOP:
                    continue
                entities.setdefault(tok, []).append(i)
        keep = {k for k, _ in Counter({k: len(v) for k, v in entities.items()}).most_common(40)}
        entities = {k: v for k, v in entities.items() if k in keep}

        vtok = sum(max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames)
        self.account_ingest(video.video_id, len(frames), vtok)
        return _Mem(
            frames=frames,
            captions=captions,
            timestamps=timestamps,
            img_emb=img_emb,
            events=events,
            event_emb=event_emb,
            entities=entities,
            duration=video.duration or (timestamps[-1] if timestamps else 0.0),
        )

    # ── retrieval mode selection ────────────────────────────────────────
    def _select_mode(self, question: str) -> str:
        q = question.lower()
        if any(c in q for c in _NEIGHBOR_CUES):
            return "NEIGHBOR"
        if any(c in q for c in _GLOBAL_CUES):
            return "VIDEO"
        try:
            resp = self.ask_vlm(
                [{"type": "text", "text": MODE_PROMPT.format(question=question)}]
            )
        except Exception:
            return "EVENT"
        up = (resp or "").upper()
        for mode in ("NEIGHBOR", "KEYFRAME", "VIDEO", "EVENT"):
            if mode in up:
                return mode
        return "EVENT"

    def _retrieve(
        self, memory: _Mem, question: str, mode: str, budget: int
    ) -> tuple[list[Frame], str]:
        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        n = len(memory.frames)

        if mode == "VIDEO" or not memory.events:
            step = max(1, n // max(1, budget))
            frames = memory.frames[::step][:budget]
            return frames, "Whole-video view."

        if mode == "KEYFRAME":
            idxs = self.topk_indices(qv, memory.img_emb, budget)
            frames = [memory.frames[i] for i in sorted(int(i) for i in idxs)]
            return frames, "Keyframes matching the question."

        best = int(self.topk_indices(qv, memory.event_emb, 1)[0])
        if mode == "NEIGHBOR":
            lo = max(0, best - self.NEIGHBOR_SPAN)
            hi = min(len(memory.events) - 1, best + self.NEIGHBOR_SPAN)
        else:  # EVENT
            lo = hi = best
        f0 = memory.events[lo]["frame_range"][0]
        f1 = memory.events[hi]["frame_range"][1]
        span = memory.frames[f0:f1]
        step = max(1, len(span) // max(1, budget))
        frames = span[::step][:budget]
        summary = "\n".join(
            f"- {memory.events[i]['text'][:400]}" for i in range(lo, hi + 1)
        )
        return frames, f"Event memory:\n{summary}"

    # ── answer: retrieve, answer, then verify-and-correct ───────────────
    def answer_question(
        self, memory: _Mem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        if not memory.frames:
            return "?", {"error": "no frames"}

        total_budget = self.frame_budget()
        n_verify = max(1, int(total_budget * self.VERIFY_FRACTION))
        n_first = max(1, total_budget - n_verify)

        mode = self._select_mode(question)
        frames, context = self._retrieve(memory, question, mode, n_first)

        parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": ANSWER_PROMPT.format(
                    context=context + "\n",
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        parts += self.render_frames(frames)
        resp = self.ask_vlm(parts)
        tentative = normalize_choice(resp, options)

        # verify-and-correct: re-look at the frames most consistent with the
        # tentative option, and allow the model to overturn itself.
        revised = tentative
        try:
            opt_idx = ord(tentative) - ord("A") if tentative != "?" else -1
            if 0 <= opt_idx < len(options):
                ov = np.asarray(self.embed_texts([options[opt_idx]])[0], dtype=np.float32)
                idxs = self.topk_indices(ov, memory.img_emb, n_verify)
                look = [memory.frames[i] for i in sorted(int(i) for i in idxs)]
                vparts: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": VERIFY_PROMPT.format(
                            tentative=tentative,
                            question=question,
                            options=format_options(options),
                        ),
                    }
                ]
                vparts += self.render_frames(look)
                vresp = self.ask_vlm(vparts)
                candidate = normalize_choice(vresp, options)
                if candidate != "?":
                    revised = candidate
                frames = frames + look
        except Exception:
            pass

        return revised, {
            "strategy": "homer_style",
            "mode": mode,
            "tentative": tentative,
            "revised": revised,
            "changed": revised != tentative,
            "num_frames_shown": len(frames),
            "budget": total_budget,
            "raw": (resp or "")[:200],
        }
