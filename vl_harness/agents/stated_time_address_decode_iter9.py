"""Candidate: treat a stated time address as a decode instruction, not a query.

Hypothesis (H1): when a question names WHEN to look ("what appears on the
screen at 21:57?", "what happens from 37:24-37:29?"), that clock expression is
an absolute address into the video. The parent has no path from that address to
pixels: it re-derives location by semantic similarity over its ingest lattice,
so `_select_frames_hybrid` is an argsort whose every output is a lattice point
(median step ~9s), and the navigation stage is shown a decimated timeline and
must guess an index from caption text. Measured on this run's own traces: the
navigation range lands within 30s of the stated instant in only 25% of cases
(median miss 173s), and only 19% of packs contain a frame within 2s of it.
Within that stratum, landing on the stated instant is what separates success
from failure (<=2s: 61.5%; 10-60s: 29.4%; >60s: 30.8%).

Mechanism: parse the address off the question string BEFORE any VLM call, then
`sample_time_range` directly at it -- instants the ingest pool never contained.
Location for these questions therefore costs zero VLM calls and bypasses both
the embedding and the navigation stage. Everything else is unchanged: a
question with no parsable address takes the parent's path exactly, and even an
addressed question keeps the parent's top-scored frames in every slot not spent
on the address, so the address frames are ADDED to the parent's evidence rather
than swapped for a blind jump.

A stated interval can be a single instant or 20 minutes wide, so the spend
adapts: a narrow address becomes a dense burst around the instant (the temporal
resolution the lattice cannot express), a wide one becomes a spread sample
across the named span (where a burst would be no better than the parent).

Axes: A (ingest -- off-lattice re-decode at answer time, addressed by the
question rather than by a sampler) + D (router -- a modality/route decision made
from the question string: literal time address vs semantic retrieval).
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


NAVIGATE_PROMPT = (
    "You are analyzing a long video ({dur:.0f}s, {n} sampled moments) to "
    "answer a question. Below is a timestamped timeline of descriptions:\n\n"
    "{timeline}\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Identify 1-4 time ranges (as start_index-end_index from the list above) "
    "that contain the information needed to answer. Also give your tentative "
    "answer based on the descriptions.\n\n"
    '{{"ranges": [[start_idx, end_idx], ...], '
    '"reasoning": "...", "tentative_answer": "<letter>"}}'
)

ANSWER_PROMPT = (
    "You are answering a question about a long video. The {k} frames below "
    "were selected using both text-based reasoning and visual similarity to "
    "the question.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Please select the best answer from the options above and directly "
    "provide the letter representing your choice without giving any "
    "explanation."
)

# Frames decoded at the address are labelled with their clock time so the model
# can tell which pixels answer "at <time>" apart from the surrounding context.
ADDRESSED_ANSWER_PROMPT = (
    "You are answering a question about a long video. The question refers to a "
    "specific time in the video, so the frames below are given in two groups.\n\n"
    "GROUP 1 ({n_addr} frames, each labelled with its clock time) is the part "
    "of the video the question asks about: {spans}. Read the answer off these "
    "frames.\n"
    "GROUP 2 ({n_ctx} frames) is context sampled from the rest of the video, "
    "for reference only.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Please select the best answer from the options above and directly "
    "provide the letter representing your choice without giving any "
    "explanation."
)

# mm:ss or hh:mm:ss. Minutes/seconds are constrained to 0-59 so scorelines and
# ratios ("2:1", "16:9") cannot be read as addresses; the value must also fall
# inside the video for the parse to be accepted.
_CLOCK = re.compile(r"(?<![\d:.])(\d{1,3}):([0-5]\d)(?::([0-5]\d))?(?![\d:.])")
# Only these joiners make two adjacent clocks one interval; "and"/"," are left
# out because they usually name two separate instants to compare.
_RANGE_JOIN = re.compile(r"\s*(?:-{1,2}|–|—|~|to|until|till|through)\s*\Z")


def parse_time_spans(question: str, duration: float) -> list[tuple[float, float]]:
    """Clock expressions in ``question`` as (start, end) seconds, in order.

    An instant becomes a zero-width span. Two clocks joined by a dash/"to"
    become one interval. Values outside the video are dropped, which is what
    rejects a stray ratio that happens to look like a timestamp.
    """
    if duration <= 0:
        return []
    matches = list(_CLOCK.finditer(question))
    if not matches:
        return []

    def seconds(m: re.Match) -> int:
        h, mi, s = m.group(1), m.group(2), m.group(3)
        if s is not None:
            return int(h) * 3600 + int(mi) * 60 + int(s)
        return int(h) * 60 + int(mi)

    spans: list[tuple[float, float]] = []
    skip = -1
    for i, m in enumerate(matches):
        if i == skip:
            continue
        t0 = seconds(m)
        if not 0.0 <= t0 <= duration:
            continue
        t1 = t0
        if i + 1 < len(matches):
            joiner = question[m.end() : matches[i + 1].start()]
            if _RANGE_JOIN.search(joiner):
                cand = seconds(matches[i + 1])
                if t0 <= cand <= duration:
                    t1 = cand
                    skip = i + 1
        spans.append((float(t0), float(t1)))
    return spans


def _merge_spans(spans: list[tuple[float, float]], pad: float) -> list[tuple[float, float]]:
    """Pad each span and union the overlaps, so a burst is never decoded twice."""
    if not spans:
        return []
    padded = sorted((max(0.0, a - pad), b + pad) for a, b in spans)
    out = [padded[0]]
    for a, b in padded[1:]:
        if a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


class StatedTimeAddressDecodeIter9(VideoMemoryHarness):
    """Parent's hybrid retrieval, plus a literal decode at a stated time."""

    FPS = 2.0
    MAX_INGEST_FRAMES = 320
    MAX_TIMELINE_CAPTIONS = 160
    NAV_WEIGHT = 0.6
    EMBED_WEIGHT = 0.4

    # At most half the request goes to the address, so an addressed question
    # keeps the parent's global evidence in the remaining slots.
    ADDRESS_BUDGET_FRAC = 0.5
    # A stated instant is widened to this half-width before decoding, so the
    # burst brackets the moment instead of betting on one still. The annotated
    # evidence sits within ~15s of the stated time for 81% of these questions,
    # so a few seconds of bracket is the right order.
    NARROW_PAD_S = 3.0
    # Finest spacing worth decoding. Below this, extra frames are near-duplicates
    # that cost slots without adding evidence, so leftover slots go to context
    # instead of packing 20 stills into a 6-second bracket.
    MIN_STRIDE_S = 0.7

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_INGEST_FRAMES)
        captions = self.caption_frames(frames)
        images = [f.image if f.image is not None else (f.caption or "") for f in frames]
        img_emb = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        return {
            "frames": frames,
            "captions": captions,
            "img_emb": img_emb,
            "duration": video.duration,
            # The stream itself, so answer time can decode instants that were
            # never in the ingest pool. _close_memory releases the reader.
            "video": video,
        }

    def _build_timeline(self, frames: list[Frame], captions: list[str]) -> str:
        n = len(frames)
        if n <= self.MAX_TIMELINE_CAPTIONS:
            return "\n".join(
                f"[{i}] {frames[i].timestamp:.1f}s: {captions[i]}"
                for i in range(n)
            )
        step = n / self.MAX_TIMELINE_CAPTIONS
        selected = [int(i * step) for i in range(self.MAX_TIMELINE_CAPTIONS)]
        return "\n".join(
            f"[{idx}] {frames[idx].timestamp:.1f}s: {captions[idx]}"
            for idx in selected
        )

    def _parse_ranges(self, resp: str, n: int) -> list[tuple[int, int]]:
        raw = extract_json_field(resp, "ranges")
        if raw:
            try:
                ranges = (
                    json.loads(raw) if raw.startswith("[") else json.loads(f"[{raw}]")
                )
                result = []
                for r in ranges:
                    if isinstance(r, list) and len(r) == 2:
                        s, e = int(r[0]), int(r[1])
                        s = max(0, min(s, n - 1))
                        e = max(s, min(e, n - 1))
                        result.append((s, e))
                if result:
                    return result
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        nums = re.findall(r"\[(\d+)\s*[-,]\s*(\d+)\]", resp)
        if nums:
            result = []
            for s, e in nums[:4]:
                s, e = int(s), int(e)
                s = max(0, min(s, n - 1))
                e = max(s, min(e, n - 1))
                result.append((s, e))
            return result
        single_nums = re.findall(r"\[(\d+)\]", resp)
        if single_nums:
            return [
                (max(0, int(x) - 2), min(n - 1, int(x) + 2))
                for x in single_nums[:4]
            ]
        return []

    def _compute_hybrid_scores(
        self,
        n_frames: int,
        ranges: list[tuple[int, int]],
        question: str,
        img_emb: Any,
    ) -> np.ndarray:
        """Combine navigation-based and embedding-based relevance scores."""
        nav_scores = np.zeros(n_frames)
        if ranges:
            for s, e in ranges:
                nav_scores[s : e + 1] = 1.0
            margin = 5
            for s, e in ranges:
                for offset in range(1, margin + 1):
                    decay = 1.0 - offset / (margin + 1)
                    if s - offset >= 0:
                        nav_scores[s - offset] = max(nav_scores[s - offset], decay * 0.5)
                    if e + offset < n_frames:
                        nav_scores[e + offset] = max(nav_scores[e + offset], decay * 0.5)
        else:
            nav_scores[:] = 1.0 / n_frames

        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        embed_sims = np.asarray(img_emb, dtype=np.float32) @ qv.reshape(-1)
        embed_min = embed_sims.min()
        embed_max = embed_sims.max()
        if embed_max - embed_min > 1e-6:
            embed_scores = (embed_sims - embed_min) / (embed_max - embed_min)
        else:
            embed_scores = np.ones(n_frames) * 0.5

        hybrid = self.NAV_WEIGHT * nav_scores + self.EMBED_WEIGHT * embed_scores
        return hybrid

    def _select_frames_hybrid(
        self,
        frames: list[Frame],
        scores: np.ndarray,
        budget: int,
    ) -> list[Frame]:
        """Select top-budget frames by hybrid score, maintaining temporal order."""
        top_indices = list(np.argsort(-scores)[:budget])
        top_indices.sort()
        return [frames[i] for i in top_indices]

    def _decode_at_spans(
        self,
        video: VideoStream,
        spans: list[tuple[float, float]],
        budget: int,
    ) -> list[Frame]:
        """Decode up to ``budget`` frames inside the stated spans, off-lattice.

        Frames are split between spans in proportion to width, so a question
        naming two moments gets both. Within a span the count is capped by
        ``MIN_STRIDE_S``: a 6-second bracket earns ~9 frames, not 20, and the
        slots that would have gone to near-duplicates are returned to context.
        """
        if budget <= 0 or not spans:
            return []
        merged = _merge_spans(spans, self.NARROW_PAD_S)
        widths = [max(b - a, 1e-3) for a, b in merged]
        total = sum(widths)
        out: list[Frame] = []
        for (a, b), w in zip(merged, widths):
            share = budget if len(merged) == 1 else max(1, int(round(budget * w / total)))
            # Never denser than MIN_STRIDE_S; a wide span therefore gets an even
            # spread across the named interval rather than a burst at one end.
            share = min(share, max(1, int((b - a) / self.MIN_STRIDE_S) + 1))
            out.extend(video.sample_time_range(a, b, share))
        # Deduplicate by decoded index and keep chronological order.
        seen: set[int] = set()
        uniq: list[Frame] = []
        for f in sorted(out, key=lambda x: x.timestamp):
            if f.index in seen:
                continue
            seen.add(f.index)
            uniq.append(f)
        return uniq[:budget]

    @staticmethod
    def _fmt_span(a: float, b: float) -> str:
        """Name a span in clock form AND seconds.

        ``render_frames(timestamps=True)`` labels frames ``[1317.0s]``, while the
        question says "at 21:57". Giving both forms lets the model match the
        named moment to the labelled pixels without doing the arithmetic.
        """

        def clock(t: float) -> str:
            t = int(round(t))
            return f"{t // 60:d}:{t % 60:02d}"

        if abs(b - a) < 0.5:
            return f"{clock(a)} (={a:.0f}s)"
        return f"{clock(a)}-{clock(b)} (={a:.0f}s-{b:.0f}s)"

    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        frames: list[Frame] = memory["frames"]
        captions: list[str] = memory["captions"]
        img_emb = memory["img_emb"]
        duration: float = memory["duration"]
        video: VideoStream = memory["video"]
        budget = self.frame_budget()

        # The address is read off the question string, before any VLM call, so
        # localization for this stratum costs nothing and cannot be diluted by
        # a similarity score.
        spans = parse_time_spans(question, duration)

        timeline = self._build_timeline(frames, captions)

        nav_parts = [
            {
                "type": "text",
                "text": NAVIGATE_PROMPT.format(
                    dur=duration,
                    n=len(frames),
                    timeline=timeline,
                    question=question,
                    options=format_options(options),
                ),
            }
        ]
        nav_resp = self.ask_vlm(nav_parts)

        ranges = self._parse_ranges(nav_resp, len(frames))
        tentative = extract_json_field(nav_resp, "tentative_answer") or "?"

        hybrid_scores = self._compute_hybrid_scores(
            len(frames), ranges, question, img_emb
        )

        addressed: list[Frame] = []
        if spans:
            addressed = self._decode_at_spans(
                video, spans, int(budget * self.ADDRESS_BUDGET_FRAC)
            )

        if not addressed:
            # No usable address: the parent's path, unchanged.
            selected_frames = self._select_frames_hybrid(
                frames, hybrid_scores, budget
            )
            answer_parts = [
                {
                    "type": "text",
                    "text": ANSWER_PROMPT.format(
                        k=len(selected_frames),
                        question=question,
                        options=format_options(options),
                    ),
                }
            ]
            answer_parts += self.render_frames(selected_frames)
        else:
            # Keep the parent's best frames in every slot the address did not
            # take, and drop parent frames that duplicate what was re-decoded.
            covered = _merge_spans(spans, self.NARROW_PAD_S)
            context = [
                f
                for f in self._select_frames_hybrid(frames, hybrid_scores, budget)
                if not any(a <= f.timestamp <= b for a, b in covered)
            ]
            context = context[: max(0, budget - len(addressed))]
            answer_parts = [
                {
                    "type": "text",
                    "text": ADDRESSED_ANSWER_PROMPT.format(
                        n_addr=len(addressed),
                        n_ctx=len(context),
                        spans=", ".join(self._fmt_span(a, b) for a, b in spans),
                        question=question,
                        options=format_options(options),
                    ),
                }
            ]
            # Timestamps only on the addressed group: they are what makes the
            # named instant identifiable among the context frames.
            answer_parts += self.render_frames(addressed, timestamps=True)
            if context:
                answer_parts.append(
                    {"type": "text", "text": "GROUP 2 -- context from the rest of the video:"}
                )
                answer_parts += self.render_frames(context)
            selected_frames = addressed + context

        answer_resp = self.ask_vlm(answer_parts)
        final = extract_json_field(answer_resp, "final_answer") or answer_resp
        letter = normalize_choice(final, options)

        if letter == "?":
            letter = normalize_choice(tentative, options)

        return letter, {
            "tentative": tentative,
            "ranges": ranges,
            "time_spans": spans,
            "num_addressed_frames": len(addressed),
            "num_selected_frames": len(selected_frames),
            "raw": answer_resp[:200],
        }
