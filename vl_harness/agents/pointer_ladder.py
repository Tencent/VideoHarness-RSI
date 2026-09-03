"""Coarse-to-fine pointer ladder: point, narrow, point again, then commit.

Parent: ``agents/survey_then_commit.py`` (frontier F_t). Its stated-clock
bypass, its AKS anchor top-up, its burst decoder and its verbatim answer
prompt are all inherited; the only thing that changes is how many times the
localiser is allowed to *look before it aims*.

The parent's own val traces say the localiser is not weak -- it is
scale-limited. Measured over its 295 survey calls, with the grid geometry held
constant at 672x324 / 12 thumbs / 27 images for every single one of them:

    answer window as fraction of the span shown    pointer hit rate
        >= 6.25%                                       75.9%   (n=54)
        2.0 - 6.0%                                     65.4%   (n=26)
        0.6 - 2.0%                                     21.9%   (n=64)
        < 1%                                            8.8%   (n=181)

A constant cannot explain an outcome that swings 8.8% -> 75.9%, so thumbnail
legibility is not the binding variable; the target's *relative* size is. And
the pointer is not blind in the hard regime either -- against a uniform-random
baseline it runs 6-37x lift there. It carries real information at every scale
and simply cannot resolve a 0.4%-wide target in ONE decision spanning a
54-minute video.

Two consequences of asking for that decision only once, both visible in the
parent's traces:

* **Precision.** Pointwise hit is 28.5%, but the *bracket* around what it
  names contains the answer 74.2% of the time (half-width D/8, three hedged
  candidates). The coarse signal is already good; only the precision is
  missing. That gap -- 74% vs 28% -- is what a second look converts.
* **Aliasing.** The one grid it draws is the ingest pool itself, ~10.3s
  apart, while the median answer window is 8s. So 24.7% of answer windows
  contain ZERO thumbnails and 46.8% contain at most one. For those questions
  the right moment is not mis-ranked, it is *absent from the input*, and no
  amount of re-ranking a single pool can recover it.

Mechanism (axis E retrieval-algorithm, as a *resolution schedule*): make
localisation recursive, so every level faces a target/span ratio inside the
regime the same frozen pointer already handles.

1. **L0 -- point coarsely.** The parent's survey, unchanged: whole-pool grids,
   three hedged timestamps.
2. **Narrow, keeping every hedge alive.** Take a bracket around each named
   moment and keep the UNION of all three. A wrong first pick must not be able
   to lock the ladder into the wrong region, so the deeper level still sees
   every region L0 considered -- narrowing without committing.
3. **L1 -- point again, finer.** Re-decode *inside* that union at spacing the
   pool never had (~4-6x finer than the whole-video grid), re-tile, and ask the
   same question again. Frames that were never in the ingest pool now enter the
   pointer's input, which is the only way the aliased 24.7% become addressable.
   Relative target size rises ~7x (0.55% -> ~4%), moving the pointer from its
   8.8-21.9% band toward its 52-65% band.
4. **Commit.** The parent's burst, aimed by the refined timestamps.

The narrowing factor is deliberately not a tuned constant. Across half-widths
D/8..D/32 the predicted joint hit is flat at 32.6-33.7%, so the mechanism is
the extra *level*, not the fraction. What the schedule does enforce is the
thing that makes a level worth a call at all: each rung must let the pointer
see at least 2x finer than the rung above, measured on the frames it actually
decoded, or it is refused rather than run.

Guards, all inherited or additive: a stated clock still bypasses the ladder
entirely (proven capability, exact address); every level that fails to parse,
names nothing, or would not actually narrow returns the level above it, so the
ladder degrades rung by rung to the parent rather than to nothing; the answer
request keeps whole-video AKS anchors so even a fully mis-aimed ladder shows
what the parent would have shown; and the answer prompt is the parent's
verbatim, so the measured difference is the resolution schedule and nothing
else.
"""

from __future__ import annotations

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
from .aks import ALL_DEPTH, aks_select
from .stated_interval_router import parse_stated_interval
from .survey_then_commit import (
    BURST_SHARE,
    GRID_COLS,
    GRID_ROWS,
    MAX_GRID_IMAGES,
    MAX_REGIONS,
    MIN_BURST_HALF_S,
    POOL_STEP_MULT,
    PAPER_INSTRUCT_PROMPT,
    _clamp,
    _fmt_clock,
    _grid_image,
    parse_survey_times,
)

AGENT_NAME = "pointer_ladder"

# ---- resolution schedule ---------------------------------------------------
# Half-width of the bracket kept around each candidate, as a fraction of the
# span that level was shown. Chosen so that even the worst case -- three
# maximally-spread candidates, nothing merging -- still narrows: 3 brackets of
# 2*span/16 is 0.375*span, a 2.7x shrink. On the parent's real traces the
# candidates are usually clustered and the union comes out near 0.18*span.
# Not tuned: across D/8..D/32 the predicted joint hit is flat at 32.6-33.7%,
# so the mechanism is the extra level, not this fraction.
NARROW_FRAC = 0.0625
# Levels of pointing after the coarse one. One is the minimal falsifiable test
# of the claim -- does adding a single narrowing rung raise the hit rate? -- and
# the projection saturates past it (joint hit is flat at 32.6-33.7% for every
# geometry tried), while each extra rung compounds the risk of narrowing away
# from the answer and costs ~11s of decode per question. The loop below is
# written for any depth, so a later iteration can raise this if the rung pays.
MAX_REFINE_LEVELS = 1
# Thumbnails a refinement level re-decodes over the kept union. ``None`` means
# "as many as the coarse rung used", so each level spends the SAME number of
# thumbnails on a strictly shorter span -- which makes the resolution gain equal
# to the narrowing ratio and needs no separate constant. This is the pointer's
# input pool, not something shown at answer time.
REFINE_POOL: int | None = None
# Stop refining once the union is this short: the commit burst (three windows of
# ~1.5 pool steps each) already resolves a region that small, so another
# pointing call buys nothing.
MIN_UNION_S = 90.0
# A level must let the pointer see at least this much finer than the level
# above, or it is not worth a call. This is the real invariant of the schedule:
# narrowing exists to buy resolution, so a level that does not buy any is
# refused rather than run.
MIN_RES_GAIN = 2.0

REFINE_PROMPT = (
    "You are given {n} grids of small video thumbnails, in order. Each thumbnail "
    "is labeled with its elapsed time in the video as [m:ss].\n"
    "These come from a SHORTER part of the video that was already judged "
    "promising, so they are spaced much more finely than before -- you are now "
    "looking at {span} of footage out of {dur} total.\n"
    "Some stretches of the video are not shown here at all; that is expected.\n\n"
    "Question that must eventually be answered: {question}\n{options}\n\n"
    "Do not answer the question yet. Your only job is to say WHEN to look more "
    "closely, more precisely than before.\n"
    "Name the {maxr} most promising moments whose surroundings are most likely "
    "to contain the answer. Prefer moments where the thing the question asks "
    "about is visible or about to happen. Only name moments you can actually "
    "see among these thumbnails.\n"
    "Reply with only timestamps in m:ss form, most promising first, separated by "
    "commas. Example: 12:30, 41:05\n"
)


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of possibly-overlapping intervals, chronological."""
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _brackets(
    centers: list[float], half: float, duration: float
) -> list[tuple[float, float]]:
    """Keep a window around every candidate -- hedges stay alive, merged."""
    hi_cap = duration if duration and duration > 0 else max(centers, default=0.0) + half
    spans = []
    for c in centers:
        lo = max(0.0, c - half)
        hi = min(hi_cap, c + half)
        if hi > lo:
            spans.append((lo, hi))
    return _merge(spans)


def _span_len(spans: list[tuple[float, float]]) -> float:
    return float(sum(hi - lo for lo, hi in spans))


@dataclass
class _Mem:
    frames: list[Frame]
    img_emb: Any
    duration: float | None
    video: VideoStream | None
    pool_step: float


class PointerLadder(VideoMemoryHarness):
    """Localisation as a resolution schedule: each level narrows the next."""

    FPS = 2.0
    MAX_INGEST_FRAMES = 320

    # -- ingest: the parent's, unchanged -----------------------------------
    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_INGEST_FRAMES)
        images = [
            f.image if f.image is not None else (f.caption or "") for f in frames
        ]
        img_emb = self.embed_images(images)
        vtok = sum(max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames)
        self.account_ingest(video.video_id, len(frames), vtok)
        ts = [float(f.timestamp) for f in frames]
        step = float(np.median(np.diff(ts))) if len(ts) > 1 else MIN_BURST_HALF_S
        return _Mem(
            frames=frames,
            img_emb=img_emb,
            duration=video.duration,
            video=video,
            pool_step=max(1.0, step),
        )

    # -- one rung of the ladder --------------------------------------------
    def _point(
        self,
        memory: _Mem,
        pool: list[Frame],
        question: str,
        options: list[str],
        *,
        prompt: str,
        span_s: float | None,
    ) -> list[float]:
        """Tile ``pool`` into labeled grids and ask the VLM when to look."""
        if not pool:
            return []
        per = GRID_COLS * GRID_ROWS
        max_grids = max(1, min(MAX_GRID_IMAGES, self.frame_budget()))
        # If the pool needs more grids than the request may hold, thin it
        # UNIFORMLY rather than letting the tail fall off the end. Truncating
        # would silently make the level blind to the later part of the span it
        # was given -- which for a refinement level is the region the rung above
        # chose, so the narrowing would be aimed and then half-discarded.
        room = max_grids * per
        if len(pool) > room:
            idx = np.linspace(0, len(pool) - 1, room).round().astype(int)
            pool = [pool[i] for i in sorted(set(int(j) for j in idx))]
        groups = [pool[i : i + per] for i in range(0, len(pool), per)]
        sheets: list[Frame] = []
        for grp in groups[:max_grids]:
            sheet = _grid_image(grp, 168)
            if sheet is None:
                continue
            sheets.append(
                Frame(
                    index=int(grp[0].index),
                    timestamp=float(grp[0].timestamp),
                    image=sheet,
                    size=sheet.size,
                    provenance={
                        "kind": "survey_grid",
                        "cols": GRID_COLS,
                        "times": [round(float(f.timestamp), 1) for f in grp],
                    },
                )
            )
        if not sheets:
            return []
        parts = self.render_frames(sheets)
        dur = memory.duration or (pool[-1].timestamp if pool else 0.0)
        parts.append(
            {
                "type": "text",
                "text": prompt.format(
                    n=len(sheets),
                    dur=_fmt_clock(dur),
                    span=_fmt_clock(span_s if span_s is not None else dur),
                    question=question,
                    options=format_options(options),
                    maxr=MAX_REGIONS,
                ),
            }
        )
        try:
            reply = self.ask_vlm(parts, max_tokens=48)
        except Exception:
            return []
        return parse_survey_times(reply, memory.duration)[:MAX_REGIONS]

    def _refine_pool(
        self, memory: _Mem, spans: list[tuple[float, float]], want: int
    ) -> list[Frame]:
        """Re-decode inside the kept union, proportionally to each span.

        These timepoints are mostly NOT in the ingest pool -- that is the point:
        the aliased windows (24.7% of them hold zero pool frames) can only
        become addressable if the pointer's input contains frames the uniform
        grid skipped.
        """
        if memory.video is None or not spans or want <= 0:
            return []
        total = _span_len(spans)
        if total <= 0:
            return []
        out: list[Frame] = []
        for lo, hi in spans:
            n = int(round(want * (hi - lo) / total))
            n = max(GRID_COLS, min(n, want))
            try:
                out.extend(memory.video.sample_time_range(lo, hi, n))
            except Exception:
                continue
        # Chronological and de-duplicated: adjacent spans can request the same
        # decoded index, and a repeated thumbnail wastes a grid cell.
        uniq: dict[int, Frame] = {}
        for f in out:
            uniq.setdefault(int(f.index), f)
        return [uniq[i] for i in sorted(uniq)]

    def _ladder(
        self, memory: _Mem, question: str, options: list[str]
    ) -> tuple[list[float], dict[str, Any]]:
        """Point coarsely, then re-point inside progressively shorter spans."""
        dur = float(memory.duration or (memory.frames[-1].timestamp if memory.frames else 0.0))
        info: dict[str, Any] = {}

        # L0: the parent's whole-pool survey.
        centers = self._point(
            memory,
            memory.frames,
            question,
            options,
            prompt=_L0_PROMPT,
            span_s=dur,
        )
        info["l0_centers"] = [round(c, 1) for c in centers]
        if not centers:
            return [], info

        span_now = dur
        step_now = memory.pool_step
        # Same thumbnail count as the coarse rung, on a shorter span: the
        # resolution gain is then the narrowing ratio itself.
        want = REFINE_POOL if REFINE_POOL else len(memory.frames)
        levels = 0
        for _ in range(MAX_REFINE_LEVELS):
            half = NARROW_FRAC * span_now
            spans = _brackets(centers, half, dur)
            union = _span_len(spans)
            if union <= MIN_UNION_S:
                info["stop"] = "union_short"
                break
            pool = self._refine_pool(memory, spans, want)
            if len(pool) < GRID_COLS * GRID_ROWS:
                info["stop"] = "pool_thin"
                break
            # The point of narrowing is resolution. Measure what this level
            # would actually let the pointer see, and refuse it if that is not
            # meaningfully finer than the level above -- otherwise the call
            # spends a request to re-ask the same question at the same scale.
            ts = sorted(float(f.timestamp) for f in pool)
            step_new = float(np.median(np.diff(ts))) if len(ts) > 1 else step_now
            if step_new <= 0 or step_now / step_new < MIN_RES_GAIN:
                info["stop"] = "no_res_gain"
                break
            finer = self._point(
                memory,
                pool,
                question,
                options,
                prompt=REFINE_PROMPT,
                span_s=union,
            )
            # A level that reads nothing usable leaves the rung above standing.
            inside = [t for t in finer if any(lo <= t <= hi for lo, hi in spans)]
            if not inside:
                info["stop"] = "nothing_inside"
                break
            centers = inside
            span_now = union
            step_now = step_new
            levels += 1
            info[f"l{levels}_centers"] = [round(c, 1) for c in centers]
            info[f"l{levels}_union_s"] = round(union, 1)
            info[f"l{levels}_step_s"] = round(step_new, 2)

        info["ladder_levels"] = levels
        info["final_span_s"] = round(span_now, 1)
        info["final_step_s"] = round(step_now, 2)
        return centers, info

    # -- commit: the parent's burst, aimed by the refined addresses --------
    def _burst(self, memory: _Mem, centers: list[float], want: int) -> list[Frame]:
        if want <= 0 or memory.video is None or not centers:
            return []
        half = max(MIN_BURST_HALF_S, POOL_STEP_MULT * memory.pool_step)
        dur = memory.duration or 0.0
        weights = [1.0 / (i + 1) for i in range(len(centers))]
        tot = sum(weights)
        out: list[Frame] = []
        for c, w in zip(centers, weights):
            n = max(2, int(round(want * w / tot)))
            lo = max(0.0, c - half)
            hi = c + half
            if dur > 0:
                hi = min(hi, dur)
            if hi <= lo:
                continue
            try:
                out.extend(memory.video.sample_time_range(lo, hi, n))
            except Exception:
                continue
        return out

    def answer_question(
        self, memory: _Mem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        k = self.frame_budget()
        n = len(memory.frames)
        if n == 0:
            return "?", {"error": "no frames", "strategy": AGENT_NAME}

        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        mat = np.asarray(memory.img_emb, dtype=np.float32)
        scores = mat @ qv.reshape(-1)

        # A stated clock is exact: it needs no ladder and outranks one.
        span = parse_stated_interval(question, memory.duration)
        info: dict[str, Any] = {}
        if span is not None:
            centers = [0.5 * (span[0] + span[1])]
            route = "stated"
        else:
            centers, info = self._ladder(memory, question, options)
            route = f"ladder{info.get('ladder_levels', 0)}" if centers else "parent"

        if not centers:
            chosen = _clamp([memory.frames[i] for i in aks_select(scores, k)], k)
            parts = self.render_frames(chosen)
            n_burst = 0
        else:
            want = max(1, int(round(k * BURST_SHARE)))
            if span is not None:
                lo, hi = span
                try:
                    burst = (
                        memory.video.sample_time_range(lo, hi, want)
                        if memory.video
                        else []
                    )
                except Exception:
                    burst = []
            else:
                burst = self._burst(memory, centers, want)

            merged: dict[int, Frame] = {int(f.index): f for f in burst}
            n_burst = len(merged)
            # Whole-video anchors: a fully mis-aimed ladder still shows what the
            # parent would have shown.
            anchor_k = max(1, k - n_burst)
            for i in aks_select(scores, anchor_k):
                merged.setdefault(int(memory.frames[i].index), memory.frames[i])

            chosen = _clamp([merged[i] for i in sorted(merged)], k)
            parts = self.render_frames(chosen, timestamps=True)

        parts.append(
            {
                "type": "text",
                "text": PAPER_INSTRUCT_PROMPT.format(
                    question=question, options=format_options(options)
                ),
            }
        )
        resp = self.ask_vlm(parts)
        letter = normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        )
        n_segments_est = int(
            np.clip(np.round(np.log2(max(n / max(k, 1), 1))), 0, ALL_DEPTH)
        )
        meta = {
            "strategy": AGENT_NAME,
            "sampled": len(chosen),
            "pool": n,
            "budget": k,
            "route": route,
            "burst_frames": n_burst,
            "pool_step_s": round(memory.pool_step, 1),
            "centers": [round(c, 1) for c in centers],
            "stated_interval": (
                [round(span[0], 1), round(span[1], 1)] if span else None
            ),
            "n_segments_est": n_segments_est,
            "score_span": float(np.max(scores) - np.min(scores)) if n else 0.0,
            "raw": (resp or "")[:200],
        }
        meta.update(info)
        return letter, meta


# The coarse rung is the parent's survey question, with the placeholders the
# shared `_point` helper fills. Kept here so both rungs read as one schedule.
_L0_PROMPT = (
    "You are given {n} grids of small video thumbnails, in order. Each thumbnail "
    "is labeled with its elapsed time in the video as [m:ss].\n"
    "The video lasts about {dur}.\n\n"
    "Question that must eventually be answered: {question}\n{options}\n\n"
    "Do not answer the question yet -- the thumbnails are too small. Your only "
    "job is to say WHEN to look more closely.\n"
    "Name the {maxr} most promising moments whose surroundings are most likely "
    "to contain the answer. Prefer moments where the thing the question asks "
    "about is visible or about to happen.\n"
    "Reply with only timestamps in m:ss form, most promising first, separated by "
    "commas. Example: 12:30, 41:05\n"
)
