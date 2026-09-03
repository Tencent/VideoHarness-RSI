"""Two-request look: survey the whole pool cheaply, then commit budget densely.

Parent: ``agents/stated_interval_router.py`` (frontier F_t), whose selector and
stated-clock route are both retained.

Every harness evaluated in this run so far -- the parent included -- issues
exactly ONE VLM request holding exactly ``frame_budget()`` images. They differ
only in how they *order or reshuffle* those slots: by CLIP similarity, by a
clock parsed from the question, by timestamp annotation, by occurrence
enumeration. None of them varied the **shape of the look**, and that shape is
what binds.

The parent's own answer traces say why. With 40 slots spread over a median
54-minute video the shown frames sit ~70s apart, while the spans that actually
contain answers are far shorter (p25 9s, median 25s): two thirds of them are
narrower than one step of the grid. So when the parent does land on the right
moment it lands *thinly*, and thin evidence is measurably worth nothing --
questions where 1-5 of the 40 shown frames fall inside the answer's span score
no better than questions where none do, while questions with 6 or more score
sharply higher. The gain is a step, not a ramp: evidence has to clear a density
threshold before the model can use it at all.

The parent cannot cross that threshold, and the reason is structural rather
than a matter of ranking. Its interval tier is carved out of the *same* k slots
as its skeleton (``skeleton_k = k - len(interval)``), so every frame committed
to a region deletes a frame of global coverage. One flat budget has to fund two
incompatible jobs -- covering the video and resolving a moment -- and
whichever wins, the other starves.

Mechanism (axis G answering x axis F packing x axis D router, as a *request
graph* rather than a selection rule): give the two jobs two requests.

1. **Survey.** The ingest pool already holds 320 uniformly-spaced frames --
   ~10s apart, seven times finer than what the parent shows -- and the parent
   throws that resolution away before the model ever sees it. Tile the whole
   pool into a handful of labeled thumbnail grids and ask the frozen VLM, in
   one cheap call, *when* the question is answered. No extra decoding: these
   frames are already in memory. The localiser is the VLM reading pixels over
   all 320 timepoints, not a similarity score over 40 -- and this run has
   already falsified score-aimed zoom, while text-aimed zoom only fires on the
   minority of questions that literally state a clock.
2. **Commit.** Re-decode a tight burst around each named timestamp at spacing
   *below* the pool step, so the burst is dense enough to clear the threshold
   instead of sampling across it. Burst half-width follows the measured pool
   step rather than a tuned constant, since the survey can only be as precise
   as the grid it read.
3. **Anchor.** Keep a minority share of the budget on the parent's own AKS
   selection spanning the whole video, so a survey that aims badly degrades to
   roughly the parent rather than to nothing, and questions whose evidence is
   genuinely diffuse keep the global spread the parent already answers with.

Guards, because the failure mode of committing budget is aiming it wrongly: the
survey may name up to three regions (hedged, not all-or-nothing); a stated
clock in the question still overrides the survey, since that address is exact
and already-proven capability; and an empty, unparseable, or out-of-duration
survey reply falls back to the parent's exact path -- same pool, same selector,
same prompt, no timestamps. The answer prompt is the parent's verbatim, so the
measured difference is the shape of the look and not prompt wording.
"""

from __future__ import annotations

import re
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

AGENT_NAME = "survey_then_commit"

# ---- survey pass -----------------------------------------------------------
# Thumbnails per grid image. The grid must stay legible enough to read a scene
# from, while keeping the survey to a handful of images.
GRID_COLS = 4
GRID_ROWS = 3
GRID_CELL = 168  # px, long side of each thumbnail inside the grid
# Hard ceiling on grid images in the survey request, well inside the per-request
# frame window.
MAX_GRID_IMAGES = 30
# Regions the survey is allowed to name. More than one hedges a bad aim.
MAX_REGIONS = 3

# ---- commit pass -----------------------------------------------------------
# Share of the answer budget spent as dense bursts at the surveyed addresses.
# The remainder stays on whole-video AKS anchors, so a mis-aimed burst degrades
# toward the parent instead of toward nothing.
BURST_SHARE = 0.6
# Burst half-width is derived from the pool spacing (the survey's own
# resolution), never below this floor.
MIN_BURST_HALF_S = 6.0
POOL_STEP_MULT = 1.5

SURVEY_PROMPT = (
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

PAPER_INSTRUCT_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    "Question: {question}\n{options}"
)

_TS = re.compile(r"(?<!\d)(\d{1,2}):([0-5]\d)(?::([0-5]\d))?(?!\d)")


def _fmt_clock(t: float) -> str:
    """Elapsed seconds as m:ss (the notation the survey is asked to reply in)."""
    t = max(0.0, float(t))
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def parse_survey_times(reply: str, duration: float | None) -> list[float]:
    """Timestamps the survey named, in reply order, de-duplicated."""
    out: list[float] = []
    for m in _TS.finditer(reply or ""):
        a, b, c = m.groups()
        secs = (
            int(a) * 3600 + int(b) * 60 + int(c)
            if c is not None
            else int(a) * 60 + int(b)
        )
        t = float(secs)
        if duration and duration > 0 and t > duration:
            continue  # a clock inside the scene, not an offset into the file
        if all(abs(t - u) > 1e-6 for u in out):
            out.append(t)
    return out


def _clamp(frames: list[Frame], k: int) -> list[Frame]:
    """First ``k`` frames, chronological.

    Deliberately not ``take_answer_frames``: that helper clamps to the budget
    *remaining across the whole question*, which the survey pass has already
    spent. The hard constraint ``render_frames`` enforces is a per-REQUEST
    window, and the answer request is entitled to the full window regardless of
    what the survey looked at -- otherwise the survey starves the very burst it
    exists to aim, which is the opposite of the point.
    """
    return sorted(frames, key=lambda f: float(f.timestamp))[: max(0, int(k))]


def _grid_image(frames: list[Frame], cell: int):
    """Tile frames into one labeled contact sheet; None if PIL is unavailable."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    thumbs = []
    for fr in frames:
        img = fr.image
        if img is None:
            continue
        try:
            from ..vlm import as_pil_image

            img = as_pil_image(img)
        except Exception:
            pass
        try:
            img = img.convert("RGB").copy()
            img.thumbnail((cell, cell))
        except Exception:
            continue
        thumbs.append((fr, img))
    if not thumbs:
        return None
    band = 13  # label strip under each thumbnail
    cw = max(t[1].size[0] for t in thumbs)
    ch = max(t[1].size[1] for t in thumbs) + band
    cols = min(GRID_COLS, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for i, (fr, img) in enumerate(thumbs):
        x = (i % cols) * cw
        y = (i // cols) * ch
        sheet.paste(img, (x, y))
        draw.text((x + 2, y + img.size[1] + 1), f"[{_fmt_clock(fr.timestamp)}]", fill=(255, 235, 90))
    return sheet


@dataclass
class _Mem:
    frames: list[Frame]
    img_emb: Any
    duration: float | None
    video: VideoStream | None
    pool_step: float


class SurveyThenCommit(VideoMemoryHarness):
    """Cheap whole-pool survey names the moment; a dense burst answers it."""

    FPS = 2.0
    MAX_INGEST_FRAMES = 320

    def build_memory(self, video: VideoStream) -> Any:
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_INGEST_FRAMES)
        images = [
            f.image if f.image is not None else (f.caption or "") for f in frames
        ]
        img_emb = self.embed_images(images)
        vtok = sum(
            max(1, f.size[0] // 28) * max(1, f.size[1] // 28) for f in frames
        )
        self.account_ingest(video.video_id, len(frames), vtok)
        # Median spacing of the pool: the finest resolution the survey can
        # resolve, and therefore the scale the commit burst must beat.
        ts = [float(f.timestamp) for f in frames]
        step = float(np.median(np.diff(ts))) if len(ts) > 1 else MIN_BURST_HALF_S
        # The stream is retained so the commit pass can decode between pool
        # points. `_close_memory` releases the reader on eviction and it
        # re-opens lazily, so this holds no fd hostage.
        return _Mem(
            frames=frames,
            img_emb=img_emb,
            duration=video.duration,
            video=video,
            pool_step=max(1.0, step),
        )

    # -- survey ------------------------------------------------------------
    def _survey(
        self, memory: _Mem, question: str, options: list[str]
    ) -> tuple[list[float], dict[str, Any]]:
        """Show the whole pool as thumbnail grids; return the times it names."""
        pool = memory.frames
        if not pool:
            return [], {"grids": 0}
        per = GRID_COLS * GRID_ROWS
        groups = [pool[i : i + per] for i in range(0, len(pool), per)]
        # The survey is itself one request, so it obeys the same per-request
        # window; under a small budget it shows fewer, denser grids.
        max_grids = max(1, min(MAX_GRID_IMAGES, self.frame_budget()))
        sheets = []
        for grp in groups[:max_grids]:
            sheet = _grid_image(grp, GRID_CELL)
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
            return [], {"grids": 0}
        parts = self.render_frames(sheets)
        dur = memory.duration or (pool[-1].timestamp if pool else 0.0)
        parts.append(
            {
                "type": "text",
                "text": SURVEY_PROMPT.format(
                    n=len(sheets),
                    dur=_fmt_clock(dur),
                    question=question,
                    options=format_options(options),
                    maxr=MAX_REGIONS,
                ),
            }
        )
        try:
            reply = self.ask_vlm(parts, max_tokens=48)
        except Exception:
            return [], {"grids": len(sheets), "survey_error": True}
        times = parse_survey_times(reply, memory.duration)[:MAX_REGIONS]
        return times, {
            "grids": len(sheets),
            "survey_raw": (reply or "")[:120],
            "survey_times": [round(t, 1) for t in times],
        }

    # -- commit ------------------------------------------------------------
    def _burst(self, memory: _Mem, centers: list[float], want: int) -> list[Frame]:
        """Decode ``want`` frames split across tight windows at ``centers``."""
        if want <= 0 or memory.video is None or not centers:
            return []
        half = max(MIN_BURST_HALF_S, POOL_STEP_MULT * memory.pool_step)
        dur = memory.duration or 0.0
        # Front-load the budget: the survey ranked its answers, so the first
        # named moment earns the most frames.
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
                continue  # decode is best-effort; anchors still answer
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

        # A stated clock is exact; it needs no survey and outranks one.
        span = parse_stated_interval(question, memory.duration)
        info: dict[str, Any] = {}
        if span is not None:
            centers = [0.5 * (span[0] + span[1])]
            route = "stated"
        else:
            centers, info = self._survey(memory, question, options)
            route = "survey" if centers else "parent"

        if not centers:
            # Nothing aimed us anywhere: the parent's exact path.
            chosen = _clamp(
                [memory.frames[i] for i in aks_select(scores, k)], k
            )
            parts = self.render_frames(chosen)
            n_burst = 0
        else:
            want = max(1, int(round(k * BURST_SHARE)))
            if span is not None:
                # The named span bounds the look directly, at pool-beating
                # spacing; no burst widening needed.
                lo, hi = span
                try:
                    burst = memory.video.sample_time_range(lo, hi, want) if memory.video else []
                except Exception:
                    burst = []
            else:
                burst = self._burst(memory, centers, want)

            merged: dict[int, Frame] = {int(f.index): f for f in burst}
            n_burst = len(merged)
            # Anchors keep whole-video reach so a bad aim degrades to ~parent.
            anchor_k = max(1, k - n_burst)
            for i in aks_select(scores, anchor_k):
                merged.setdefault(int(memory.frames[i].index), memory.frames[i])

            chosen = [merged[i] for i in sorted(merged)]
            chosen = _clamp(chosen, k)
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
            "burst_half_s": round(max(MIN_BURST_HALF_S, POOL_STEP_MULT * memory.pool_step), 1),
            "pool_step_s": round(memory.pool_step, 1),
            "centers": [round(c, 1) for c in centers],
            "stated_interval": ([round(span[0], 1), round(span[1], 1)] if span else None),
            "n_segments_est": n_segments_est,
            "score_span": float(np.max(scores) - np.min(scores)) if n else 0.0,
            "raw": (resp or "")[:200],
        }
        meta.update(info)
        return letter, meta
