"""Abstract interface for VL-Harness video-memory harnesses.

Counterpart of the text example's ``memory_system.MemorySystem``. Two structural
differences from the text version:

1. **Ingest once, answer many.** A long video is ingested a single time into a
   (harness-defined) multimodal memory; many questions are then answered against
   it. To reuse the outer loop / benchmark / proposer machinery unchanged, the
   inner-loop unit is still one ``(input, target)`` example, where ``input`` is a
   JSON string encoding ``{episode_id, video_ref, question, options}`` and
   ``target`` is the correct option letter. ``predict`` parses it, ingests the
   video (cached per video id, thread-safe), then answers.

2. **Visual-token cost is the Pareto currency.** Every image shown to the VLM
   costs visual tokens; the base class accounts them per answer via ``ask_vlm``
   and exposes them through ``get_last_prompt_info`` so the loop can build an
   accuracy-vs-visual-token frontier instead of accuracy-vs-chars.

Subclasses implement ``build_memory(video)`` and ``answer_question(memory, ...)``.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .video import Frame, VideoStream, _resize_long_side, load_video
from .vlm import ContentParts, VLMCallable, frame_token_cost

_LETTERS = [chr(ord("A") + i) for i in range(26)]


def extract_json_field(text: str, field: str, default: str = "") -> str:
    """Extract a field from (possibly messy / truncated) JSON in a VLM response."""
    if not text:
        return default
    try:
        data = json.loads(text)
        if isinstance(data, dict) and field in data:
            return str(data.get(field, default))
    except (json.JSONDecodeError, TypeError):
        pass
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and field in data:
                return str(data.get(field, default))
        except json.JSONDecodeError:
            pass
    # Prefer the last quoted value for this field (works on truncated JSON).
    m = re.findall(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text)
    if m:
        return m[-1]
    # Unquoted single letter, e.g. "final_answer": A
    m2 = re.findall(rf'"{re.escape(field)}"\s*:\s*([A-Za-z])\b', text)
    return m2[-1] if m2 else default


_CAPTION_NUM_RE = re.compile(r"^\s*(\d{1,3})\s*[.):]\s*(.+)$")
_CAPTION_FRAME_RE = re.compile(r"^\s*frame\s+(\d{1,3})\s*[:.]\s*(.+)$", re.IGNORECASE)


def parse_time_reference(ref: Any) -> tuple[float, float] | None:
    """'MM:SS-MM:SS' (minutes may exceed 99) or 'HH:MM:SS-...' -> seconds.

    LVBench ships an evidence window with every question. It is the only thing
    in this pipeline that says WHERE the answer actually is, which makes it the
    ground truth for asking whether a harness's frame selection had any chance
    of mattering. Returns None when the annotation is absent or malformed.
    """

    def _clock(s: str) -> float | None:
        s = (s or "").strip()
        if not s or s.lower() == "none":
            return None
        parts = s.split(":")
        if not all(re.fullmatch(r"\d+", p) for p in parts):
            return None
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return None

    ref = str(ref or "")
    if "-" not in ref:
        return None
    lo, _, hi = ref.partition("-")
    t0, t1 = _clock(lo), _clock(hi)
    if t0 is None:
        return None
    if t1 is None or t1 < t0:
        t1 = t0
    return (t0, t1)


def _parse_numbered_captions(text: str, n: int) -> list[str] | None:
    """Split a '<i>. caption' reply into exactly n captions, else None.

    Tolerates a chatty preamble, a 'Frame i:' numbering style, and captions that
    wrap onto a following line. Returning None (rather than a padded or
    truncated list) is deliberate: a caller that cannot trust the alignment must
    re-caption frame by frame, because a silently shifted caption list corrupts
    every downstream retrieval.
    """
    if not text or n <= 0:
        return None
    text = re.sub(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", " ", text, flags=re.I)
    found: dict[int, str] = {}
    last: int | None = None
    for line in text.splitlines():
        m = _CAPTION_NUM_RE.match(line) or _CAPTION_FRAME_RE.match(line)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n and idx not in found:
                found[idx] = m.group(2).strip()
                last = idx
            else:
                last = None
        elif last is not None and line.strip():
            found[last] += " " + line.strip()
    if len(found) != n:
        return None
    return [found[i] for i in range(1, n + 1)]


def _downscale_for_caption(image: Any, side: int) -> Any:
    """Shrink a frame for the write-time caption pass only.

    The answer-time copy is untouched, so this changes captioning throughput
    without changing the visual budget a harness spends at answer time.
    """
    if image is None or not side:
        return image
    try:
        return _resize_long_side(image, side)
    except Exception:
        return image


def normalize_choice(pred: str, options: list[str]) -> str:
    """Map a free-form VLM answer to an option letter (A/B/C/...).

    Handles: bare letter, "A.", "(B)", "final_answer"-style short strings,
    Thinking-model tails (last line is the letter), or substring match against
    option text. Does NOT scan long reasoning for incidental letters like "N"
    from "None".
    """
    n = max(1, len(options))
    allowed = set(_LETTERS[:n])
    if not pred:
        return "?"
    # Strip common think-blocks from Thinking models.
    pred = re.sub(r"<think>[\s\S]*?</think>", " ", pred, flags=re.IGNORECASE)
    pred = pred.strip()

    def _letter(s: str) -> str | None:
        m = re.match(r"^\(?\s*([A-Za-z])\s*[\)\.:]?\s*$", s.strip())
        if m and m.group(1).upper() in allowed:
            return m.group(1).upper()
        return None

    # Short answers / last non-empty line (Thinking models often end with "B").
    if len(pred) <= 8:
        hit = _letter(pred)
        if hit:
            return hit
    lines = [ln.strip() for ln in pred.splitlines() if ln.strip()]
    for ln in reversed(lines[-5:]):
        hit = _letter(ln)
        if hit:
            return hit
        # "The answer is B" / "final_answer: C" on a late line
        m = re.search(
            r"(?:final[_\s-]?answer|answer)\s*(?:is|=|:)?\s*\(?([A-Za-z])\b",
            ln,
            flags=re.IGNORECASE,
        )
        if m and m.group(1).upper() in allowed:
            return m.group(1).upper()

    m = re.search(
        r"(?:final[_\s-]?answer|answer)\s*(?:is|=|:)?\s*\(?([A-Za-z])\b",
        pred,
        flags=re.IGNORECASE,
    )
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()

    # Option-text match: prefer the *last* occurrence in the response.
    low = pred.lower()
    best_i, best_pos = None, -1
    for i, opt in enumerate(options):
        body = re.sub(r"^[A-Za-z][\.\):]\s*", "", opt).strip().lower()
        if len(body) < 3:
            continue
        pos = low.rfind(body)
        if pos > best_pos:
            best_pos, best_i = pos, i
    if best_i is not None:
        return _LETTERS[best_i]

    m2 = re.match(r"^\(?\s*([A-Za-z])\s*[\)\.:\-]", pred)
    if m2 and m2.group(1).upper() in allowed:
        return m2.group(1).upper()
    return "?"


def format_options(options: list[str]) -> str:
    """Render options as 'A. ...\\nB. ...' whether or not they carry letters."""
    lines = []
    for i, opt in enumerate(options):
        if re.match(r"^[A-Za-z][\.\):]\s*", opt):
            lines.append(opt)
        else:
            lines.append(f"{_LETTERS[i]}. {opt}")
    return "\n".join(lines)


class VideoMemoryHarness(ABC):
    """Base class for video-memory harnesses.

    Args:
        vlm: multimodal callable (see ``vlm.VLMCallable``).
        embedder: optional ``MultimodalEmbedder`` for retrieval harnesses.
        target_side: longer-side px used for visual-token accounting of frames.
    """

    def __init__(
        self,
        vlm: VLMCallable,
        embedder: Any | None = None,
        target_side: int = 224,
    ):
        self._vlm = vlm
        self._embedder = embedder
        self.target_side = target_side
        self._memories: dict[str, Any] = {}
        self._ingest_cost: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._global_lock = threading.Lock()
        self._local = threading.local()
        self._state = "{}"

    # ── subclass API ────────────────────────────────────────────────────
    @abstractmethod
    def build_memory(self, video: VideoStream) -> Any:
        """Ingest a video into a (harness-defined) memory object.

        Called once per unique video. May sample frames, caption, embed, build
        hierarchical/graph structures, etc. Return any object; it is passed back
        to ``answer_question``. Track write-time visual work via ``account_ingest``.
        """

    @abstractmethod
    def answer_question(
        self, memory: Any, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        """Answer one MCQ against a prebuilt memory. Return (letter, metadata).

        Use ``self.ask_vlm(parts)`` for every VLM call so visual tokens are
        accounted. Use ``normalize_choice(pred, options)`` to coerce the letter.
        """

    # ── frame sampling + visual-token accounting ─────────────────────────
    DEFAULT_INGEST_FPS = 2.0
    DEFAULT_MAX_INGEST_FRAMES = 320

    # Visual budget for one sweep arm, set from config ``video.frame_budget``
    # (None = the historical 320-frame default). Changing this one number is what
    # turns the 7 search axes from inert (everything fits, so nothing has to be
    # selected) into live. It binds READ time, not WRITE time:
    #   - render_frames always refuses more than the budget, so no harness can
    #     show the VLM more than K frames in one request;
    #   - sample_ingest_frames clamps to the budget at ANSWER time, so a uniform
    #     baseline becomes uniform-K with no code change, but is left alone
    #     inside build_memory, where a retrieval harness must still be free to
    #     embed/caption a large candidate pool it will later select K frames from.
    # Constraining the candidate pool too would make retrieval impossible and the
    # sweep would compare uniform-K against uniform-K.
    FRAME_BUDGET: int | None = None

    def frame_budget(self) -> int:
        """Effective per-request frame ceiling: the sweep budget, else the cap.

        A set budget REPLACES ``MAX_REQUEST_FRAMES`` rather than being min'd with
        it. Taking the minimum silently capped every arm at 320 and made budgets
        above that impossible, which is exactly the regime the vendor's own
        protocol lives in (224K visual tokens is ~1009 frames at 222 tok/frame).
        """
        budget = getattr(self, "FRAME_BUDGET", None)
        if budget:
            return int(budget)
        return int(getattr(self, "MAX_REQUEST_FRAMES", 320) or 320)

    def answer_frames_used(self) -> int:
        """Frames already shown to the VLM in the current ``predict`` call."""
        return int(getattr(self._local, "last_num_frames", 0) or 0)

    def answer_frames_remaining(self) -> int:
        """Matched-budget leftover for multi-turn inspect / retrieve loops."""
        return max(0, self.frame_budget() - self.answer_frames_used())

    def take_answer_frames(self, frames: list, k: int | None = None) -> list:
        """Return up to ``k`` (default: remaining budget) frames, chronological."""
        rem = self.answer_frames_remaining()
        if k is None:
            k = rem
        k = max(0, min(int(k), rem, len(frames)))
        if k <= 0:
            return []
        return list(frames[:k])

    def sample_ingest_frames(
        self,
        video: VideoStream,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> list[Frame]:
        """Sample the standard ingest pool at 2 fps, capped at 320 frames.

        New harnesses should use this instead of fixed 32/64-frame uniform pools.
        A specialized harness may pass explicit values only when its hypothesis
        requires a different ingestion policy; answer-time top-k remains free.

        At answer time an explicit ``max_frames`` is still clamped to
        ``frame_budget()``, so a sweep arm binds every harness including ones
        that hardcode 320. Inside ``build_memory`` it is not: that pool is the
        write-time candidate set, which the budget does not govern.
        """
        video.num_available()
        rate = fps if fps is not None else getattr(
            self, "FPS", self.DEFAULT_INGEST_FPS
        )
        cap = max_frames if max_frames is not None else getattr(
            self, "MAX_INGEST_FRAMES", self.DEFAULT_MAX_INGEST_FRAMES
        )
        if not getattr(self._local, "in_build_memory", False):
            budget = getattr(self, "FRAME_BUDGET", None)
            # A sweep budget overrides the harness's own constant in both
            # directions: arms must be comparable at K, whether K is below the
            # harness's default or above it.
            cap = int(budget) if budget else min(int(cap), self.frame_budget())
        if video.duration and video.duration > 0:
            target = int(round(video.duration * rate))
        else:
            target = video.num_available()
        return video.sample_uniform(max(1, min(target, int(cap))))

    # Hard invariant: a single VLM request may include at most this many frames.
    # This is a FRAMEWORK-LEVEL cap, not a suggestion: any harness that tries to
    # push more frames into one request raises. Long videos must therefore be
    # ingested over MULTIPLE passes (adaptive sampling + memory), which is the
    # mechanism we want the evolution to discover -- rather than a single uniform
    # dump that misses key frames. How many passes and how many frames per pass
    # (<= this cap) is decided by the candidate harness / inner loop, not fixed.
    MAX_REQUEST_FRAMES = 320

    def render_frames(
        self,
        frames: list[Frame],
        tokens_per_frame: int | None = None,
        *,
        timestamps: bool = False,
    ) -> ContentParts:
        """Turn frames into VLM content parts AND account their visual cost.

        Works for real frames (emits image parts) and mock frames (emits a text
        placeholder carrying the caption so ``StubVLM`` can read ANSWER_HINTs).
        In both cases the per-frame visual-token cost is added to the running
        answer cost, so the Pareto x-axis is consistent across backends.

        Enforces the hard per-request frame cap (``MAX_REQUEST_FRAMES``): sending
        more than the cap into a single VLM call raises, forcing multi-pass
        ingestion + memory instead of one oversized uniform dump.

        If ``timestamps=True``, each real frame is preceded by a ``[12.3s]``
        text part. Default is off so agents that already annotate time do not
        double-stamp.
        """
        cap = self.frame_budget()
        if len(frames) > cap:
            raise RuntimeError(
                f"[{type(self).__name__}] single VLM request exceeds the hard "
                f"frame cap: {len(frames)} > {cap}. Split the video into multiple "
                f"passes (each <= {cap} frames) and aggregate into memory."
            )
        parts: ContentParts = []
        added_frames = 0
        added_tokens = 0
        for fr in frames:
            cost = (
                tokens_per_frame
                if tokens_per_frame is not None
                else frame_token_cost(fr.size[0], fr.size[1])
            )
            if timestamps:
                parts.append({"type": "text", "text": f"[{fr.timestamp:.1f}s]"})
            if fr.raw_png is not None:
                # Already-encoded PNG: hand it over as bytes so the client does
                # not decode and re-compress every frame on the way out.
                parts.append({"type": "image", "image": fr.raw_png, "size": fr.size})
            elif fr.image is not None:
                parts.append({"type": "image", "image": fr.image, "size": fr.size})
            else:
                parts.append(
                    {
                        "type": "text",
                        "text": f"[frame@{fr.timestamp:.1f}s] {fr.caption or ''}",
                    }
                )
            added_frames += 1
            added_tokens += cost
        # Timestamps of everything actually shown, so the loop can score whether
        # the answer's evidence window was ever on screen -- the quantity that
        # bounds what any frame-selection axis can win.
        times = getattr(self._local, "last_frame_times", None)
        if times is None:
            times = []
            self._local.last_frame_times = times
        times.extend(fr.timestamp for fr in frames)
        self._local.last_num_frames = (
            getattr(self._local, "last_num_frames", 0) or 0
        ) + added_frames
        self._local.last_visual_tokens = (
            getattr(self._local, "last_visual_tokens", 0) or 0
        ) + added_tokens
        return parts

    def ask_vlm(self, parts: ContentParts) -> str:
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        self._local.last_prompt_text = text
        self._local.last_prompt_len = len(text)
        self._local.last_prompt_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        resp = self._vlm(parts)
        # If the backend reports REAL visual tokens for this call, accumulate them
        # (Pareto currency). StubVLM has no such method -> we keep the estimate.
        getter = getattr(self._vlm, "pop_last_visual_tokens", None)
        if getter is not None:
            rv = getter()
            if rv is not None:
                self._local.last_real_visual_tokens = (
                    getattr(self._local, "last_real_visual_tokens", 0) or 0
                ) + int(rv)
                self._local.has_real_visual_tokens = True
        return resp
    

    def caption_frame(self, frame: Frame, prompt: str | None = None) -> str:
        """Return a caption for a frame WITHOUT charging answer-time visual cost.

        Mock frames carry a preset caption. Real frames are captioned by the VLM
        (a write-time cost; record it via ``account_ingest`` from build_memory).
        """
        if frame.caption is not None:
            return frame.caption
        parts = [
            {
                "type": "text",
                "text": prompt or "Describe this video frame in one factual sentence.",
            },
            {"type": "image", "image": frame.image, "size": frame.size},
        ]
        resp = self._vlm(parts)
        return extract_json_field(resp, "final_answer") or resp

    # Captioning a 320-frame ingest pool one request per frame dominates
    # build_memory wall clock. Batching keeps EVERY frame captioned -- fidelity
    # matters for the published-system replicas -- while cutting request count
    # by CAPTION_BATCH_SIZE. The short per-frame token budget also stops vLLM
    # from reserving KV cache for a generation length a caption never uses.
    CAPTION_BATCH_SIZE = 8
    CAPTION_TOKENS_PER_FRAME = 64
    CAPTION_TARGET_SIDE = 336
    # Deliberately 1: build_memory already runs inside the inner loop's
    # question-level thread pool, so any concurrency here MULTIPLIES with it and
    # oversubscribes the VLM service (measured: 8 caption workers made a 32-frame
    # ingest 2x slower, 16 workers 3x slower, purely from queueing).
    CAPTION_MAX_WORKERS = 1

    def caption_frames(
        self,
        frames: list[Frame],
        *,
        prompt: str | None = None,
        batch_size: int | None = None,
        tokens_per_frame: int | None = None,
        target_side: int | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        """Caption many frames with batched VLM requests (write-time cost).

        Semantically equivalent to ``[self.caption_frame(f) for f in frames]``:
        one caption per input frame, same order. Any batch whose reply cannot be
        parsed into exactly one caption per frame falls back to per-frame
        captioning, so a malformed response costs latency rather than silently
        corrupting memory with misaligned captions.
        """
        bs = batch_size or getattr(self, "CAPTION_BATCH_SIZE", 8)
        tpf = tokens_per_frame or getattr(self, "CAPTION_TOKENS_PER_FRAME", 64)
        side = (
            target_side
            if target_side is not None
            else getattr(self, "CAPTION_TARGET_SIDE", 336)
        )
        workers = max_workers or getattr(self, "CAPTION_MAX_WORKERS", 4)
        base_prompt = prompt or "Describe this video frame in one factual sentence."

        captions: list[str | None] = [f.caption for f in frames]
        todo = [i for i, c in enumerate(captions) if c is None]
        if not todo:
            return [c or "" for c in captions]
        if bs <= 1:
            batches = [[i] for i in todo]
        else:
            batches = [todo[i : i + bs] for i in range(0, len(todo), bs)]

        def run_batch(idxs: list[int]) -> tuple[list[int], list[str]]:
            if len(idxs) == 1:
                return idxs, [self.caption_frame(frames[idxs[0]], base_prompt)]
            k = len(idxs)
            header = (
                f"You are given {k} video frames in chronological order. "
                f"{base_prompt} Describe each frame separately, using only what "
                f"is visible in that frame.\n"
                f"Output ONLY a numbered list of exactly {k} lines:\n"
                f"1. <caption for frame 1>\n...\n{k}. <caption for frame {k}>\n"
                f"No preamble, no headers, no explanation."
            )
            parts: ContentParts = [{"type": "text", "text": header}]
            for n, i in enumerate(idxs, start=1):
                fr = frames[i]
                parts.append({"type": "text", "text": f"Frame {n}:"})
                parts.append(
                    {
                        "type": "image",
                        "image": _downscale_for_caption(fr.image, side),
                        "size": fr.size,
                    }
                )
            # Budget generously: a truncated reply fails to parse and costs k
            # single-frame retries, which is far more expensive than the tokens.
            resp = self._caption_vlm_call(parts, max(768, tpf * k + 128))
            parsed = _parse_numbered_captions(resp, len(idxs))
            if parsed is None:
                return idxs, [self.caption_frame(frames[i], base_prompt) for i in idxs]
            return idxs, parsed

        if workers > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as exe:
                done = list(exe.map(run_batch, batches))
        else:
            done = [run_batch(b) for b in batches]

        for idxs, caps in done:
            for i, cap in zip(idxs, caps):
                captions[i] = cap
        return [c or "" for c in captions]

    def _caption_vlm_call(self, parts: ContentParts, max_tokens: int) -> str:
        """VLM call for captioning: no thinking, bounded output, no answer cost.

        StubVLM and other simple backends accept ``parts`` only, so the kwargs
        are dropped on TypeError rather than requiring every backend to match
        the real client's signature.
        """
        try:
            return self._vlm(parts, enable_thinking=False, max_tokens=max_tokens)
        except TypeError:
            return self._vlm(parts)

    # -- embedding / retrieval helpers -----------------------------------
    def embed_texts(self, texts: list[str]):
        return self._embedder.embed_text(texts)

    def embed_images(self, images: list) -> Any:
        return self._embedder.embed_image(images)

    @staticmethod
    def topk_indices(query_vec, matrix, k: int) -> list[int]:
        """Indices of the top-k rows in `matrix` by cosine sim to query_vec.

        Assumes L2-normalized rows (MultimodalEmbedder normalizes), so a plain
        dot product ranks by cosine similarity.
        """
        import numpy as np

        if matrix is None or len(matrix) == 0:
            return []
        sims = np.asarray(matrix) @ np.asarray(query_vec).reshape(-1)
        k = max(1, min(k, len(sims)))
        return list(np.argsort(-sims)[:k])

    def account_ingest(self, video_id: str, num_frames: int, visual_tokens: int):
        """Record write-time (offline preprocessing) visual work for a video."""
        with self._global_lock:
            self._ingest_cost[video_id] = {
                "num_frames": num_frames,
                "visual_tokens": visual_tokens,
            }

    def _reset_answer_cost(self):
        self._local.last_visual_tokens = 0
        self._local.last_num_frames = 0
        self._local.last_prompt_len = 0
        self._local.last_prompt_hash = None
        self._local.last_prompt_text = None
        self._local.last_real_visual_tokens = 0
        self._local.has_real_visual_tokens = False
        self._local.last_frame_times = []
    
    # ── (input, target) contract used by inner_loop ─────────────────────
    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        self._reset_answer_cost()
        ep = json.loads(input)
        # Identity only, never the target: a harness that needs to look itself up
        # in benchmark metadata (the oracle arm) can, without widening the
        # answer_question signature for everyone else.
        self._local.episode_id = ep.get("episode_id")
        memory = self._get_memory(ep)
        letter, meta = self.answer_question(
            memory, ep["question"], ep["options"]
        )
        meta = dict(meta or {})
        meta["parse_fail"] = (letter == "?")
        est = getattr(self._local, "last_visual_tokens", 0)
        if getattr(self._local, "has_real_visual_tokens", False):
            vt = getattr(self._local, "last_real_visual_tokens", 0)
            meta["visual_tokens_estimate"] = est
        else:
            vt = est
        meta["visual_tokens"] = vt
        meta["num_frames"] = getattr(self._local, "last_num_frames", 0)
        return letter, meta
    
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        """Default: no online learning. Homer-style cross-question skill
        accumulation is an *option* a proposer may implement by overriding this.
        """
        return

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state

    # ── per-video ingest cache ──────────────────────────────────────────
    def _cache_key(self, ep: dict[str, Any]) -> str:
        ref = ep.get("video_ref")
        if isinstance(ref, dict):
            return str(ref.get("video_id", ep.get("episode_id", "?")))
        if isinstance(ref, str) and ref.strip().startswith("{"):
            try:
                return str(json.loads(ref).get("video_id", ref[:32]))
            except json.JSONDecodeError:
                return ref[:64]
        return str(ref)

    def _get_memory(self, ep: dict[str, Any]) -> Any:
        key = self._cache_key(ep)
        if key in self._memories:
            return self._memories[key]
        with self._locks[key]:
            if key in self._memories:
                return self._memories[key]
            video = load_video(ep["video_ref"], target_side=self.target_side)
            self._local.in_build_memory = True
            try:
                memory = self.build_memory(video)
            finally:
                self._local.in_build_memory = False
            self._memories[key] = memory
            return memory

    # ── introspection used by inner_loop ────────────────────────────────
    def get_last_prompt_info(self) -> dict[str, Any]:
        est = getattr(self._local, "last_visual_tokens", 0)
        if getattr(self._local, "has_real_visual_tokens", False):
            vt = getattr(self._local, "last_real_visual_tokens", 0)
        else:
            vt = est
        return {
            "prompt_len": getattr(self._local, "last_prompt_len", None),
            "prompt_hash": getattr(self._local, "last_prompt_hash", None),
            "prompt_text": getattr(self._local, "last_prompt_text", None),
            "visual_tokens": vt,
            "visual_tokens_estimate": est,
            "num_frames": getattr(self._local, "last_num_frames", 0),
            "frame_times": list(getattr(self._local, "last_frame_times", []) or []),
        }
    
    def get_visual_cost(self) -> dict[str, Any]:
        """Aggregate write-time ingest cost across all ingested videos."""
        total_frames = sum(c["num_frames"] for c in self._ingest_cost.values())
        total_tokens = sum(c["visual_tokens"] for c in self._ingest_cost.values())
        n = max(1, len(self._ingest_cost))
        return {
            "ingest_videos": len(self._ingest_cost),
            "ingest_frames_avg": total_frames / n,
            "ingest_visual_tokens_avg": total_tokens / n,
        }
