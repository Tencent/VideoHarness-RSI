"""Matched VideoSEAL structural baseline (decoupled planner / inspector).

Paper: VideoSEAL (ICML 2026). This arm ports the planner--inspector split into
``VideoMemoryHarness`` under the same frozen Qwen3-VL-8B for both roles.

Declared deviations from the published VideoSEAL system:

1. No GRPO: the planner is prompted, not a trained VideoSEAL checkpoint.
2. Planner and inspector share Qwen3-VL-8B-Instruct.
3. Clip-index captions come from the same 8B; no subtitle OCR tool.
4. Max rollout steps default to 8 (paper allows up to 16).
5. Each VisualInspect call is capped at ``frame_budget()`` frames (per-call K).
   Cumulative frames across inspects may exceed K; Table 1 reports that cost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..harness import (
    VideoMemoryHarness,
    extract_json_field,
    format_options,
    normalize_choice,
)
from ..video import Frame, VideoStream

AGENT_NAME = "videoseal_matched"
CLIP_SECONDS = 16.0
MAX_STEPS = 8
DEFAULT_TOP_K = 4
INSPECT_CAP = 64

PLANNER_PROMPT = (
    "You plan evidence search for a long-video multiple-choice question.\n"
    "Actions (reply with ONE JSON object only):\n"
    '  {{"action":"VisualRetrieve","query":"<text>","top_k":{top_k}}}\n'
    '  {{"action":"VisualInspect","t0":<sec>,"t1":<sec>}}\n'
    '  {{"action":"Stop"}}\n'
    "Prefer Retrieve to locate candidate clips, then Inspect a short window.\n"
    "Stop only when the inspector has already answered, or budget is exhausted.\n\n"
    "Question: {question}\n"
    "Options:\n{options}\n\n"
    "Search memory:\n{history}\n\n"
    "Remaining visual frame budget: {remaining}\n"
)

INSPECT_PROMPT = (
    "You are the inspector. Decide if the frames below are sufficient to answer.\n"
    "Question: {question}\n"
    "Options:\n{options}\n\n"
    "Window: {t0:.1f}s–{t1:.1f}s. Prior planner note: {note}\n\n"
    "Reply with ONE JSON object only:\n"
    '{{"sufficient":0|1,"feedback":"<what is missing or confirmed>",'
    '"final_answer":"<letter or empty>"}}\n'
    "Set sufficient=1 only if the frames support a confident letter choice."
)


@dataclass
class _Clip:
    t0: float
    t1: float
    caption: str
    frame_idxs: list[int] = field(default_factory=list)


@dataclass
class _Mem:
    video: VideoStream
    frames: list[Frame]
    timestamps: list[float]
    clips: list[_Clip]
    clip_emb: Any
    duration: float
    video_id: str = ""


def _parse_planner(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text or "")
    data = None
    if m:
        try:
            raw = json.loads(m.group(0))
            if isinstance(raw, dict):
                data = raw
        except (json.JSONDecodeError, TypeError, AttributeError):
            data = None
    if data is None:
        up = (text or "").upper()
        if "STOP" in up:
            return {"action": "Stop"}
        if "INSPECT" in up:
            nums = re.findall(r"(\d+(?:\.\d+)?)", text or "")
            t0 = float(nums[0]) if nums else 0.0
            t1 = float(nums[1]) if len(nums) > 1 else t0 + CLIP_SECONDS
            return {"action": "VisualInspect", "t0": t0, "t1": t1}
        return {
            "action": "VisualRetrieve",
            "query": extract_json_field(text or "", "query", default="key evidence")
            or "key evidence",
            "top_k": DEFAULT_TOP_K,
        }
    action = str(data.get("action") or "").strip()
    up = action.upper()
    if up in ("STOP",) or action.lower() == "stop":
        return {"action": "Stop"}
    if "INSPECT" in up:
        nums = re.findall(r"(\d+(?:\.\d+)?)", text or "")
        t0 = float(data.get("t0", nums[0] if nums else 0.0) or 0.0)
        t1 = float(
            data.get("t1", nums[1] if len(nums) > 1 else t0 + CLIP_SECONDS)
            or (t0 + CLIP_SECONDS)
        )
        return {"action": "VisualInspect", "t0": t0, "t1": t1}
    query = str(data.get("query") or "key evidence")
    top_k = int(data.get("top_k") or DEFAULT_TOP_K)
    return {"action": "VisualRetrieve", "query": query, "top_k": top_k}


def _parse_inspect(text: str, options: list[str]) -> dict[str, Any]:
    out = {"sufficient": 0, "feedback": (text or "")[:240], "final_answer": ""}
    m = re.search(r"\{[\s\S]*\}", text or "")
    data = None
    if m:
        try:
            raw = json.loads(m.group(0))
            if isinstance(raw, dict):
                data = raw
        except (json.JSONDecodeError, TypeError, AttributeError):
            data = None
    if data is not None:
        s = data.get("sufficient", 0)
        if isinstance(s, str):
            sufficient = 1 if s.strip() in {"1", "true", "True"} else 0
        else:
            sufficient = 1 if int(s or 0) else 0
        feedback = str(data.get("feedback") or "").strip()
        letter = normalize_choice(str(data.get("final_answer") or ""), options)
        return {
            "sufficient": sufficient,
            "feedback": feedback or out["feedback"],
            "final_answer": "" if letter == "?" else letter,
        }
    letter = normalize_choice(text or "", options)
    if letter != "?":
        out["final_answer"] = letter
    return out


class VideoSEALMatched(VideoMemoryHarness):
    """Decoupled planner–inspector under the frozen-VLM interface."""

    FPS = 2.0
    MAX_INGEST_FRAMES = 320

    def build_memory(self, video: VideoStream) -> Any:
        vid = str(
            getattr(video, "video_id", None)
            or getattr(video, "path", None)
            or "unknown"
        )
        frames = self.sample_ingest_frames(video, max_frames=self.MAX_INGEST_FRAMES)
        if not frames:
            return _Mem(video, [], [], [], None, 0.0, vid)

        timestamps = [float(f.timestamp) for f in frames]
        duration = float(video.duration or (timestamps[-1] if timestamps else 0.0))

        clips: list[_Clip] = []
        reps: list[Frame] = []
        t = 0.0
        while t < max(duration, CLIP_SECONDS) - 1e-6:
            t1 = t + CLIP_SECONDS
            idxs = [
                i
                for i, ts in enumerate(timestamps)
                if (t - 1e-6) <= ts < (t1 + 1e-6)
            ]
            if idxs:
                mid = idxs[len(idxs) // 2]
                clips.append(
                    _Clip(
                        t0=t,
                        t1=min(t1, duration),
                        caption="",
                        frame_idxs=idxs,
                    )
                )
                reps.append(frames[mid])
            t = t1
            if len(clips) > 500:
                break

        captions = self.caption_frames(reps) if reps else []
        for c, cap in zip(clips, captions):
            c.caption = cap

        if clips:
            texts = [
                c.caption if c.caption else f"clip {c.t0:.0f}-{c.t1:.0f}"
                for c in clips
            ]
            clip_emb = np.asarray(self.embed_texts(texts), dtype=np.float32)
        else:
            clip_emb = None

        mem = _Mem(
            video=video,
            frames=frames,
            timestamps=timestamps,
            clips=clips,
            clip_emb=clip_emb,
            duration=duration,
            video_id=vid,
        )
        self.account_ingest(vid, len(frames), 0)
        return mem

    def _retrieve(self, memory: _Mem, query: str, top_k: int) -> list[_Clip]:
        if not memory.clips:
            return []
        k = max(1, min(int(top_k or DEFAULT_TOP_K), len(memory.clips)))
        if memory.clip_emb is None:
            return memory.clips[:k]
        qv = np.asarray(
            self.embed_texts([query or "important evidence"]), dtype=np.float32
        )[0]
        idxs = self.topk_indices(qv, memory.clip_emb, k)
        return [memory.clips[i] for i in idxs]

    def _inspect_frames(
        self, memory: _Mem, t0: float, t1: float, k: int
    ) -> list[Frame]:
        if k <= 0:
            return []
        pool = [
            f
            for f in memory.frames
            if (t0 - 1e-6) <= float(f.timestamp) <= (t1 + 1e-6)
        ]
        if not pool and memory.video is not None:
            try:
                pool = memory.video.sample_time_range(t0, t1, max(1, k))
            except Exception:
                pool = []
        if not pool:
            return []
        if len(pool) <= k:
            return pool
        step = max(1, len(pool) // k)
        return pool[::step][:k]

    def answer_question(
        self, memory: _Mem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        if not memory.frames:
            return "?", {"error": "no frames", "strategy": AGENT_NAME}

        history: list[str] = []
        final_letter = "?"
        steps = 0
        inspected_once = False
        opts = format_options(options)
        raw = ""
        per_call = min(INSPECT_CAP, self.frame_budget())

        for step in range(MAX_STEPS):
            steps = step + 1
            try:
                raw = self.ask_vlm(
                    [
                        {
                            "type": "text",
                            "text": PLANNER_PROMPT.format(
                                top_k=DEFAULT_TOP_K,
                                question=question,
                                options=opts,
                                history="\n".join(history) if history else "(empty)",
                                remaining=per_call,
                            ),
                        }
                    ]
                )
            except Exception as exc:
                history.append(f"planner_error: {exc}")
                break

            act = _parse_planner(raw or "")
            name = str(act.get("action") or "").lower()
            if name in ("stop",) or str(act.get("action")) == "Stop":
                history.append("Stop")
                break

            if str(act.get("action")) in (
                "VisualRetrieve",
                "Retrieve",
                "retrieve",
            ) or name in ("visualretrieve", "retrieve"):
                query = str(act.get("query") or "key evidence")
                top_k = int(act.get("top_k") or DEFAULT_TOP_K)
                hits = self._retrieve(memory, query, top_k)
                if hits:
                    lines = [f"[{c.t0:.0f}-{c.t1:.0f}s] {c.caption}" for c in hits]
                    history.append(f"Retrieve({query}) ->\n" + "\n".join(lines))
                else:
                    history.append(f"Retrieve({query}) ->\n(none)")
                continue

            t0 = float(act.get("t0") or 0.0)
            t1 = float(act.get("t1") or (t0 + CLIP_SECONDS))
            k = per_call
            frames = self._inspect_frames(memory, t0, t1, k)[:k]
            if not frames:
                history.append(f"Inspect({t0:.1f}-{t1:.1f}) empty")
                continue

            note = (history[-1] if history else "")[:200]
            try:
                parts = [
                    {
                        "type": "text",
                        "text": INSPECT_PROMPT.format(
                            question=question,
                            options=opts,
                            t0=t0,
                            t1=t1,
                            note=note,
                        ),
                    }
                ]
                parts.extend(self.render_frames(frames))
                insp_raw = self.ask_vlm(parts)
            except Exception as exc:
                history.append(f"inspect_error: {exc}")
                break

            inspected_once = True
            verdict = _parse_inspect(insp_raw or "", options)
            history.append(
                f"Inspect({t0:.1f}-{t1:.1f}) sufficient={verdict['sufficient']}"
                f" feedback={verdict.get('feedback') or ''}"
            )
            if int(verdict.get("sufficient") or 0) == 1:
                letter = verdict.get("final_answer") or "?"
                if letter and letter != "?":
                    final_letter = letter
                    break

        if final_letter == "?":
            k = per_call
            t0, t1 = 0.0, float(memory.duration or 0.0)
            frames = (
                self._inspect_frames(memory, t0, t1, k) or memory.frames
            )[:k]
            if frames:
                prompt = (
                    "Answer the multiple-choice question from these frames.\n"
                    f"Question: {question}\nOptions:\n{opts}\n"
                    "Reply with the letter only."
                )
                parts = [{"type": "text", "text": prompt}]
                parts.extend(self.render_frames(frames))
                try:
                    resp = self.ask_vlm(parts)
                    letter = normalize_choice(
                        extract_json_field(resp or "", "final_answer", default="")
                        or (resp or ""),
                        options,
                    )
                    if letter != "?":
                        final_letter = letter
                    history.append(f"forced_final_inspect frames={len(frames)}")
                    raw = resp
                except Exception as exc:
                    history.append(f"inspect_error: {exc}")

        return final_letter, {
            "strategy": AGENT_NAME,
            "steps": steps,
            "num_frames_shown": self.answer_frames_used(),
            "budget": self.frame_budget(),
            "history_preview": "\n".join(history)[:800],
            "raw": (raw or "")[:200] if raw else None,
        }
