"""Video decoding / frame-sampling utilities for VL-Harness.

Two backends behind one ``VideoStream`` abstraction:

- **mock**: frames are pre-defined descriptors (timestamp + caption + optional
  ANSWER_HINT). No heavy deps; used to exercise the full pipeline offline with
  ``StubVLM``. A "needle" frame carries the ground-truth answer marker so a
  harness that *retrieves the right moment* scores correctly.
- **real**: an mp4 path decoded lazily via ``decord`` (preferred) or OpenCV.
  Frames are resized so the longer side == ``target_side`` px, which fixes the
  per-frame visual-token cost for equal-budget comparisons.

Sampling helpers (uniform, and a light shot-boundary detector) are shared.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Real video decoding (decord/OpenCV+ffmpeg) is not reliably thread-safe: running
# concurrent decodes from the inner-loop's ThreadPoolExecutor can trip a libav
# frame-threading assertion. Serialize all real decode behind one lock and force
# single-threaded ffmpeg. The slow part (the VLM call) still runs concurrently.
_DECODE_LOCK = threading.RLock()
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")


@dataclass
class Frame:
    """A single sampled frame."""

    index: int
    timestamp: float
    image: Any = None  # PIL.Image | bytes | path | None (mock text-only)
    caption: str | None = None  # preset caption (mock) or filled by a captioner
    size: tuple[int, int] = (224, 224)  # (w, h) for visual-token accounting
    # PNG bytes when they are already on hand. _image_to_data_url sends PNG for
    # frames built by Image.fromarray, so carrying the encoded form lets the
    # answer path skip a per-frame PNG compression -- 1009 of them per question
    # at the report budget, which profiling showed to be the dominant cost.
    raw_png: Any = None
    # Optional derivation record for synthetic images (survey grids, etc.).
    # Plain decoded frames leave this unset. Does not change pixels or scoring.
    provenance: dict[str, Any] | None = None

    def to_image_part(self, tokens: int | None = None) -> dict[str, Any]:
        """Render as a vlm content 'image' part (token cost from size)."""
        part: dict[str, Any] = {"type": "image", "size": self.size}
        if self.image is not None:
            part["image"] = self.image
        else:
            # Mock frame with no pixels: represent it to the VLM as a textual
            # stand-in so StubVLM can still read any ANSWER_HINT it carries.
            part = {
                "type": "text",
                "text": f"[frame@{self.timestamp:.1f}s] {self.caption or ''}",
            }
        if tokens is not None:
            part["tokens"] = tokens
        return part


@dataclass
class VideoStream:
    """A long video, decoded lazily. Frames indexed by sample position."""

    video_id: str
    duration: float = 0.0
    fps: float = 1.0
    target_side: int = 224
    _frames: list[Frame] | None = None  # eager (mock)
    _path: str | None = None  # real mp4
    meta: dict[str, Any] = field(default_factory=dict)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_mock(cls, spec: dict[str, Any]) -> "VideoStream":
        """Build a mock video from a spec dict.

        spec = {
          "video_id": str, "duration": float,
          "frames": [{"t": float, "caption": str, "hint": "B"?}, ...]
        }
        """
        frames = []
        for i, f in enumerate(spec.get("frames", [])):
            cap = f.get("caption", "")
            if f.get("hint"):
                cap = f"{cap}  ANSWER_HINT: {f['hint']}"
            frames.append(
                Frame(
                    index=i,
                    timestamp=float(f.get("t", i)),
                    image=None,
                    caption=cap,
                    size=tuple(f.get("size", (224, 224))),
                )
            )
        return cls(
            video_id=spec["video_id"],
            duration=float(spec.get("duration", len(frames))),
            fps=float(spec.get("fps", 1.0)),
            _frames=frames,
            meta=spec.get("meta", {}),
        )

    @classmethod
    def from_path(cls, path: str, target_side: int = 224) -> "VideoStream":
        return cls(video_id=Path(path).stem, _path=path, target_side=target_side)

    # -- frame access ---------------------------------------------------------
    def _ensure_real_reader(self):
        if getattr(self, "_reader", None) is not None:
            return self._reader
        with _DECODE_LOCK:
            if getattr(self, "_reader", None) is not None:
                return self._reader
            try:
                import decord  # lazy

                decord.bridge.set_bridge("native")
                self._reader = decord.VideoReader(self._path)
                self._backend = "decord"
                self.fps = float(self._reader.get_avg_fps() or 1.0)
                self._n = len(self._reader)
                self.duration = self._n / self.fps if self.fps else 0.0
            except Exception:
                import cv2  # lazy fallback

                cv2.setNumThreads(1)
                self._reader = cv2.VideoCapture(self._path)
                self._backend = "cv2"
                self.fps = float(self._reader.get(cv2.CAP_PROP_FPS) or 1.0)
                self._n = int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.duration = self._n / self.fps if self.fps else 0.0
        return self._reader

    def _frame_cache_dir(self):
        root = os.environ.get("VL_HARNESS_DATA", "~/.cache/vl-harness/data")
        return (
            Path(os.path.expanduser(root))
            / ".frame_cache"
            / Path(self._path or "novideo").stem
            / str(self.target_side)
        )

    def _read_real_indices(self, indices: list[int]) -> list[Frame]:
        """Decode frames, reusing a PNG cache on disk.

        At the report's budget a single question needs ~1000 frames, and decode
        is serialised behind a global lock with single-threaded ffmpeg, so the
        bottleneck moves off the GPU entirely: arms that sample the same indices
        (uniform, oracle, and the prompt variants all do) would each pay the full
        decode again. PNG is what ``_image_to_data_url`` already sends for frames
        built by ``Image.fromarray``, so caching the encoded bytes is
        byte-identical downstream and does not invalidate the VLM response cache.
        """
        from PIL import Image  # lazy

        cdir = self._frame_cache_dir()
        cached: dict[int, Frame] = {}
        missing: list[int] = []
        for idx in indices:
            p = cdir / f"{idx}.png"
            try:
                if p.exists():
                    import io

                    data = p.read_bytes()
                    img = Image.open(io.BytesIO(data))  # lazy: .size from header
                    cached[idx] = Frame(
                        index=idx,
                        timestamp=idx / self.fps if self.fps else float(idx),
                        image=img,
                        size=img.size,
                        raw_png=data,
                    )
                    continue
            except Exception:
                pass
            missing.append(idx)

        if missing:
            with _DECODE_LOCK:
                fresh = self._read_real_indices_locked(missing)
            try:
                cdir.mkdir(parents=True, exist_ok=True)
                for fr in fresh:
                    tmp = cdir / f".{fr.index}.tmp.png"
                    fr.image.save(tmp, format="PNG")
                    dst = cdir / f"{fr.index}.png"
                    tmp.replace(dst)
                    fr.raw_png = dst.read_bytes()
            except Exception:
                pass  # cache is an optimisation; never fail the run over it
            for fr in fresh:
                cached[fr.index] = fr

        return [cached[i] for i in indices if i in cached]

    def _read_real_indices_locked(self, indices: list[int]) -> list[Frame]:
        from PIL import Image  # lazy

        reader = self._ensure_real_reader()
        frames: list[Frame] = []
        if self._backend == "decord":
            try:
                batch = reader.get_batch(indices).asnumpy()
            except Exception:
                # decord's threaded decoder fails outright on some MVBench clips
                # ("Error sending packet"), which took that whole benchmark to
                # zero results. Fall back to the OpenCV path for this video
                # rather than losing it.
                import cv2

                cv2.setNumThreads(1)
                self._reader = cv2.VideoCapture(self._path)
                self._backend = "cv2"
                self.fps = float(self._reader.get(cv2.CAP_PROP_FPS) or 1.0)
                self._n = int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                return self._read_real_indices_locked(indices)
            for pos, idx in enumerate(indices):
                img = Image.fromarray(batch[pos])
                img = _resize_long_side(img, self.target_side)
                frames.append(
                    Frame(
                        index=idx,
                        timestamp=idx / self.fps if self.fps else float(idx),
                        image=img,
                        size=img.size,
                    )
                )
        else:  # cv2
            import cv2

            for idx in indices:
                reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, bgr = reader.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = _resize_long_side(Image.fromarray(rgb), self.target_side)
                frames.append(
                    Frame(
                        index=idx,
                        timestamp=idx / self.fps if self.fps else float(idx),
                        image=img,
                        size=img.size,
                    )
                )
        return frames

    def num_available(self) -> int:
        if self._frames is not None:
            return len(self._frames)
        self._ensure_real_reader()
        return int(getattr(self, "_n", 0))

    def sample_uniform(self, k: int) -> list[Frame]:
        """Return k frames evenly spaced across the whole video."""
        n = self.num_available()
        if n == 0:
            return []
        k = max(1, min(k, n))
        if self._frames is not None:
            step = n / k
            idxs = sorted({int(i * step) for i in range(k)})
            return [self._frames[i] for i in idxs]
        step = n / k
        idxs = sorted({int(i * step) for i in range(k)})
        return self._read_real_indices(idxs)

    def sample_time_range(self, t_start: float, t_end: float, k: int) -> list[Frame]:
        """Return up to k frames evenly spaced inside [t_start, t_end] seconds.

        Needed by harnesses that know *when* to look rather than just how many
        frames to take -- e.g. the oracle arm, which reads a benchmark's
        ground-truth evidence window to upper-bound what any retriever could win.
        """
        n = self.num_available()
        if n == 0 or k <= 0:
            return []
        fps = self.fps or 1.0
        i0 = max(0, min(n - 1, int(round(t_start * fps))))
        i1 = max(i0, min(n - 1, int(round(t_end * fps))))
        span = i1 - i0
        if span == 0:
            idxs = [i0]
        else:
            k = max(1, min(k, span + 1))
            step = span / (k - 1) if k > 1 else span
            idxs = sorted({int(round(i0 + i * step)) for i in range(k)})
        if self._frames is not None:
            return [self._frames[i] for i in idxs if 0 <= i < n]
        return self._read_real_indices(idxs)

    def all_frames(self) -> list[Frame]:
        """All pre-sampled frames (mock) or a 1fps decode (real)."""
        if self._frames is not None:
            return list(self._frames)
        n = self.num_available()
        stride = max(1, int(self.fps))  # ~1 fps
        return self._read_real_indices(list(range(0, n, stride)))


def _resize_long_side(img, side: int):
    w, h = img.size
    if max(w, h) <= side:
        return img
    if w >= h:
        nw, nh = side, max(1, int(h * side / w))
    else:
        nw, nh = max(1, int(w * side / h)), side
    return img.resize((nw, nh))


def load_video(video_ref: str | dict[str, Any], target_side: int = 224) -> VideoStream:
    """Dispatch a video reference to a VideoStream.

    - dict or JSON string with a "frames" key -> mock video
    - path ending in a video extension           -> real decode
    """
    if isinstance(video_ref, dict):
        return VideoStream.from_mock(video_ref)
    if isinstance(video_ref, str) and video_ref.strip().startswith("{"):
        return VideoStream.from_mock(json.loads(video_ref))
    return VideoStream.from_path(str(video_ref), target_side=target_side)


# -- shot-boundary detection (light, optional) -------------------------------
def detect_shot_boundaries(frames: list[Frame], threshold: float = 0.5) -> list[int]:
    """Return indices where a shot change likely occurs.

    Uses mean-color distance between consecutive frames when pixels are
    available; for mock (no pixels) returns [] (uniform sampling is used).
    """
    try:
        import numpy as np
    except Exception:
        return []
    prev = None
    boundaries = []
    for i, fr in enumerate(frames):
        if fr.image is None:
            continue
        arr = np.asarray(fr.image).astype("float32")
        sig = arr.reshape(-1, arr.shape[-1]).mean(axis=0) if arr.ndim == 3 else arr.mean()
        if prev is not None:
            dist = float(np.abs(sig - prev).mean()) / 255.0
            if dist > threshold:
                boundaries.append(i)
        prev = sig
    return boundaries
