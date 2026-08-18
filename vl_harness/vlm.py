"""VLM abstraction for VL-Harness.

This is the multimodal counterpart of the text example's ``llm.py``. It reuses
the same gateway idea (an OpenAI-compatible ``api_base`` served locally by e.g.
``modelscope server`` or vLLM for Qwen3-VL), but the message payload is a list
of *content parts* that may mix text and images.

A "content part" is one of:
    {"type": "text",  "text": "..."}
    {"type": "image", "image": <PIL.Image | bytes(PNG/JPEG) | filesystem path>,
                       "tokens": <int optional, visual-token cost override>}

Three implementations are provided:
    - ``StubVLM``   : dependency-free, deterministic. Used for plumbing smoke
                      tests. It reads an ``ANSWER_HINT: X`` marker embedded in
                      the text parts (mock captions carry it) and echoes it.
    - ``VLM``       : real client. Lazily imports ``litellm`` and talks to an
                      OpenAI-compatible endpoint (local Qwen3-VL server).
    - ``make_local_vlm`` / ``make_stub_vlm`` : convenience constructors.

The embedder (``MultimodalEmbedder``) mirrors this split: a hashing stub for
offline plumbing and a real CLIP/SigLIP path behind lazy imports.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ── Visual-token cost model ──────────────────────────────────────────────
# Qwen3-VL bills a variable number of visual tokens per image depending on the
# resolution it is fed. For the Pareto x-axis we need a *deterministic* cost so
# harnesses can be compared at equal budget. We approximate: an image resized so
# its longer side is ``side`` px costs ~ (side/28)^2 tokens (28px = one ViT
# patch merged block, matching Qwen's patch accounting closely enough for
# ranking). Callers may override per-part via {"tokens": N}.
PATCH_PX = 28


def frame_token_cost(width: int, height: int) -> int:
    """Approximate visual-token cost of one image at the given pixel size."""
    w = max(1, width // PATCH_PX)
    h = max(1, height // PATCH_PX)
    return int(w * h)


ContentParts = list[dict[str, Any]]


@runtime_checkable
class VLMCallable(Protocol):
    """A VLM is a callable taking multimodal content parts -> response text."""

    def __call__(self, parts: ContentParts) -> str:
        ...


def _image_to_data_url(image: Any) -> str:
    """Encode a PIL image / raw bytes / filesystem path as a data: URL."""
    from pathlib import Path as _P
    if isinstance(image, (str, _P)):
        raw = open(image, "rb").read()
        mime = "image/png" if str(image).lower().endswith(".png") else "image/jpeg"
        return "data:" + mime + ";base64," + base64.b64encode(raw).decode()
    if isinstance(image, (bytes, bytearray)):
        return "data:image/png;base64," + base64.b64encode(bytes(image)).decode()
    # assume PIL.Image
    buf = io.BytesIO()
    fmt = getattr(image, "format", None) or "PNG"
    image.save(buf, format=fmt)
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return "data:" + mime + ";base64," + base64.b64encode(buf.getvalue()).decode()


_ANSWER_HINT_RE = re.compile(r"ANSWER_HINT:\s*([A-Za-z])", re.IGNORECASE)


def _text_of(parts: ContentParts) -> str:
    return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")


class StubVLM:
    """Deterministic, dependency-free VLM for offline plumbing tests.

    Reads an ``ANSWER_HINT: X`` marker from the text parts (mock captions
    carry it) and echoes that letter; otherwise returns ``default``. Also
    estimates visual tokens from any {"tokens": N} on image parts so the
    Pareto plumbing works without a real model.
    """

    def __init__(self, default: str = "A"):
        self.default = default
        self.total_calls = 0
        self._local = threading.local()

    def __call__(self, parts: ContentParts, **kwargs) -> str:
        self.total_calls += 1
        vt = 0
        for p in parts:
            if p.get("type") == "image":
                vt += int(p.get("tokens", 0) or 0)
        self._local.last_visual_tokens = vt or None
        text = _text_of(parts)
        m = _ANSWER_HINT_RE.search(text)
        letter = m.group(1).upper() if m else self.default
        return json.dumps({"final_answer": letter})

    def pop_last_visual_tokens(self):
        return getattr(self._local, "last_visual_tokens", None)

    def get_usage(self) -> dict[str, Any]:
        return {"model": "stub", "calls": self.total_calls}




# ---------------------------------------------------------------------------
# Real VLM (OpenAI-compatible endpoint, e.g. local Qwen3-VL server)
# ---------------------------------------------------------------------------
CACHE_DIR = Path(
    os.path.expanduser(
        os.environ.get("VL_HARNESS_VLM_CACHE_DIR", "~/.cache/vl-harness/vlm")
    )
)


class VLM:
    """VLM caller backed by litellm against an OpenAI-compatible endpoint.

    Handles multimodal messages, on-disk response caching, and retries. Costs
    are not billed for local endpoints; we track calls and *visual tokens*
    (the Pareto currency) separately from text tokens.
    """

    def __init__(
        self,
        model: str = "Qwen3-VL-4B-Instruct",
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        timeout: float = 1800.0,
        use_cache: bool = True,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.use_cache = use_cache
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_visual_tokens = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _normalized_model(self) -> str:
        known = ("openai/", "gemini/", "openrouter/", "anthropic/", "ollama/")
        if self.api_base and not self.model.startswith(known):
            return f"openai/{self.model}"
        return self.model

    def _to_openai_messages(self, parts: ContentParts) -> list[dict[str, Any]]:
        content = []
        for p in parts:
            if p.get("type") == "text":
                content.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "image":
                url = _image_to_data_url(p["image"])
                content.append({"type": "image_url", "image_url": {"url": url}})
        return [{"role": "user", "content": content}]

    def _cache_path(
        self,
        messages: list[dict[str, Any]],
        enable_thinking: bool | None = None,
        max_tokens: int | None = None,
        *,
        legacy: bool = False,
    ) -> Path:
        """Path for a cached response.

        ``api_base`` is deliberately NOT part of the key: the same weights served
        at a different address produce the same answer, and keying on it silently
        voided the whole cache whenever the endpoint moved. ``legacy=True``
        reproduces the old key so entries written before that fix stay readable.
        """
        payload = {
            "model": self._normalized_model(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "enable_thinking": self.enable_thinking if enable_thinking is None else enable_thinking,
            "messages": messages,
        }
        if legacy:
            payload["api_base"] = self.api_base
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        return CACHE_DIR / f"{digest}.json"

    def __call__(
        self,
        parts: ContentParts,
        enable_thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from litellm import completion as litellm_completion  # lazy

        self._local.last_visual_tokens = None
        messages = self._to_openai_messages(parts)

        # Per-call overrides for generation params (e.g. cheap, no-thinking
        # captioning during ingest, while answers keep thinking on).
        eff_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        eff_max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.use_cache:
            for legacy in (False, True):
                cpath = self._cache_path(
                    messages, eff_thinking, eff_max_tokens, legacy=legacy
                )
                if not cpath.exists():
                    continue
                try:
                    cached = json.loads(cpath.read_text())
                    cvt = cached.get("visual_tokens")
                    if cvt is not None:
                        self._local.last_visual_tokens = int(cvt)
                    return cached["content"]
                except (OSError, json.JSONDecodeError, KeyError):
                    pass

        call_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": eff_max_tokens,
            "timeout": self.timeout,
            "proxy": None,
            # Our local OpenAI-compatible Qwen service accepts this field directly.
            "enable_thinking": eff_thinking,
        }
        if self.api_base:
            call_kwargs["base_url"] = self.api_base
            call_kwargs["api_key"] = (
                self.api_key or os.environ.get("OPENAI_API_KEY") or "local"
            )
        elif self.api_key:
            call_kwargs["api_key"] = self.api_key

        resp = litellm_completion(
            model=self._normalized_model(), messages=messages, **call_kwargs
        )
        content = resp.choices[0].message.content or ""
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                for c in content
            )

        usage = getattr(resp, "usage", None)
        vt = None
        if usage is not None:
            vt = getattr(usage, "visual_tokens", None)
            if vt is None and isinstance(usage, dict):
                vt = usage.get("visual_tokens")
            if vt is None:
                extra = getattr(usage, "model_extra", None) or getattr(usage, "__dict__", None)
                if isinstance(extra, dict):
                    vt = extra.get("visual_tokens")
            vt = int(vt) if vt is not None else None
        self._local.last_visual_tokens = vt
        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.total_output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self.total_visual_tokens += int(vt or 0)

        if self.use_cache:
            try:
                # Must mirror the READ key: writing under the default params made
                # every per-call override (captioning) a permanent cache miss.
                self._cache_path(messages, eff_thinking, eff_max_tokens).write_text(
                    json.dumps({"content": content, "visual_tokens": vt})
                )
            except OSError:
                pass
        return content

    def pop_last_visual_tokens(self):
        """Real visual tokens for the most recent __call__ on THIS thread."""
        return getattr(self._local, "last_visual_tokens", None)

    def get_usage(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "visual_tokens": self.total_visual_tokens,
        }


# ---------------------------------------------------------------------------
# Multimodal embedder (for retrieval-based harnesses)
# ---------------------------------------------------------------------------
class MultimodalEmbedder:
    """Text + image embedder with stub / local-real / remote-real backends.

    - ``backend='stub'``: deterministic pseudo-embeddings (for plumbing only)
    - ``backend='real'`` + no ``api_base``: in-process sentence-transformers
    - ``backend='real'`` + ``api_base``: HTTP embedding service (recommended)
    """

    def __init__(
        self,
        backend: str = "stub",
        dim: int = 256,
        model: str | None = None,
        api_base: str | None = None,
        timeout: float = 60.0,
    ):
        self.backend = backend
        self.dim = dim
        self.model_name = model
        self.api_base = (api_base or "").rstrip("/") or None
        self.timeout = float(timeout)
        self._model = None
        self._client = None
        self._lock = threading.Lock()

    # -- stub implementation --------------------------------------------------
    def _hash_vec(self, text: str):
        import numpy as np

        vec = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = float((vec * vec).sum()) ** 0.5
        if norm > 0:
            vec /= norm
        return vec

    def embed_text(self, texts: list[str]):
        import numpy as np

        if self.backend == "stub":
            return np.stack([self._hash_vec(t) for t in texts]) if texts else np.zeros(
                (0, self.dim), dtype="float32"
            )
        if self.api_base:
            return self._remote_embed_text(texts)
        return self._real_embed_text(texts)

    def embed_image(self, images: list[Any]):
        import numpy as np

        if self.backend == "stub":
            # Hash a cheap fingerprint of each image (path/bytes) into a vector.
            fps = []
            for im in images:
                if isinstance(im, (str, Path)):
                    fps.append(str(im))
                elif isinstance(im, (bytes, bytearray)):
                    fps.append(hashlib.md5(bytes(im)).hexdigest())
                else:
                    fps.append(repr(im)[:64])
            return (
                np.stack([self._hash_vec(f) for f in fps])
                if fps
                else np.zeros((0, self.dim), dtype="float32")
            )
        if self.api_base:
            return self._remote_embed_image(images)
        return self._real_embed_image(images)

    # -- remote implementation (HTTP service) --------------------------------
    def _ensure_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import httpx  # lazy

                    self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _remote_embed_text(self, texts: list[str]):
        import numpy as np

        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        c = self._ensure_client()
        r = c.post(f"{self.api_base}/embed/text", json={"texts": texts})
        r.raise_for_status()
        data = r.json()
        emb = np.asarray(data.get("embeddings", []), dtype="float32")
        if emb.ndim != 2:
            raise ValueError("embed service returned invalid text embedding shape")
        return emb

    def _remote_embed_image(self, images: list[Any]):
        import numpy as np

        if not images:
            return np.zeros((0, self.dim), dtype="float32")
        payload = []
        for im in images:
            if isinstance(im, (str, Path)):
                payload.append({"type": "path", "data": str(im)})
            elif isinstance(im, (bytes, bytearray)):
                payload.append({
                    "type": "base64",
                    "data": base64.b64encode(bytes(im)).decode(),
                })
            else:
                payload.append({"type": "data_url", "data": _image_to_data_url(im)})

        c = self._ensure_client()
        r = c.post(f"{self.api_base}/embed/image", json={"images": payload})
        r.raise_for_status()
        data = r.json()
        emb = np.asarray(data.get("embeddings", []), dtype="float32")
        if emb.ndim != 2:
            raise ValueError("embed service returned invalid image embedding shape")
        return emb

    # -- local real implementation (lazy) ------------------------------------
    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer  # lazy

                    name = self.model_name or "clip-ViT-B-32"
                    self._model = SentenceTransformer(name)
        return self._model

    def _real_embed_text(self, texts: list[str]):
        m = self._ensure_model()
        return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def _real_embed_image(self, images: list[Any]):
        from PIL import Image  # lazy

        m = self._ensure_model()
        pil = [
            Image.open(im) if isinstance(im, (str, Path)) else im for im in images
        ]
        return m.encode(pil, normalize_embeddings=True, convert_to_numpy=True)

# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------
def make_local_vlm(
    model: str = os.environ.get("VL_MODEL", "Qwen3-VL-4B-Instruct"),
    host: str = os.environ.get("LOCAL_VLM_HOST", "localhost"),
    port: int = int(os.environ.get("LOCAL_VLM_PORT", "8000")),
    **kwargs,
) -> VLM:
    return VLM(model=model, api_base=f"http://{host}:{port}/v1", **kwargs)


def make_stub_vlm(default: str = "A") -> StubVLM:
    return StubVLM(default=default)
