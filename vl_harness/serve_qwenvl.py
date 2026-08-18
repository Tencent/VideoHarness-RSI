"""Minimal OpenAI-compatible server for a local Qwen3-VL model.

Exposes ``/v1/chat/completions`` (and ``/v1/models``) so the VL-Harness client
(``vlm.VLM`` via litellm with ``api_base``) can talk to a *frozen* local Qwen3-VL
exactly as it would to any OpenAI endpoint. Accepts multimodal messages whose
content is a list of ``{"type":"text"}`` / ``{"type":"image_url"}`` parts, where
image_url is a base64 data URL (what ``vlm.VLM`` sends).

Run:
    python -m vl_harness.serve_qwenvl --port 8000
    # or:  VL_MODEL_PATH=/path/to/model python -m vl_harness.serve_qwenvl

Notes:
- Defaults to the modelscope snapshot of Qwen3-VL-4B-Instruct.
- Auto-selects device (cuda > mps > cpu). On CPU this is slow; use small
  subsets / few frames. Responses are deterministic (do_sample=False).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import os
import re
import time
import uuid

MODEL_PATH_DEFAULT = os.path.expanduser(
    os.environ.get(
        "VL_MODEL_PATH",
        "~/.cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master",
    )
)

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


def _pick_device_dtype(device_arg: str):
    import torch

    if device_arg != "auto":
        dev = device_arg
    elif torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    default_dtype = "float32" if dev == "cpu" else "bfloat16"
    name = os.environ.get("VL_SERVE_DTYPE", default_dtype)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        name, torch.float32
    )
    return dev, dtype


class Engine:
    """Loads the model/processor once and runs multimodal generation."""

    def __init__(self, model_path: str, device: str = "auto"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_path = model_path
        self.device, self.dtype = _pick_device_dtype(device)
        print(f"[serve] loading {model_path} on {self.device} ({self.dtype})", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self.model_name = os.path.basename(model_path.rstrip("/")) or "qwen3-vl"
        print("[serve] model ready", flush=True)

    def _decode_image(self, url: str):
        from PIL import Image

        m = _DATA_URL_RE.match(url.strip())
        if not m:
            raise ValueError("only base64 data URLs are supported")
        try:
            raw = base64.b64decode(m.group("data"))
        except binascii.Error as e:
            raise ValueError(f"bad base64 image: {e}") from e
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _to_chat(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI messages to Qwen chat format (text + PIL images)."""
        chat = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                chat.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue
            parts = []
            for c in content:
                ctype = c.get("type")
                if ctype == "text":
                    parts.append({"type": "text", "text": c.get("text", "")})
                elif ctype == "image_url":
                    url = c["image_url"]
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    parts.append({"type": "image", "image": self._decode_image(url)})
            chat.append({"role": role, "content": parts})
        return chat

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> dict:
        torch = self.torch
        chat = self._to_chat(messages)
        inputs = self.processor.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        in_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        gen = out[0][in_len:]
        text = self.processor.decode(gen, skip_special_tokens=True).strip()
        return {"text": text, "prompt_tokens": in_len, "completion_tokens": int(gen.shape[-1])}


def build_app(engine: Engine):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="vl-harness-qwen3vl")

    @app.get("/v1/models")
    def list_models():
        return {"data": [{"id": engine.model_name, "object": "model"}], "object": "list"}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        messages = body.get("messages", [])
        max_new = int(body.get("max_tokens") or 512)
        try:
            result = engine.generate(messages, max_new_tokens=max_new)
        except Exception as e:  # surface a clean error to the client
            return JSONResponse(status_code=500, content={"error": {"message": str(e)}})
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", engine.model_name),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }

    return app


def main():
    ap = argparse.ArgumentParser(description="OpenAI-compatible Qwen3-VL server")
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LOCAL_VLM_PORT", "8000")))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    import uvicorn

    engine = Engine(args.model_path, device=args.device)
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
