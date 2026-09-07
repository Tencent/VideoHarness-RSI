"""LiteLLM-based proposer for meta-harness evolution loop.

Drop-in replacement for `claude_wrapper` in `meta_harness.py::propose_claude()`.
Uses any OpenAI-compatible endpoint via litellm; no Claude Code CLI required.

Design:
- Load SKILL.md, frontier_val.json, evolution_summary.jsonl, current F_t.
- Ask the model to output ONE new candidate as a JSON object containing:
    { "name": snake_case, "hypothesis": str, "axis": str,
      "base_system": str, "components": [str], "code": <full Python file> }
- Extract, write agents/<name>.py, and write pending_eval.json.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litellm import completion as litellm_completion

from .proposer_archive import render_proposer_archive

# ---- Config knobs (overridable via env) ------------------------------------
# Proposer talks to an Anthropic-compatible endpoint via litellm
# (custom_llm_provider, model id, and base URL are independent of the frozen
# inner-loop VLM in config.yaml).
DEFAULT_MODEL = os.environ.get("PROPOSER_MODEL", "claude-sonnet-4-6")
# Endpoint: PROPOSER_API_BASE, else ANTHROPIC_BASE_URL, else empty (SDK default).
DEFAULT_API_BASE = os.environ.get(
    "PROPOSER_API_BASE",
    os.environ.get("ANTHROPIC_BASE_URL", ""),
)
# Force the Anthropic protocol so litellm hits /v1/messages when a custom base is set.
DEFAULT_LLM_PROVIDER = os.environ.get("PROPOSER_LLM_PROVIDER", "anthropic")
DEFAULT_MAX_TOKENS = int(os.environ.get("PROPOSER_MAX_TOKENS", "8000"))
DEFAULT_TEMPERATURE = float(os.environ.get("PROPOSER_TEMPERATURE", "0.7"))
NUM_CANDIDATES = int(os.environ.get("PROPOSER_NUM_CANDIDATES", "1"))


@dataclass
class ProposerResult:
    """Mimics the subset of claude_wrapper.RunResult used by meta_harness.py."""

    exit_code: int
    stderr: str = ""
    duration_seconds: float = 0.0

    def show(self) -> None:
        print(f"  proposer completed in {self.duration_seconds:.1f}s", flush=True)


# ---- Context assembly ------------------------------------------------------
def _read_skill_md(evolve_dir: Path) -> str:
    skill = evolve_dir.parent / "skills" / "vl-harness" / "SKILL.md"
    if skill.exists():
        return skill.read_text()
    return ""


def _read_memory_system_iface(evolve_dir: Path) -> str:
    p = evolve_dir / "harness.py"
    if p.exists():
        body = p.read_text()
        return "## harness.py (base interface)\n```python\n" + body + "\n```"
    return ""


# ---- Prompt --------------------------------------------------------------
_SYSTEM_PROMPT = """You are a research engineer contributing ONE iteration of the VL-Harness \
evolution loop over video-memory-harness code for long-video multiple-choice QA.

Follow the SKILL.md rules provided in the user message strictly. In particular:

- One mechanism per candidate; do NOT just tune numbers.
- Do NOT mention dataset names or hard-code dataset-specific hints.
- The class MUST subclass `VideoMemoryHarness` and implement `build_memory(video)` \
and `answer_question(memory, question, options)`.
- Use `from ..harness import VideoMemoryHarness, extract_json_field, format_options, normalize_choice`.
- Use `self.render_frames(frames)` to show frames (this accounts visual tokens), \
`self.ask_vlm(parts)` for the answer call, and `normalize_choice(pred, options)` \
to coerce the option letter.
- Optimize accuracy at LOW visual-token cost (the Pareto currency).

Your response MUST be a single JSON object, with this exact schema:

{
  "name": "<snake_case_name>",
  "hypothesis": "<falsifiable claim>",
  "axis": "exploitation" | "exploration",
  "base_system": "<name of baseline you built on>",
  "components": ["tag1", "tag2"],
  "code": "<full Python source of agents/<name>.py>"
}

Do NOT wrap the JSON in markdown. Do NOT add any commentary before or after \
the JSON. The `code` field must contain a self-contained Python module."""


def _build_user_prompt(
    iteration: int,
    evolve_dir: Path,
    logs_dir: Path,
    pending_eval_path: Path,
) -> str:
    skill = _read_skill_md(evolve_dir)
    parent = "aks"
    frontier = logs_dir / "frontier_val.json"
    if frontier.exists():
        try:
            data = json.loads(frontier.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
        for key, val in data.items():
            if str(key).startswith("_"):
                continue
            if isinstance(val, dict) and val.get("best_system"):
                parent = str(val["best_system"])
                break
    archive = render_proposer_archive(
        logs_dir, evolve_dir / "agents", parent=parent
    )
    iface = _read_memory_system_iface(evolve_dir)

    return f"""# Iteration {iteration}

## SKILL.md
{skill}

## ARCHIVE
The pack below is historical harness source, scores, and traces.
The candidate you write must still copy F_t (`{parent}`).

{archive}

{iface}

## Your task
Design ONE new memory system that could plausibly improve the current frontier. \
Follow the JSON schema described in the system prompt. Copy agents/{parent}.py \
and splice. \
Make sure the code imports match the existing agents' style and can be imported \
with `from vl_harness.agents.<name> import *`.
Write pending_eval.json conceptually via the JSON you return (the outer loop writes it to `{pending_eval_path}`).
"""


# ---- Parsing --------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    """Robustly pull the top-level JSON object out of the model response."""
    raw = raw.strip()

    # 1. Whole string is JSON.
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # 2. Fenced code block.
    m = _JSON_FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced { ... } we can find.
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = raw[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break

    raise ValueError(
        "Could not parse JSON from proposer response. First 500 chars:\n"
        + raw[:500]
    )


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_").lower()
    if not name:
        raise ValueError("candidate name is empty after sanitization")
    return name


# ---- Public entry ---------------------------------------------------------
def run(
    task_prompt: str,
    iteration: int,
    evolve_dir: Path,
    logs_dir: Path,
    pending_eval_path: Path,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    api_key: str | None = None,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    num_candidates: int = NUM_CANDIDATES,
    timeout: float = 300.0,
) -> ProposerResult:
    """Call litellm to propose new memory systems.

    On success: writes agents/<name>.py file(s) and pending_eval.json.
    Returns a ProposerResult mimicking claude_wrapper.RunResult.
    """
    # Resolve api_key for the proposer endpoint:
    #   1. explicit kwarg
    #   2. PROPOSER_API_KEY
    #   3. ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY
    api_key = (
        api_key
        or os.environ.get("PROPOSER_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    )
    if not api_key:
        return ProposerResult(
            exit_code=2,
            stderr="No proposer API key resolved (ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY / PROPOSER_API_KEY).",
        )

    agents_dir = evolve_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    user_prompt = _build_user_prompt(
        iteration=iteration,
        evolve_dir=evolve_dir,
        logs_dir=logs_dir,
        pending_eval_path=pending_eval_path,
    )
    # Prepend the caller's task prompt so run-directory paths are known.
    user_prompt = task_prompt + "\n\n" + user_prompt

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.time()
    candidates_meta: list[dict[str, Any]] = []
    errors: list[str] = []

    for k in range(num_candidates):
        # Add anti-duplication reminder for later candidates.
        if k > 0 and candidates_meta:
            already = ", ".join(c["name"] for c in candidates_meta)
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        f"Propose ANOTHER distinct candidate exploring a different "
                        f"mechanism than: {already}. Same JSON schema."
                    ),
                }
            ]

        try:
            resp = litellm_completion(
                model=model,
                # Force the configured protocol; transmit model id verbatim.
                custom_llm_provider=llm_provider,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
            )
        except Exception as exc:  # network / auth / rate-limit
            errors.append(f"candidate {k}: litellm call failed: {exc}")
            continue

        raw = resp.choices[0].message.content or ""
        try:
            parsed = _extract_json(raw)
        except Exception as exc:
            errors.append(f"candidate {k}: JSON parse failed: {exc}")
            continue

        try:
            name = _sanitize_name(parsed["name"])
            code = parsed["code"]
            if not isinstance(code, str) or "class" not in code:
                raise ValueError("`code` field missing or not a valid module")
        except (KeyError, ValueError) as exc:
            errors.append(f"candidate {k}: schema check failed: {exc}")
            continue

        # Write agent file.
        agent_file = agents_dir / f"{name}.py"
        agent_file.write_text(code)

        candidates_meta.append(
            {
                "name": name,
                "file": f"agents/{name}.py",
                "hypothesis": parsed.get("hypothesis", ""),
                "axis": parsed.get("axis", "exploration"),
                "base_system": parsed.get("base_system", ""),
                "components": parsed.get("components", []),
            }
        )
        print(
            f"  proposed candidate: {name} "
            f"(axis={parsed.get('axis', '?')})",
            flush=True,
        )

    duration = time.time() - t0

    if not candidates_meta:
        return ProposerResult(
            exit_code=1,
            stderr="No valid candidates produced. Errors:\n" + "\n".join(errors),
            duration_seconds=duration,
        )

    # Write pending_eval.json.
    pending_eval_path.parent.mkdir(parents=True, exist_ok=True)
    pending_eval_path.write_text(
        json.dumps(
            {"iteration": iteration, "candidates": candidates_meta},
            indent=2,
        )
    )
    return ProposerResult(exit_code=0, stderr="\n".join(errors), duration_seconds=duration)
