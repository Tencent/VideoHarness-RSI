"""Proposer archive: historical harness source, scores, and traces.

The outer loop injects this pack into the proposer context each iteration.
Full on-disk files remain readable if a section is truncated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_SOURCE_CHARS = 16_000
_REQ_TEXT_CHARS = 400
_RESULTS_CAP = 350
_CTX_QUESTIONS_PER_HARNESS = 24
_DEFAULT_MAX_CHARS = 180_000


def _max_chars() -> int:
    raw = os.environ.get("PROPOSER_ARCHIVE_MAX_CHARS", "").strip()
    if raw:
        try:
            return max(20_000, int(raw))
        except ValueError:
            pass
    return _DEFAULT_MAX_CHARS


def _read(path: Path, cap: int | None = None) -> str:
    try:
        text = path.read_text()
    except OSError:
        return ""
    if cap is not None and len(text) > cap:
        return text[:cap] + f"\n# ... truncated {len(text) - cap} chars\n"
    return text


def _harness_names_from_summary(logs_dir: Path) -> list[str]:
    path = logs_dir / "evolution_summary.jsonl"
    names: list[str] = []
    seen: set[str] = set()
    if not path.is_file():
        return names
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("system", "base_system", "champion_at_proposal"):
            name = str(row.get(key) or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _harness_names_from_frontier(logs_dir: Path) -> list[str]:
    path = logs_dir / "frontier_val.json"
    names: list[str] = []
    if not path.is_file():
        return names
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return names
    for key, val in data.items():
        if str(key).startswith("_"):
            continue
        if isinstance(val, dict) and val.get("best_system"):
            names.append(str(val["best_system"]))
        elif isinstance(val, dict):
            names.extend(str(k) for k in val if k not in {"accuracy", "model"})
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _harness_names_from_val_dirs(logs_dir: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for val in logs_dir.glob("*/*/*/val.json"):
        harness = val.parent.parent.name
        if harness and harness not in seen:
            seen.add(harness)
            names.append(harness)
    return names


def archive_harness_names(
    logs_dir: Path, parent: str | None = None, extra: list[str] | None = None
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(name: str | None) -> None:
        n = (name or "").strip()
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)

    add(parent)
    for n in _harness_names_from_summary(logs_dir):
        add(n)
    for n in _harness_names_from_frontier(logs_dir):
        add(n)
    for n in extra or []:
        add(n)
    for n in _harness_names_from_val_dirs(logs_dir):
        add(n)
    return ordered


def _compact_val_results(val_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(val_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    results = data.get("results") or []
    compact = []
    for i, r in enumerate(results[:_RESULTS_CAP]):
        compact.append(
            {
                "i": i,
                "pred": r.get("prediction"),
                "tgt": r.get("target"),
                "ok": r.get("was_correct"),
                "pf": r.get("parse_fail"),
                "frames": r.get("num_frames"),
            }
        )
    return {
        "accuracy": data.get("accuracy"),
        "correct": data.get("correct"),
        "total": data.get("total"),
        "parse_fail": data.get("parse_fail"),
        "num_frames": data.get("num_frames"),
        "visual_tokens": data.get("visual_tokens"),
        "results": compact,
    }


def _compact_contexts(ctx_path: Path) -> list[dict[str, Any]]:
    if not ctx_path.is_file():
        return []
    wrong: list[dict[str, Any]] = []
    right: list[dict[str, Any]] = []
    try:
        lines = ctx_path.read_text().splitlines()
    except OSError:
        return []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        reqs = []
        for rq in rec.get("requests") or []:
            if rq.get("phase") == "ingest":
                continue
            reqs.append(
                {
                    "phase": rq.get("phase"),
                    "num_images": rq.get("num_images"),
                    "text": (rq.get("text") or "")[:_REQ_TEXT_CHARS],
                    "response": (rq.get("response") or "")[:_REQ_TEXT_CHARS],
                }
            )
        packed = {
            "i": rec.get("episode_index", i),
            "qid": rec.get("question_id"),
            "ok": rec.get("was_correct"),
            "pred": rec.get("final_prediction"),
            "n_req": rec.get("num_requests"),
            "requests": reqs[-2:],
        }
        (wrong if rec.get("was_correct") is False else right).append(packed)
    n_wrong = min(len(wrong), _CTX_QUESTIONS_PER_HARNESS)
    n_right = min(len(right), max(0, _CTX_QUESTIONS_PER_HARNESS - n_wrong))
    return wrong[:n_wrong] + right[:n_right]


def _source_block(agents_dir: Path, name: str) -> str:
    path = agents_dir / f"{name}.py"
    if not path.is_file():
        return f"### agents/{name}.py\n(missing)\n"
    body = _read(path, _SOURCE_CHARS)
    return f"### agents/{name}.py\n```python\n{body}\n```\n"


def _trace_block(logs_dir: Path, name: str) -> str:
    matches = list(logs_dir.glob(f"*/{name}/*/val.json"))
    if not matches:
        return f"### traces `{name}`\n(no val.json yet)\n"
    parts = [f"### traces `{name}`"]
    for val_path in sorted(matches)[:3]:
        rel = val_path.relative_to(logs_dir)
        parts.append(f"#### {rel}")
        compact = _compact_val_results(val_path)
        parts.append("```json")
        parts.append(json.dumps(compact, ensure_ascii=False))
        parts.append("```")
        ctx_path = val_path.with_name("val_contexts.jsonl")
        ctx = _compact_contexts(ctx_path)
        if ctx:
            parts.append("sampled answer-time requests (wrong first):")
            parts.append("```json")
            parts.append(json.dumps(ctx, ensure_ascii=False))
            parts.append("```")
    return "\n".join(parts) + "\n"


def render_proposer_archive(
    logs_dir: Path,
    agents_dir: Path,
    parent: str | None = None,
    extra_names: list[str] | None = None,
) -> str:
    """Markdown pack: scores, historical source, compact traces."""
    names = archive_harness_names(logs_dir, parent=parent, extra=extra_names)
    chunks: list[str] = [
        "# Proposer archive",
        "Injected historical harness source, scores, and traces.",
        "The candidate you write must still copy the current F_t.",
        "Truncated sections: Read the on-disk files under the run directory.",
        "",
        "## Scores",
    ]
    frontier = logs_dir / "frontier_val.json"
    summary = logs_dir / "evolution_summary.jsonl"
    if frontier.is_file():
        chunks.append("### frontier_val.json")
        chunks.append("```json")
        chunks.append(_read(frontier, 20_000))
        chunks.append("```")
    if summary.is_file():
        chunks.append("### evolution_summary.jsonl")
        chunks.append("```jsonl")
        chunks.append(_read(summary, 40_000))
        chunks.append("```")
    if not frontier.is_file() and not summary.is_file():
        chunks.append("(no scores yet — first iteration after baselines)")

    chunks.append("")
    chunks.append("## Historical harness source")
    if not names:
        chunks.append("(none yet)")
    for name in names:
        chunks.append(_source_block(agents_dir, name))

    chunks.append("")
    chunks.append("## Traces")
    if not names:
        chunks.append("(none yet)")
    for name in names:
        chunks.append(_trace_block(logs_dir, name))

    text = "\n".join(chunks).rstrip() + "\n"
    cap = _max_chars()
    if len(text) > cap:
        text = text[:cap] + (
            f"\n\n# archive truncated at {cap} chars; "
            "Read val.json / val_contexts.jsonl / agents/*.py on disk for the rest.\n"
        )
    return text


def write_proposer_archive(
    logs_dir: Path,
    agents_dir: Path,
    parent: str | None = None,
    extra_names: list[str] | None = None,
) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "proposer_archive.md"
    path.write_text(
        render_proposer_archive(
            logs_dir, agents_dir, parent=parent, extra_names=extra_names
        )
    )
    return path
