"""Real benchmark loaders for VL-Harness.

Currently implements **MLVU** (multiple-choice dev set). The loader is designed
to run on *whatever subset of videos is present locally*: an annotation is only
turned into an episode if its video file can be found under the data dir. This
lets us iterate on a handful of downloaded videos before (or without) pulling
the full multi-GB benchmark.

Expected on-disk layout (created by ``download_mlvu.py``)::

    <VL_HARNESS_DATA>/mlvu/
        mlvu_dev.json            # normalized annotations (list of dicts)
        videos/**/*.mp4          # raw videos (any nesting; matched by basename)

Each normalized annotation dict has keys::

    {video_name, question, candidates: [str,...], answer, task_type, question_id, duration}

``answer`` may be an option letter ("A"), a "(A)"/"A." form, or the full text of
the correct candidate; all are normalized to a letter here.

An *episode* (inner-loop example) is::

    {"input": json({episode_id, video_ref: <mp4 path>, question, options}),
     "target": <letter>, "meta": {task_type, duration, question_id}}
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def data_root() -> Path:
    return Path(
        os.path.expanduser(os.environ.get("VL_HARNESS_DATA", "~/.cache/vl-harness/data"))
    )


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------
def _answer_to_letter(answer: Any, candidates: list[str]) -> str | None:
    """Map an MLVU answer (letter / "(A)" / full option text) to a letter."""
    if answer is None:
        return None
    s = str(answer).strip()
    # bare or decorated letter, e.g. "A", "(A)", "A.", "A) foo"
    m = re.match(r"^\(?\s*([A-Ha-h])\s*[).:\-]?", s)
    if m and (len(s) <= 3 or not _looks_like_option_text(s)):
        idx = _LETTERS.index(m.group(1).upper())
        if idx < len(candidates):
            return _LETTERS[idx]
    # match against candidate text (case/space-insensitive)
    norm = _norm(s)
    for i, c in enumerate(candidates):
        if _norm(c) == norm:
            return _LETTERS[i]
    # candidate text possibly prefixed with its own letter, e.g. "(A) foo"
    for i, c in enumerate(candidates):
        cc = re.sub(r"^\(?[A-Ha-h]\)?[).:\-]?\s*", "", str(c)).strip()
        if _norm(cc) == norm:
            return _LETTERS[i]
    # last resort: leading letter even if longer string
    if m:
        idx = _LETTERS.index(m.group(1).upper())
        if idx < len(candidates):
            return _LETTERS[idx]
    return None


def _looks_like_option_text(s: str) -> bool:
    return len(s.split()) > 1


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


# ---------------------------------------------------------------------------
# Video file index
# ---------------------------------------------------------------------------
_TASK_FOLDERS = {
    "plotQA": "1_plotQA",
    "needle": "2_needle",
    "ego": "3_ego",
    "count": "4_count",
    "order": "5_order",
    "anomaly_reco": "6_anomaly_reco",
    "topic_reasoning": "7_topic_reasoning",
}


def _index_videos(videos_dir: Path) -> dict[str, list[str]]:
    """basename / stem -> list of absolute paths (handles cross-task collisions)."""
    index: dict[str, list[str]] = {}
    if not videos_dir.exists():
        return index
    for p in videos_dir.rglob("*"):
        if p.suffix.lower() in _VIDEO_EXTS and p.is_file():
            for key in (p.name, p.stem):
                index.setdefault(key, [])
                if str(p) not in index[key]:
                    index[key].append(str(p))
    return index


def _resolve_video(
    video_name: str,
    index: dict[str, list[str]],
    task_type: str | None = None,
) -> str | None:
    name = str(video_name)
    candidates: list[str] = []
    for key in (name, Path(name).name, Path(name).stem):
        candidates.extend(index.get(key, []))
    # dedupe preserve order
    seen = set()
    paths = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            paths.append(c)
    if not paths:
        return None
    if task_type:
        folder = _TASK_FOLDERS.get(task_type, "")
        if folder:
            for p in paths:
                if f"/{folder}/" in p.replace("\\", "/") or p.replace("\\", "/").endswith(
                    f"/{folder}/{Path(name).name}"
                ):
                    return p
        # test split videos live under videos/test/
        if task_type and "test" in str(task_type).lower():
            for p in paths:
                if "/test/" in p.replace("\\", "/"):
                    return p
    # prefer path under videos/test if name looks like test_*
    if str(video_name).startswith("test_"):
        for p in paths:
            if "/test/" in p.replace("\\", "/"):
                return p
    return paths[0]


# ---------------------------------------------------------------------------
# MLVU
# ---------------------------------------------------------------------------
def _mlvu_dir() -> Path:
    return data_root() / "mlvu"


def _load_mlvu_annotations(split: str = "dev") -> list[dict]:
    path = _mlvu_dir() / ("mlvu_test.json" if split == "test" else "mlvu_dev.json")
    if not path.exists():
        raise FileNotFoundError(
            f"MLVU annotations not found at {path}. Run:\n"
            f"    python scripts/download_mlvu.py --source official --all-mcq"
            + (" --include-test" if split == "test" else "")
        )
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", [])
    return data


def _mlvu_episodes_from_anns(anns: list[dict], split: str) -> list[dict]:
    index = _index_videos(_mlvu_dir() / "videos")
    episodes: list[dict] = []
    skipped = 0
    for i, a in enumerate(anns):
        video_name = a.get("video_name") or a.get("video") or a.get("video_path")
        candidates = a.get("candidates") or a.get("options") or []
        question = a.get("question", "")
        task_type = a.get("task_type", a.get("question_type", ""))
        if not video_name or not candidates or not question:
            skipped += 1
            continue
        path = _resolve_video(video_name, index, task_type=task_type or ("test" if split == "test" else None))
        if path is None:
            skipped += 1
            continue
        letter = _answer_to_letter(a.get("answer"), list(candidates))
        if letter is None:
            skipped += 1
            continue
        options = [f"{_LETTERS[j]}. {c}" for j, c in enumerate(candidates)]
        qid = a.get("question_id", a.get("qid", f"mlvu_{split}_{i}"))
        inp = {
            "episode_id": str(qid),
            "video_ref": path,
            "question": question,
            "options": options,
        }
        episodes.append(
            {
                "input": json.dumps(inp),
                "target": letter,
                "meta": {
                    "task_type": task_type,
                    "duration": a.get("duration"),
                    "question_id": str(qid),
                    "split": split,
                },
            }
        )
    if not episodes:
        raise RuntimeError(
            f"No MLVU {split} episodes have a matching local video. Download videos "
            f"into {_mlvu_dir() / 'videos'} (see download_mlvu.py). "
            f"({len(anns)} annotations, {skipped} skipped)."
        )
    return episodes


def _mlvu_episodes() -> list[dict]:
    return _mlvu_episodes_from_anns(_load_mlvu_annotations("dev"), "dev")


def _mlvu_test_episodes() -> list[dict]:
    return _mlvu_episodes_from_anns(_load_mlvu_annotations("test"), "test")




# ---------------------------------------------------------------------------
# LVBench
# ---------------------------------------------------------------------------
def _lvbench_dir():
    return data_root() / "lvbench"


def _parse_lvbench_question(text):
    """Split an LVBench question string into (stem, [option_texts])."""
    lines = [ln for ln in str(text).split("\n") if ln.strip() != ""]
    stem_parts = []
    opts = []
    for ln in lines:
        m = re.match(r"^\(([A-H])\)\s*(.*)$", ln.strip())
        if m:
            opts.append(m.group(2).strip())
        else:
            if not opts:
                stem_parts.append(ln.strip())
    stem = " ".join(stem_parts).strip()
    return stem, opts


def _load_lvbench_annotations():
    path = _lvbench_dir() / "video_info.meta.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"LVBench annotations not found at {path}. Run:\n"
            f"    python scripts/download_lvbench.py"
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _lvbench_episodes():
    records = _load_lvbench_annotations()
    # Optional debug filter: restrict to specific video keys to run a stable subset.
    # Example:
    #   export LVBENCH_KEYS="-hgaSElC3wU,16Z-XQh9jhk,20lTg3yUrO4"
    # This is useful when benchmarking pipeline stability before full 83-video runs.
    keys_env = os.environ.get("LVBENCH_KEYS", "").strip()
    if keys_env:
        allowed = {k.strip() for k in keys_env.split(",") if k.strip()}
        records = [r for r in records if str(r.get("key", "")) in allowed]
    index = _index_videos(_lvbench_dir() / "videos")
    episodes = []
    skipped = 0
    for rec in records:
        key = rec.get("key")
        if not key:
            skipped += 1
            continue
        path = _resolve_video(str(key), index)
        if path is None:
            skipped += 1
            continue
        for qa in rec.get("qa", []):
            stem, opts = _parse_lvbench_question(qa.get("question", ""))
            if not stem or len(opts) < 2:
                skipped += 1
                continue
            letter = _answer_to_letter(qa.get("answer"), opts)
            if letter is None:
                skipped += 1
                continue
            options = [f"{_LETTERS[j]}. {c}" for j, c in enumerate(opts)]
            qid = str(qa.get("uid", f"lvbench_{len(episodes)}"))
            qtypes = qa.get("question_type", [])
            inp = {
                "episode_id": f"{key}_{qid}",
                "video_ref": path,
                "question": stem,
                "options": options,
            }
            episodes.append(
                {
                    "input": json.dumps(inp),
                    "target": letter,
                    "meta": {
                        "task_type": ",".join(qtypes) if isinstance(qtypes, list) else str(qtypes),
                        "duration": (rec.get("video_info", {}) or {}).get("duration_minutes"),
                        "question_id": qid,
                        "time_reference": qa.get("time_reference"),
                        "split": "lvbench",
                    },
                }
            )
    if not episodes:
        raise RuntimeError(
            f"No LVBench episodes have a matching local video. Download videos into "
            f"{_lvbench_dir() / 'videos'} (see download_lvbench.py). "
            f"({len(records)} records, {skipped} skipped)."
        )
    return episodes

# ---------------------------------------------------------------------------
# Video-MME (evaluated WITHOUT subtitles)
# ---------------------------------------------------------------------------
def _video_mme_dir() -> Path:
    return data_root() / "video_mme"


def _load_video_mme_annotations() -> list[dict]:
    path = _video_mme_dir() / "video_mme.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Video-MME annotations not found at {path}. Run:\n"
            f"    python scripts/download_video_mme.py"
        )
    with open(path) as f:
        return json.load(f)


def _videomme_episodes() -> list[dict]:
    anns = _load_video_mme_annotations()
    index = _index_videos(_video_mme_dir() / "videos")
    episodes: list[dict] = []
    skipped = 0
    for i, a in enumerate(anns):
        video_name = a.get("video_name")
        candidates = a.get("candidates") or []
        question = a.get("question", "")
        if not video_name or not candidates or not question:
            skipped += 1
            continue
        path = _resolve_video(video_name, index)
        if path is None:
            skipped += 1
            continue
        letter = _answer_to_letter(a.get("answer"), list(candidates))
        if letter is None:
            skipped += 1
            continue
        options = [f"{_LETTERS[j]}. {c}" for j, c in enumerate(candidates)]
        qid = str(a.get("question_id", f"videomme_{i}"))
        inp = {
            "episode_id": qid,
            "video_ref": path,
            "question": question,
            "options": options,
        }
        episodes.append(
            {
                "input": json.dumps(inp),
                "target": letter,
                "meta": {
                    "task_type": a.get("task_type", ""),
                    "domain": a.get("domain"),
                    "sub_category": a.get("sub_category"),
                    "duration": a.get("duration"),
                    "question_id": qid,
                    "split": "videomme",
                },
            }
        )
    if not episodes:
        raise RuntimeError(
            f"No Video-MME episodes have a matching local video. Download videos into "
            f"{_video_mme_dir() / 'videos'} (see download_video_mme.py). "
            f"({len(anns)} annotations, {skipped} skipped)."
        )
    return episodes


# ---------------------------------------------------------------------------
# MVBench
# ---------------------------------------------------------------------------
def _mvbench_dir() -> Path:
    return data_root() / "mvbench"


def _load_mvbench_annotations() -> list[dict]:
    path = _mvbench_dir() / "mvbench.json"
    if not path.exists():
        raise FileNotFoundError(
            f"MVBench annotations not found at {path}. Run:\n"
            f"    python scripts/download_mvbench.py"
        )
    with open(path) as f:
        return json.load(f)


def _mvbench_episodes() -> list[dict]:
    anns = _load_mvbench_annotations()
    vdir = _mvbench_dir() / "videos"
    index = _index_videos(vdir)
    episodes: list[dict] = []
    skipped = 0
    for i, a in enumerate(anns):
        candidates = a.get("candidates") or []
        question = a.get("question", "")
        if not candidates or not question:
            skipped += 1
            continue
        # Prefer the resolved relative path baked in at download time; fall back
        # to a global basename match (handles re-extracted / moved video trees).
        path = None
        rel = a.get("video_relpath")
        if rel:
            cand = str(vdir / rel)
            if Path(cand).exists():
                path = cand
        if path is None:
            path = _resolve_video(a.get("video", ""), index)
        if path is None:
            skipped += 1
            continue
        letter = _answer_to_letter(a.get("answer"), list(candidates))
        if letter is None:
            skipped += 1
            continue
        options = [f"{_LETTERS[j]}. {c}" for j, c in enumerate(candidates)]
        task_type = a.get("task_type", "")
        qid = f"mvbench_{task_type}_{i}"
        inp = {
            "episode_id": qid,
            "video_ref": path,
            "question": question,
            "options": options,
        }
        episodes.append(
            {
                "input": json.dumps(inp),
                "target": letter,
                "meta": {
                    "task_type": task_type,
                    "question_id": qid,
                    "split": "mvbench",
                },
            }
        )
    if not episodes:
        raise RuntimeError(
            f"No MVBench episodes have a matching local video. Download videos into "
            f"{_mvbench_dir() / 'videos'} (see download_mvbench.py). "
            f"({len(anns)} annotations, {skipped} skipped)."
        )
    return episodes


# ---------------------------------------------------------------------------
# Registry + split entrypoint
# ---------------------------------------------------------------------------
_LOADERS: dict[str, Callable[[], list[dict]]] = {
    "mlvu": _mlvu_episodes,
    "mlvu_test": _mlvu_test_episodes,
    "lvbench": _lvbench_episodes,
    "video_mme": _videomme_episodes,
    "mvbench": _mvbench_episodes,
}

_LVBENCH_QUESTION_SPLIT = (
    Path(__file__).resolve().parent.parent / "manifests" / "lvbench_split_seed42.json"
)


def _lvbench_episode_id(ep: dict[str, Any]) -> str:
    inp = ep.get("input")
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except json.JSONDecodeError:
            inp = {}
    if isinstance(inp, dict) and inp.get("episode_id"):
        return str(inp["episode_id"])
    meta = ep.get("meta") or {}
    vid = meta.get("video_id")
    qid = meta.get("question_id")
    if vid and qid:
        return f"{vid}_{qid}"
    return str(qid or "")


def _lvbench_question_split_enabled() -> bool:
    raw = os.environ.get("LVBENCH_QUESTION_SPLIT", "").strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _LVBENCH_QUESTION_SPLIT.is_file()


def load_real_splits_3way(
    task: str,
    num_train: int = 20,
    num_val: int = 40,
    num_test: int = 40,
    shuffle_seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], Callable]:
    from .data import get_evaluator  # local import to avoid cycle at import time

    if task not in _LOADERS:
        raise NotImplementedError(
            f"Real loader for '{task}' not implemented yet. Available: {sorted(_LOADERS)}"
        )
    pool = _LOADERS[task]()
    if task == "lvbench" and _lvbench_question_split_enabled():
        if not _LVBENCH_QUESTION_SPLIT.is_file():
            raise RuntimeError(
                f"LVBENCH_QUESTION_SPLIT is on but missing {_LVBENCH_QUESTION_SPLIT}"
            )
        man = json.loads(_LVBENCH_QUESTION_SPLIT.read_text())
        by_id = {_lvbench_episode_id(ep): ep for ep in pool}

        def _take(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            missing: list[str] = []
            for rec in recs:
                eid = str(rec.get("episode_id") or "")
                ep = by_id.get(eid)
                if ep is None:
                    missing.append(eid)
                else:
                    out.append(ep)
            if missing:
                raise RuntimeError(
                    f"lvbench question split missing {len(missing)} ids, "
                    f"e.g. {missing[:4]}"
                )
            return out

        val_full = _take(man.get("val") or [])
        test_full = _take(man.get("test") or [])
        train: list[dict[str, Any]] = []
        if num_val <= 0:
            val = []
        else:
            val = val_full[:num_val]
        if num_test <= 0:
            test = []
        elif num_val <= 0 and num_test >= len(val_full) + len(test_full):
            test = val_full + test_full
        else:
            test = test_full[:num_test]
        print(
            f"[split] lvbench pinned question-level seed=42 "
            f"val={len(val)}q test={len(test)}q "
            f"manifest={_LVBENCH_QUESTION_SPLIT.name}",
            flush=True,
        )
        return train, val, test, get_evaluator(task)
    random.Random(shuffle_seed).shuffle(pool)
    total = num_train + num_val + num_test
    if total and total < len(pool):
        pool = pool[:total]
    train = pool[:num_train]
    val = pool[num_train : num_train + num_val]
    test = pool[num_train + num_val :]
    return train, val, test, get_evaluator(task)
