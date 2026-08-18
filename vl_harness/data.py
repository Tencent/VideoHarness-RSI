"""Datasets for VL-Harness.

Each *example* fed to the inner loop is one ``(video, question)`` pair:

    {"input": <json string>, "target": <option letter>}

where the JSON string encodes ``{episode_id, video_ref, question, options}``.
``video_ref`` is either a mock spec dict (see ``video.VideoStream.from_mock``)
or a path to a real mp4 for benchmark tasks.

Tasks:
- ``mock_niah`` : self-generating "needle-in-haystack" long-video MCQ. A single
  needle frame carries the answer; the more of the video a harness actually
  looks at / retrieves, the higher its accuracy. Runs fully offline with
  ``StubVLM`` to validate the whole pipeline.
- real benchmark tasks (videomme_long / mlvu / lvbench / egoschema / hourvideo)
  are registered lazily via ``loaders_real`` once data is downloaded.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable

MOCK_TASKS = ["mock_niah"]
# Real tasks are wired in P2 (see loaders_real.py). Kept here so ALL_TASKS is
# the single source of truth the inner loop advertises.
REAL_TASKS: list[str] = [
    "videomme_long",
    "video_mme",
    "mvbench",
    "mlvu",
    "mlvu_test",
    "lvbench",
    "egoschema",
    "hourvideo",
]
ALL_TASKS = MOCK_TASKS + REAL_TASKS

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# ---------------------------------------------------------------------------
# MCQ evaluator
# ---------------------------------------------------------------------------
def eval_mcq(prediction: str, target: str, **kwargs) -> bool:
    """Exact-match on option letter (case-insensitive). Low-noise signal."""
    if prediction is None:
        return False
    p = str(prediction).strip().upper()
    t = str(target).strip().upper()
    # Accept a bare letter or a leading letter like "B. blue"
    if p and p[0].isalpha():
        p = p[0]
    if t and t[0].isalpha():
        t = t[0]
    return p == t


def get_evaluator(task: str) -> Callable:
    return eval_mcq


# ---------------------------------------------------------------------------
# Mock needle-in-haystack generator
# ---------------------------------------------------------------------------
_COLORS = ["red", "blue", "green", "yellow"]
_OBJECTS = ["box", "umbrella", "backpack", "mug"]


def _gen_mock_episode(rng: random.Random, idx: int, num_frames: int = 60) -> dict:
    ans_i = rng.randrange(4)
    answer = _LETTERS[ans_i]
    color = _COLORS[ans_i]
    obj = _OBJECTS[rng.randrange(len(_OBJECTS))]
    needle_t = rng.randrange(num_frames)

    frames = []
    for t in range(num_frames):
        if t == needle_t:
            frames.append(
                {"t": t, "caption": f"a person clearly holds a {color} {obj}", "hint": answer}
            )
        else:
            distract = _COLORS[(ans_i + 1 + (t % 3)) % 4]
            frames.append(
                {"t": t, "caption": f"an ordinary street scene, a {distract} car drives by (t={t})"}
            )

    spec = {
        "video_id": f"mock_{idx}",
        "duration": float(num_frames),
        "fps": 1.0,
        "frames": frames,
    }
    question = f"What color is the {obj} the person holds?"
    options = [f"{_LETTERS[i]}. {c}" for i, c in enumerate(_COLORS)]
    inp = {
        "episode_id": f"mock_{idx}",
        "video_ref": spec,
        "question": question,
        "options": options,
    }
    return {"input": json.dumps(inp), "target": answer}


def _load_mock(n: int, seed: int, num_frames: int = 60) -> list[dict]:
    rng = random.Random(seed)
    return [_gen_mock_episode(rng, i, num_frames=num_frames) for i in range(n)]


# ---------------------------------------------------------------------------
# Split entrypoints (mirror the text example's signatures)
# ---------------------------------------------------------------------------
def load_dataset_splits_3way(
    task: str,
    num_train: int = 20,
    num_val: int = 40,
    num_test: int = 40,
    shuffle_seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], Callable]:
    evaluator = get_evaluator(task)
    if task in MOCK_TASKS:
        total = num_train + num_val + num_test
        pool = _load_mock(total, seed=shuffle_seed)
        random.Random(shuffle_seed).shuffle(pool)
        train = pool[:num_train]
        val = pool[num_train : num_train + num_val]
        test = pool[num_train + num_val :]
        return train, val, test, evaluator
    if task in REAL_TASKS:
        from .loaders_real import load_real_splits_3way

        return load_real_splits_3way(
            task, num_train, num_val, num_test, shuffle_seed
        )
    raise ValueError(f"Unknown task: {task}")


def load_dataset_splits(
    task: str,
    num_train: int,
    num_test: int = 20,
    shuffle_seed: int = 42,
) -> tuple[list[dict], list[dict], Callable]:
    train, _val, test, evaluator = load_dataset_splits_3way(
        task, num_train=num_train, num_val=0, num_test=num_test, shuffle_seed=shuffle_seed
    )
    return train, test, evaluator
