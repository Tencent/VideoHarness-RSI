"""Fetch the MLVU benchmark for VL-Harness.

MLVU/MVLU is CC-BY-NC-SA-4.0 (research / non-commercial; gated HF terms).
MLVU does not own raw-video copyright. This script writes a local cache under
``$VL_HARNESS_DATA``; do not commit those files.

Annotations (tiny, non-gated) come from ``sy1998/MLVU_dev``.
Videos come from the gated official ``MLVU/MVLU`` (individual mp4s) and/or
``MLVU/MLVU_Test`` (split tar for the test set).

Examples::

    # Dev MCQ annotations only
    python -m vl_harness.download_mlvu

    # Small subset (smallest N per task)
    HF_TOKEN=hf_xxx python -m vl_harness.download_mlvu \\
        --source official --tasks needle,plotQA --max-videos 20

    # FULL Dev MCQ videos (~190GB, 1122 unique files) — what you usually want
    HF_TOKEN=hf_xxx python -m vl_harness.download_mlvu \\
        --source official --all-mcq --workers 8

    # Also pull Test MCQ annotations + videos (~77GB tar parts)
    HF_TOKEN=hf_xxx python -m vl_harness.download_mlvu \\
        --source official --all-mcq --include-test --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .loaders_real import _mlvu_dir

SY1998_REPO = "sy1998/MLVU_dev"
OFFICIAL_REPO = "MLVU/MVLU"
TEST_REPO = "MLVU/MLVU_Test"
PARQUET_PATH = "mlvu/test-00000-of-00001.parquet"

_TASK_ALIASES = {
    "plotQA": "1_plotQA",
    "needle": "2_needle",
    "ego": "3_ego",
    "count": "4_count",
    "order": "5_order",
    "anomaly_reco": "6_anomaly_reco",
    "topic_reasoning": "7_topic_reasoning",
}


def _strip_options(question: str) -> str:
    q = re.split(r"\n\s*\(?[A-H]\)?[).:]", question)[0]
    return q.strip()


def build_annotations() -> Path:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    out_dir = _mlvu_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[mlvu] downloading annotations parquet from {SY1998_REPO} ...")
    p = hf_hub_download(SY1998_REPO, PARQUET_PATH, repo_type="dataset")
    table = pq.read_table(p)
    rows = table.to_pylist()
    anns = []
    for r in rows:
        anns.append(
            {
                "video_name": r["video_name"],
                "question": _strip_options(r["question"]),
                "candidates": list(r["candidates"]),
                "answer": r["answer"],
                "task_type": r["task_type"],
                "question_id": r.get("question_id", ""),
                "duration": r.get("duration"),
            }
        )
    out = out_dir / "mlvu_dev.json"
    out.write_text(json.dumps(anns, ensure_ascii=False))
    print(f"[mlvu] wrote {len(anns)} DEV MCQ annotations -> {out}")
    return out


def build_test_annotations() -> Path:
    """Download Test MCQ questions + ground truth into mlvu_test.json."""
    from huggingface_hub import hf_hub_download

    out_dir = _mlvu_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    q_path = hf_hub_download(TEST_REPO, "test_multi_choice_tasks.json", repo_type="dataset")
    gt_path = hf_hub_download(
        TEST_REPO, "test-ground-truth/test_mcq_gt.json", repo_type="dataset"
    )
    questions = json.loads(Path(q_path).read_text())
    gt = json.loads(Path(gt_path).read_text())
    # GT list aligns with questions; also keyed by question_id when present.
    gt_by_id = {
        str(g.get("question_id", "")): g for g in gt if g.get("question_id") is not None
    }
    anns = []
    for i, q in enumerate(questions):
        g = gt_by_id.get(str(q.get("question_id", "")), gt[i] if i < len(gt) else {})
        answer = g.get("answer", q.get("answer"))
        candidates = q.get("candidates") or g.get("candidates") or []
        anns.append(
            {
                "video_name": q.get("video") or q.get("video_name"),
                "question": _strip_options(q.get("question", "")),
                "candidates": list(candidates),
                "answer": answer,
                "task_type": q.get("question_type") or q.get("task_type") or "",
                "question_id": q.get("question_id", f"test_{i}"),
                "duration": q.get("duration"),
                "split": "test",
            }
        )
    out = out_dir / "mlvu_test.json"
    out.write_text(json.dumps(anns, ensure_ascii=False))
    print(f"[mlvu] wrote {len(anns)} TEST MCQ annotations -> {out}")
    return out


def _needed_video_names(task: str | None) -> set[str]:
    ann = _mlvu_dir() / "mlvu_dev.json"
    if not ann.exists():
        return set()
    data = json.loads(ann.read_text())
    return {
        a["video_name"]
        for a in data
        if task in (None, "all") or a.get("task_type") == task
    }


def _link_or_skip(local: str, target: Path) -> bool:
    """Symlink HF cache file into our data dir. Returns True if newly linked."""
    if target.exists() or target.is_symlink():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(local)
    return True


def download_official_subset(task: str, max_videos: int) -> None:
    """Download the smallest ``max_videos`` mp4s for ``task``."""
    from huggingface_hub import HfApi, hf_hub_download

    folder = _TASK_ALIASES.get(task)
    if folder is None:
        raise ValueError(f"Unknown task '{task}'. Choose from {sorted(_TASK_ALIASES)}")
    api = HfApi()
    files = [
        f
        for f in api.list_repo_files(OFFICIAL_REPO, repo_type="dataset")
        if f.startswith(f"MLVU/video/{folder}/") and f.lower().endswith(".mp4")
    ]
    if not files:
        raise RuntimeError(f"No videos found for {folder} in {OFFICIAL_REPO}")
    infos = api.get_paths_info(OFFICIAL_REPO, files, repo_type="dataset")
    infos = sorted(infos, key=lambda it: getattr(it, "size", 0) or 0)
    chosen = infos[:max_videos]
    dest = _mlvu_dir() / "videos" / folder
    dest.mkdir(parents=True, exist_ok=True)
    new = 0
    for it in chosen:
        sz = (getattr(it, "size", 0) or 0) / 1e6
        target = dest / Path(it.path).name
        if target.exists() or target.is_symlink():
            print(f"[mlvu] skip (exists) {it.path} ({sz:.1f} MB)")
            continue
        print(f"[mlvu] downloading {it.path} ({sz:.1f} MB) ...")
        local = hf_hub_download(OFFICIAL_REPO, it.path, repo_type="dataset")
        if _link_or_skip(local, target):
            new += 1
    print(f"[mlvu] task={task}: {new} new / {len(chosen)} selected -> {dest}")


def _dev_mcq_repo_paths() -> list[tuple[str, str]]:
    """Return [(repo_path, local_relpath)] for every Dev MCQ video."""
    ann = _mlvu_dir() / "mlvu_dev.json"
    if not ann.exists():
        raise FileNotFoundError(f"Missing {ann}; run without --skip-annotations first")
    data = json.loads(ann.read_text())
    pairs = []
    seen = set()
    for a in data:
        task = a["task_type"]
        folder = _TASK_ALIASES[task]
        name = a["video_name"]
        repo_path = f"MLVU/video/{folder}/{name}"
        if repo_path in seen:
            continue
        seen.add(repo_path)
        pairs.append((repo_path, f"videos/{folder}/{name}"))
    return pairs


def download_all_mcq(workers: int = 8) -> None:
    """Download every video referenced by Dev MCQ annotations (~190GB)."""
    from huggingface_hub import hf_hub_download

    pairs = _dev_mcq_repo_paths()
    root = _mlvu_dir()
    todo = []
    for repo_path, rel in pairs:
        target = root / rel
        if target.exists() or target.is_symlink():
            continue
        todo.append((repo_path, target))
    print(
        f"[mlvu] Dev MCQ videos: {len(pairs)} unique, "
        f"{len(pairs) - len(todo)} already local, {len(todo)} to download",
        flush=True,
    )
    if not todo:
        print("[mlvu] nothing to download for Dev MCQ")
        return

    done = 0
    failed = []

    def _one(item):
        repo_path, target = item
        local = hf_hub_download(OFFICIAL_REPO, repo_path, repo_type="dataset")
        _link_or_skip(local, target)
        return repo_path

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_one, it): it for it in todo}
        for fut in as_completed(futs):
            repo_path, target = futs[fut]
            try:
                fut.result()
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"[mlvu] progress {done}/{len(todo)}", flush=True)
            except Exception as e:
                failed.append((repo_path, str(e)))
                print(f"[mlvu] FAIL {repo_path}: {e}", flush=True)
    print(f"[mlvu] Dev MCQ done: ok={done} fail={len(failed)}")
    if failed:
        print("[mlvu] first failures:", failed[:5])


def download_test_videos(workers: int = 4) -> None:
    """Download MLVU Test split video tar parts, concat, extract mp4s."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    parts = [f"MLVU_Test/test_video.tar.gz.part-{s}" for s in "abcdefgh"]
    infos = api.get_paths_info(TEST_REPO, parts, repo_type="dataset", expand=True)
    total = sum((getattr(i, "size", 0) or 0) for i in infos)
    print(f"[mlvu] Test video parts: {len(parts)}, ~{total / 1e9:.1f} GB", flush=True)

    part_dir = _mlvu_dir() / "test_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    local_parts = []
    for p in parts:
        print(f"[mlvu] downloading {p} ...", flush=True)
        local = hf_hub_download(TEST_REPO, p, repo_type="dataset")
        local_parts.append(local)

    merged = _mlvu_dir() / "test_video.tar.gz"
    if not merged.exists():
        print(f"[mlvu] concatenating -> {merged}", flush=True)
        with open(merged, "wb") as out:
            for lp in local_parts:
                with open(lp, "rb") as inp:
                    shutil.copyfileobj(inp, out)
    else:
        print(f"[mlvu] skip concat (exists) {merged}")

    dest = _mlvu_dir() / "videos" / "test"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[mlvu] extracting mp4s -> {dest}", flush=True)
    with tarfile.open(merged, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.lower().endswith(".mp4")]
        for i, m in enumerate(members, 1):
            base = Path(m.name).name
            target = dest / base
            if target.exists():
                continue
            src = tar.extractfile(m)
            if src is None:
                continue
            with open(target, "wb") as f:
                shutil.copyfileobj(src, f)
            if i % 20 == 0 or i == len(members):
                print(f"[mlvu] extracted {i}/{len(members)}", flush=True)
    print(f"[mlvu] Test videos ready under {dest}")


def download_sy1998_zips(zip_parts: list[int], task: str | None) -> None:
    from huggingface_hub import hf_hub_download

    needed = _needed_video_names(task)
    dest = _mlvu_dir() / "videos"
    dest.mkdir(parents=True, exist_ok=True)
    for n in zip_parts:
        fname = f"video_part_{n}.zip"
        print(f"[mlvu] downloading {fname} (tens of GB) ...")
        local = hf_hub_download(SY1998_REPO, fname, repo_type="dataset")
        with zipfile.ZipFile(local) as zf:
            for m in zf.namelist():
                base = Path(m).name
                if not base.lower().endswith(".mp4"):
                    continue
                if needed and base not in needed:
                    continue
                target = dest / base
                if target.exists():
                    continue
                with zf.open(m) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        print(f"[mlvu] extracted needed videos from {fname} -> {dest}")


def main():
    ap = argparse.ArgumentParser(description="Download MLVU benchmark")
    ap.add_argument(
        "--source",
        choices=["none", "official", "sy1998"],
        default="none",
        help="video source; 'none' = annotations only",
    )
    ap.add_argument("--task", default="needle", help="task or comma-separated list")
    ap.add_argument("--tasks", default=None, help="overrides --task")
    ap.add_argument("--max-videos", type=int, default=5)
    ap.add_argument(
        "--all-mcq",
        action="store_true",
        help="download ALL videos referenced by Dev MCQ (~190GB)",
    )
    ap.add_argument(
        "--include-test",
        action="store_true",
        help="also download Test MCQ annotations + videos (~77GB)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--zip-parts", type=int, nargs="*", default=[8])
    ap.add_argument("--skip-annotations", action="store_true")
    args = ap.parse_args()

    if not args.skip_annotations:
        build_annotations()
    if args.include_test:
        build_test_annotations()

    if args.source == "official":
        if args.all_mcq:
            download_all_mcq(workers=args.workers)
        else:
            tasks = [
                t.strip() for t in (args.tasks or args.task).split(",") if t.strip()
            ]
            for t in tasks:
                download_official_subset(t, args.max_videos)
        if args.include_test:
            download_test_videos(workers=args.workers)
    elif args.source == "sy1998":
        download_sy1998_zips(args.zip_parts, args.task)
    else:
        print(
            "[mlvu] annotations ready. No videos fetched (source=none).\n"
            "       Re-run with --source official --all-mcq [--include-test]."
        )


if __name__ == "__main__":
    main()
