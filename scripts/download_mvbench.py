"""Fetch the MVBench benchmark for VL-Harness.

Source: HF dataset ``OpenGVLab/MVBench`` (MIT license, gated terms apply).
This script writes a local cache under ``$VL_HARNESS_DATA``; do not commit
those files. 320 NTU RGB+D videos are not redistributed by MVBench.

- Annotations: 20 per-task files ``json/<task>.json``. Each record is
  ``{video, question, candidates, answer}`` where ``video`` is a bare basename
  (e.g. ``video_6480.mp4``) and ``answer`` is the full text of the correct
  candidate. We merge them into one ``mvbench.json`` and tag each record with
  its ``task_type``.
- Videos: 11 archives ``video/*.zip`` (~17 GB). Internal layout varies by source
  dataset (flat ``ssv2_video/93560.webm``, nested ``data0613/star/...``,
  ``vlnqa/stop/...``), so we extract **preserving the archive structure** and
  then resolve each annotation's basename to an on-disk relative path
  (``video_relpath``). When a basename appears under more than one top-level
  folder, we pick the folder that holds the majority of that task's videos.

Note: 320 NTU RGB+D videos are NOT redistributed (see
``video/MVBench_videos_ntu.txt``); annotations referencing them are skipped by
the loader until those videos are added manually.

On-disk layout produced::

    <VL_HARNESS_DATA>/mvbench/
        mvbench.json            # normalized annotations (+ video_relpath)
        videos/<zip-internal-paths...>   # extracted videos (structure preserved)

Examples::

    # annotations only (tiny)
    python scripts/download_mvbench.py

    # annotations + all videos (~17 GB) + path resolution
    python scripts/download_mvbench.py --download-videos --workers 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from vl_harness.loaders_real import _mvbench_dir

REPO = "OpenGVLab/MVBench"
TASKS = [
    "action_antonym", "action_count", "action_localization", "action_prediction",
    "action_sequence", "character_order", "counterfactual_inference",
    "egocentric_navigation", "episodic_reasoning", "fine_grained_action",
    "fine_grained_pose", "moving_attribute", "moving_count", "moving_direction",
    "object_existence", "object_interaction", "object_shuffle",
    "scene_transition", "state_change", "unexpected_action",
]
_VID_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v")


def build_annotations() -> Path:
    from huggingface_hub import hf_hub_download

    out_dir = _mvbench_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    anns = []
    for t in TASKS:
        p = hf_hub_download(REPO, f"json/{t}.json", repo_type="dataset")
        rows = json.loads(Path(p).read_text())
        for r in rows:
            anns.append(
                {
                    "video": r.get("video"),
                    "question": r.get("question", ""),
                    "candidates": list(r.get("candidates") or []),
                    "answer": r.get("answer"),
                    "task_type": t,
                    "video_relpath": None,  # filled by resolve_paths()
                }
            )
        print(f"[mvbench] {t}: {len(rows)} questions", flush=True)
    out = out_dir / "mvbench.json"
    out.write_text(json.dumps(anns, ensure_ascii=False))
    print(f"[mvbench] wrote {len(anns)} questions across {len(TASKS)} tasks -> {out}", flush=True)
    return out


def _list_video_zips() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith("video/") and f.endswith(".zip"))


def _extract_zip_preserving(zip_path: str, dest: Path) -> int:
    """Extract video files keeping the archive's internal directory structure."""
    new = 0
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            if m.endswith("/"):
                continue
            if not m.lower().endswith(_VID_EXTS):
                continue
            target = dest / m
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            new += 1
    return new


def resolve_paths() -> None:
    """Bake the on-disk relative video path into each annotation.

    Indexes every extracted video by basename, then resolves each annotation.
    For basenames that collide across source folders, prefer the top-level
    folder containing the majority of that task's videos.
    """
    out_dir = _mvbench_dir()
    vdir = out_dir / "videos"
    ann_path = out_dir / "mvbench.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing {ann_path}; run build_annotations first")

    anns = json.loads(ann_path.read_text())
    idx: dict[str, list[str]] = defaultdict(list)
    for p in vdir.rglob("*"):
        if p.is_file() and p.suffix.lower() in _VID_EXTS:
            idx[p.name].append(str(p.relative_to(vdir)))

    by_task: dict[str, list[str]] = defaultdict(list)
    for a in anns:
        by_task[a.get("task_type", "")].append(str(a.get("video", "")))
    task_folder: dict[str, str] = {}
    for t, vids in by_task.items():
        folders: Counter = Counter()
        for v in vids:
            for rel in idx.get(Path(v).name, []):
                folders[rel.split("/")[0]] += 1
        if folders:
            task_folder[t] = folders.most_common(1)[0][0]

    resolved = 0
    missing = 0
    for a in anns:
        name = Path(str(a.get("video", ""))).name
        rels = idx.get(name, [])
        if not rels:
            a["video_relpath"] = None
            missing += 1
            continue
        if len(rels) == 1:
            a["video_relpath"] = rels[0]
        else:
            tf = task_folder.get(a.get("task_type", ""))
            a["video_relpath"] = next(
                (r for r in rels if r.split("/")[0] == tf), rels[0]
            )
        resolved += 1
    ann_path.write_text(json.dumps(anns, ensure_ascii=False))
    print(
        f"[mvbench] resolved {resolved}/{len(anns)} video paths "
        f"({missing} without a local video, e.g. NTU RGB+D)",
        flush=True,
    )


def download_videos(workers: int = 4) -> None:
    from huggingface_hub import hf_hub_download

    zips = _list_video_zips()
    dest = _mvbench_dir() / "videos"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[mvbench] {len(zips)} video archives to fetch", flush=True)

    def _one(repo_path: str):
        local = hf_hub_download(REPO, repo_path, repo_type="dataset")
        n = _extract_zip_preserving(local, dest)
        return repo_path, n

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_one, z): z for z in zips}
        for fut in as_completed(futs):
            z = futs[fut]
            try:
                _, n = fut.result()
                done += 1
                print(f"[mvbench] {z}: +{n} videos ({done}/{len(zips)} zips done)", flush=True)
            except Exception as e:
                failed.append((z, str(e)))
                print(f"[mvbench] FAIL {z}: {e}", flush=True)
    if failed:
        print("[mvbench] failed zips:", failed, flush=True)
    # Bake resolved relative paths into the annotations.
    resolve_paths()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download MVBench benchmark")
    ap.add_argument("--download-videos", action="store_true", help="also fetch videos (~17 GB)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-annotations", action="store_true")
    args = ap.parse_args()

    if not args.skip_annotations:
        build_annotations()
    if args.download_videos:
        download_videos(workers=args.workers)
    else:
        print(
            "[mvbench] annotations ready. No videos fetched. Re-run with "
            f"--download-videos to pull video zips into {_mvbench_dir() / 'videos'}.",
            flush=True,
        )


if __name__ == "__main__":
    main()
