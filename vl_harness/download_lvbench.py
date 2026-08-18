"""Fetch the LVBench benchmark for VL-Harness.

LVBench (THUDM/LVBench) is CC-BY-NC-SA-4.0: academic / non-commercial only.
This script does not relicense it. Videos are NOT redistributed by LVBench or
by this repo; each record's ``key`` is a YouTube id.

**Preferred video source is ModelScope** (``AI-ModelScope/LVBench``), which is
the snapshot behind ``manifests/lvbench_videos.json``. YouTube via yt-dlp is
opt-in (``--source youtube``) and may conflict with YouTube's terms of service;
use it only if you have the right to fetch those videos.

The loader only builds an episode for a video that is present locally, so you
can iterate on a subset.

Examples::

    # annotations only (no video download)
    python -m vl_harness.download_lvbench

    # annotations + ModelScope videos (preferred)
    python -m vl_harness.download_lvbench --download-videos --max-videos 0

    # opt-in YouTube via yt-dlp (ToS / copyright: your responsibility)
    python -m vl_harness.download_lvbench --download-videos --source youtube --max-videos 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .loaders_real import _lvbench_dir, _load_lvbench_annotations

LVBENCH_REPO = "THUDM/LVBench"
META_FILE = "video_info.meta.jsonl"


def build_annotations() -> Path:
    from huggingface_hub import hf_hub_download

    out_dir = _lvbench_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lvbench] downloading {META_FILE} from {LVBENCH_REPO} ...", flush=True)
    local = hf_hub_download(LVBENCH_REPO, META_FILE, repo_type="dataset")
    target = out_dir / META_FILE
    shutil.copyfile(local, target)
    # Count questions for a quick sanity report.
    n_vid = 0
    n_qa = 0
    with open(target) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_vid += 1
            n_qa += len(rec.get("qa", []))
    print(f"[lvbench] wrote {n_vid} videos / {n_qa} questions -> {target}", flush=True)
    return target


def _have_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None


def download_videos(max_videos: int = 10) -> None:
    """Download up to max_videos LVBench videos from YouTube via yt-dlp.

    Requires network access to YouTube and a working yt-dlp binary. Videos are
    saved as videos/<key>.mp4 so the loader can match them by basename.
    """
    if not _have_yt_dlp():
        raise RuntimeError(
            "yt-dlp not found. Install it (pip install yt-dlp) and ensure YouTube "
            "is reachable from this machine."
        )
    records = _load_lvbench_annotations()
    dest = _lvbench_dir() / "videos"
    dest.mkdir(parents=True, exist_ok=True)
    keys = []
    for rec in records:
        k = rec.get("key")
        if k and k not in keys:
            keys.append(k)
    if max_videos and max_videos > 0:
        keys = keys[:max_videos]
    ok = 0
    fail = []
    for i, key in enumerate(keys, 1):
        target = dest / f"{key}.mp4"
        if target.exists():
            print(f"[lvbench] skip (exists) {key}", flush=True)
            ok += 1
            continue
        url = f"https://www.youtube.com/watch?v={key}"
        print(f"[lvbench] ({i}/{len(keys)}) downloading {key} ...", flush=True)
        cmd = [
            "yt-dlp", "-f", "mp4", "-o", str(dest / "%(id)s.%(ext)s"), url,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if res.returncode == 0 and target.exists():
                ok += 1
            else:
                fail.append((key, (res.stderr or res.stdout)[:200]))
        except Exception as e:
            fail.append((key, str(e)))
    print(f"[lvbench] videos: ok={ok} fail={len(fail)} / {len(keys)} selected", flush=True)
    if fail:
        print("[lvbench] first failures:", fail[:5], flush=True)
        print(
            "[lvbench] If all failed, this machine likely cannot reach YouTube. "
            "Download videos elsewhere and copy them into "
            f"{dest} as <youtube_id>.mp4.", flush=True,
        )


def download_videos_modelscope(max_videos=0):
    """Download LVBench videos from ModelScope (AI-ModelScope/LVBench)."""
    """No YouTube/cookies needed. Files are videos/<youtube_id>.mp4, matching"""
    """the loader's basename lookup. ~83 of 103 videos available (rest are"""
    """copyright-removed). Videos land in data/lvbench/videos/."""
    from modelscope.hub.api import HubApi
    from modelscope.hub.snapshot_download import dataset_snapshot_download

    repo = "AI-ModelScope/LVBench"
    api = HubApi()
    files = api.get_dataset_files(repo_id=repo, revision="master")
    names = []
    for f in files:
        p = f.get("Path") or f.get("path") if isinstance(f, dict) else str(f)
        if p:
            names.append(p)
    vids = [n for n in names if n.lower().endswith(".mp4")]
    if max_videos and max_videos > 0:
        vids = vids[:max_videos]
    print(f"[lvbench] ModelScope: {len(vids)} mp4 to fetch", flush=True)
    dest_root = _lvbench_dir()
    cache = dataset_snapshot_download(
        repo, allow_patterns=vids, local_dir=str(dest_root / "_ms_cache"),
    )
    # Move/copy the mp4s into videos/ (flat, basename-matched by the loader).
    import glob
    vids_dir = dest_root / "videos"
    vids_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in glob.glob(str(Path(cache) / "videos" / "*.mp4")):
        target = vids_dir / Path(f).name
        if target.exists():
            continue
        shutil.copyfile(f, target)
        n += 1
    print(f"[lvbench] ModelScope: {n} new videos -> {vids_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Download LVBench benchmark")
    ap.add_argument("--source", choices=["youtube", "modelscope"], default="modelscope",
                    help="video source; modelscope is preferred (no YouTube / yt-dlp).")
    ap.add_argument("--download-videos", action="store_true",
                    help="also fetch videos (ModelScope by default; YouTube only with --source youtube)")
    ap.add_argument("--max-videos", type=int, default=10)
    ap.add_argument("--skip-annotations", action="store_true")
    args = ap.parse_args()

    if not args.skip_annotations:
        build_annotations()
    if args.download_videos:
        if args.source == "modelscope":
            download_videos_modelscope(max_videos=args.max_videos)
        else:
            print(
                "[lvbench] WARNING: --source youtube uses yt-dlp. YouTube terms "
                "generally forbid downloading; you must have the right to fetch "
                "these videos. Prefer --source modelscope.",
                flush=True,
            )
            download_videos(max_videos=args.max_videos)
    else:
        print(
            "[lvbench] annotations ready. No videos fetched. Re-run with "
            "--download-videos (default --source modelscope), or copy mp4s named "
            f"<youtube_id>.mp4 into {_lvbench_dir() / 'videos'}.",
            flush=True,
        )


if __name__ == "__main__":
    main()
