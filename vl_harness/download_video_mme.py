"""Fetch the Video-MME benchmark for VL-Harness, evaluated **without subtitles**.

Source: HF dataset ``lmms-lab/Video-MME``.

Terms (upstream): academic research only; commercial use prohibited; do not
distribute, publish, copy, or modify Video-MME without prior approval. Video
copyright stays with the owners. This script writes a local cache under
``$VL_HARNESS_DATA``; do not commit those files.


- Annotations: ``videomme/test-00000-of-00001.parquet`` (2700 questions over ~900
  videos). The ``options`` column already carries an ``"A. "`` style prefix and
  ``answer`` is an option letter, so we strip the prefix and keep clean candidate
  text for the loader.
- Videos: ``videos_chunked_01..20.zip`` (~101 GB total). Videos are stored as
  ``<videoID>.mp4`` (a YouTube id), matched by basename by the loader.
- Subtitles: ``subtitle.zip`` is intentionally NOT downloaded. Video-MME has two
  eval conditions (with / without subtitles); this harness evaluates the
  **w/o sub** condition, so no subtitle text is ever fetched or injected.

On-disk layout produced::

    <VL_HARNESS_DATA>/video_mme/
        video_mme.json          # normalized annotations (list of dicts)
        videos/<videoID>.mp4    # extracted videos (flat, basename-matched)

Examples::

    # annotations only (tiny)
    python -m vl_harness.download_video_mme

    # annotations + all videos (~101 GB, background this)
    python -m vl_harness.download_video_mme --download-videos --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .loaders_real import _video_mme_dir

REPO = "lmms-lab/Video-MME"
ANNOTATION_PARQUET = "videomme/test-00000-of-00001.parquet"
VIDEO_ZIPS = [f"videos_chunked_{i:02d}.zip" for i in range(1, 21)]
_VID_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v")

_OPT_PREFIX = re.compile(r"^[A-Z]\s*[.)]\s*")


def _strip_opt_prefix(opt: str) -> str:
    return _OPT_PREFIX.sub("", str(opt)).strip()


def build_annotations() -> Path:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    out_dir = _video_mme_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[video_mme] downloading annotations {ANNOTATION_PARQUET} ...", flush=True)
    p = hf_hub_download(REPO, ANNOTATION_PARQUET, repo_type="dataset")
    rows = pq.read_table(p).to_pylist()
    anns = []
    for r in rows:
        opts = [_strip_opt_prefix(o) for o in (r.get("options") or [])]
        anns.append(
            {
                "video_name": r.get("videoID"),  # -> videos/<videoID>.mp4
                "question": r.get("question", ""),
                "candidates": opts,
                "answer": r.get("answer"),  # option letter
                "task_type": r.get("task_type", ""),
                "domain": r.get("domain"),
                "sub_category": r.get("sub_category"),
                "duration": r.get("duration"),
                "question_id": r.get("question_id"),
            }
        )
    out = out_dir / "video_mme.json"
    out.write_text(json.dumps(anns, ensure_ascii=False))
    n_vid = len({a["video_name"] for a in anns})
    print(f"[video_mme] wrote {len(anns)} questions / {n_vid} videos -> {out}", flush=True)
    return out


def _extract_zip_flat(zip_path: str, dest: Path) -> int:
    """Extract video files to dest/<basename>, flattening any internal dirs."""
    new = 0
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            if m.endswith("/"):
                continue
            base = Path(m).name
            if not base.lower().endswith(_VID_EXTS):
                continue
            target = dest / base
            if target.exists():
                continue
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            new += 1
    return new


def download_videos(workers: int = 4) -> None:
    from huggingface_hub import hf_hub_download

    dest = _video_mme_dir() / "videos"
    dest.mkdir(parents=True, exist_ok=True)

    def _one(zname: str):
        local = hf_hub_download(REPO, zname, repo_type="dataset")
        n = _extract_zip_flat(local, dest)
        return zname, n

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_one, z): z for z in VIDEO_ZIPS}
        for fut in as_completed(futs):
            z = futs[fut]
            try:
                _, n = fut.result()
                done += 1
                print(
                    f"[video_mme] {z}: +{n} videos "
                    f"({done}/{len(VIDEO_ZIPS)} zips done)",
                    flush=True,
                )
            except Exception as e:
                failed.append((z, str(e)))
                print(f"[video_mme] FAIL {z}: {e}", flush=True)
    n_local = len(list(dest.glob("*")))
    print(
        f"[video_mme] videos ready under {dest} "
        f"({n_local} files; {len(failed)} zips failed)",
        flush=True,
    )
    if failed:
        print("[video_mme] failed zips:", failed, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Video-MME benchmark (w/o subtitles)")
    ap.add_argument("--download-videos", action="store_true", help="also fetch videos (~101 GB)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-annotations", action="store_true")
    args = ap.parse_args()

    if not args.skip_annotations:
        build_annotations()
    if args.download_videos:
        download_videos(workers=args.workers)
    else:
        print(
            "[video_mme] annotations ready. No videos fetched. Re-run with "
            "--download-videos to pull the 20 video zips into "
            f"{_video_mme_dir() / 'videos'}.",
            flush=True,
        )


if __name__ == "__main__":
    main()
