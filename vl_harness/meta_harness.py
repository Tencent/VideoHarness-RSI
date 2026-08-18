"""Autonomous evolution loop for memory systems.

Validation-only during evolution. Test evaluation requires explicit finalization.
Uses claude_wrapper + meta-harness skill to propose new memory systems.

    uv run python meta_harness.py --iterations 20 --fresh
    uv run python meta_harness.py --iterations 10 --run-name my-run
    uv run python meta_harness.py --run-name my-run --test
"""

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import claude_wrapper
from benchmark import get_model_short_name, load_results

EVOLVE_DIR = Path(__file__).parent
CONFIG_PATH = EVOLVE_DIR / "config_k40.yaml"
AGENTS_DIR = EVOLVE_DIR / "agents"
BASELINE_FILES = {
    "__init__.py",
    "uniform_frames_no_memory.py",
    "uniform_frames_qwen3_paper_prompt.py",
    "dense_caption_text_rag.py",
    "keyframe_image_rag.py",
    "hybrid_router_rag.py",
    "worldmm_style.py",
    "homer_style.py",
    "pilot_uniform_k.py",
    "aks.py",
    "embed_navigate_hybrid_iter2.py",
    "timestamped_aks_iter5.py",
}

# These are updated per-run if --run-name is set
LOGS_DIR = EVOLVE_DIR / "logs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
RESULTS_DIR = LOGS_DIR / "results"
FINALIZED = LOGS_DIR / "finalized.json"

PROPOSER_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Agent",
    "Write",
    "Edit",
    "Bash",
]

# Model passed to the Claude Code CLI. Override with PROPOSER_MODEL.
PROPOSER_MODEL = os.environ.get("PROPOSER_MODEL", "claude-sonnet-4-6")

# Active search space for the current run (set in run_evolve); recorded per
# candidate in the ledger.
_ACTIVE_SEARCH_SPACE = "full"

_interrupted = False

# ── ANSI colors ──────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _pct(val):
    s = f"{val:.1f}%"
    if val >= 60:
        return _green(s)
    elif val >= 40:
        return _yellow(s)
    return _red(s)


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


# Platform-safe ceiling for select()-based timeouts. proc.communicate() is
# backed by select() and raises OverflowError if the timeout exceeds ~INT_MAX/1000
# seconds (~2.1e6 s). We poll in bounded chunks so an arbitrarily large logical
# timeout still works without ever passing an over-large value to communicate().
_SELECT_TIMEOUT_CEIL = 2_000_000  # ~23 days; safely under the select() limit


def run_cmd(cmd, timeout=2592000, cwd=None):  # 临时无上限（≈30天）。原值 21600。
    """Wraps subprocess.run; returns CompletedProcess with returncode=124 on timeout.

    Runs the child in its own process group so a timeout kills the whole tree
    (uv -> python benchmark.py -> python -m vl_harness.inner_loop). Plain
    subprocess.run only kills the direct child, leaving orphaned inner_loop
    processes that keep hammering the same VLM endpoint and slow down (or
    silently corrupt) every subsequent candidate evaluation.

    To avoid OverflowError from select()'s max-timeout clamp, we poll with a
    chunked timeout (<= _SELECT_TIMEOUT_CEIL) until the real deadline elapses.

    `timeout=None` means "no caller-supplied cap" and falls back to the default
    (callers such as run_benchmark always forward the kwarg, so None must not
    reach the arithmetic below).
    """
    if timeout is None:
        timeout = 2592000
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout)
            chunk = min(remaining, _SELECT_TIMEOUT_CEIL)
            try:
                out, err = proc.communicate(timeout=chunk)
                return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
            except subprocess.TimeoutExpired:
                # This slice expired but the child is still running -> keep polling.
                continue
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        return subprocess.CompletedProcess(
            cmd, returncode=124, stdout="", stderr=f"Timed out after {timeout}s"
        )


def run_benchmark(args, timeout=None):
    return run_cmd(
        [
            "uv",
            "run",
            "python",
            "benchmark.py",
            "--logs-dir",
            str(LOGS_DIR),
            "--results-dir",
            str(RESULTS_DIR),
        ]
        + args,
        cwd=str(EVOLVE_DIR),
        timeout=timeout,
    )


def run_memory_benchmarks(names, extra_args, concurrency, phase, timeout=None):
    """Run isolated memory-system benchmarks concurrently.

    Each system writes to a distinct log/result directory, while the shared Qwen
    gateway bounds actual GPU concurrency. Train stays ordered inside each system.
    `timeout` (seconds) caps each candidate bench; a timed-out candidate returns
    returncode 124 and is reported as TIMEOUT (skipped from the frontier).
    """
    names = list(names)
    workers = max(1, min(int(concurrency), len(names)))

    def run_one(name):
        t0 = time.time()
        result = run_benchmark(["--memory", name] + extra_args, timeout=timeout)
        return name, result, time.time() - t0

    outcomes = {}
    if workers == 1:
        for name in names:
            name, result, elapsed = run_one(name)
            outcomes[name] = (result, elapsed)
            if result.returncode == 0:
                status = _green("OK")
            elif result.returncode == 124:
                status = _red("TIMEOUT")
            else:
                status = _red("FAIL")
            print(f"      {status} {phase}/{name} ({_elapsed(elapsed)})", flush=True)
        return outcomes

    print(
        f"    launching {len(names)} {phase} job(s) with candidate_concurrency={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, result, elapsed = future.result()
            except Exception as exc:
                print(f"      {_red('FAIL')} {phase}/{name}: {exc}", flush=True)
                continue
            outcomes[name] = (result, elapsed)
            if result.returncode == 0:
                status = _green("OK")
            elif result.returncode == 124:
                status = _red("TIMEOUT")
            else:
                status = _red("FAIL")
            print(f"      {status} {phase}/{name} ({_elapsed(elapsed)})", flush=True)
    return outcomes


def recent_axes_union(k=3):
    """Return sorted union of axis tags used in the last k iterations.

    Reads EVOLUTION_SUMMARY (logs/<run>/evolution_summary.jsonl).  If the file
    does not exist, returns [].  Uses the `components` field when present,
    otherwise falls back to the `axis` field (skipping the generic values
    "exploitation" / "exploration").
    """
    if not EVOLUTION_SUMMARY.exists():
        return []
    by_iter = {}
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        it = row.get("iteration")
        if it is None:
            continue
        by_iter.setdefault(it, []).append(row)

    axes = set()
    for it in sorted(by_iter.keys(), reverse=True)[:k]:
        for row in by_iter[it]:
            comps = row.get("components")
            if comps:
                axes.update(str(x) for x in comps)
            else:
                ax = row.get("axis")
                if ax and ax not in ("exploitation", "exploration"):
                    axes.add(str(ax))
    return sorted(axes)


def render_task_prompt(iteration, num_datasets, search_space="full"):
    """Build the prompt for the proposer Claude session."""
    recent_axes = ", ".join(recent_axes_union()) or "(none yet)"
    if search_space == "text_only":
        space_note = (
            "## SEARCH SPACE: TEXT-ONLY (controlled ablation)\n"
            "You MUST restrict every candidate to TEXTUAL memory and text->text "
            "retrieval ONLY. HARD CONSTRAINTS:\n"
            "- Do NOT show any image/frame to the VLM at answer time "
            "(no self.render_frames of real pixels; no image content parts).\n"
            "- Do NOT use image or visual-embedding retrieval "
            "(no self.embed_images-based ranking to select what the VLM sees).\n"
            "- Memory may be built by captioning frames at INGEST time, but at "
            "ANSWER time only textual captions may be placed in the prompt.\n"
            "This emulates a text-domain harness; it is the control condition.\n\n"
        )
    else:
        space_note = (
            "## SEARCH SPACE: FULL (all 7 video axes, cross-modal allowed)\n"
            "You MAY use any modality: caption text, keyframe images, visual "
            "embeddings, structured memory, and cross-modal retrieval routing "
            "(text->text / text->image / image->image / fusion). Prefer changing a "
            "fundamental mechanism over tuning constants.\n\n"
            "## ARCHITECTURAL JUMP HINT (do NOT ignore)\n"
            "If the current frontier system is a SINGLE-PASS uniform-frame system "
            "with NO independent build_memory stage (i.e. it just samples frames "
            "at answer time and dumps them into the VLM), then increment-only "
            "mutations (adjusting frame density, tweaking prompts, adding a "
            "verification pass) exhaust their headroom quickly. In that regime you "
            "MUST also explore an ARCHITECTURAL BRANCH JUMP: at least ONE of your "
            "3 candidates should introduce a full 'build_memory + retrieve + "
            "answer' pipeline, where build_memory does a UNIFORM 320-frame ingest "
            "and writes a persistent memory (textual captions, keyframe "
            "embeddings, or a hybrid), and answer_question RETRIEVES from that "
            "memory (embedding kNN / captioned keyword match / cross-modal "
            "router) for questions that lack explicit timestamps. Reference "
            "existing implementations of this pattern in agents/ (e.g. "
            "hybrid_router_rag.py, dense_caption_text_rag.py, keyframe_image_rag.py, "
            "episodic_narrative_memory.py) -- do NOT re-derive them; instead, "
            "COMBINE their memory/retrieval mechanism with mechanisms proven to "
            "work on the current frontier (e.g. temporal_redistribute's "
            "timestamp-aware allocation for the SUBSET of questions that DO carry "
            "explicit timestamps). Mark such a candidate axis=\"exploration\".\n\n"
        )
    return (
        f"Run iteration {iteration} of the evolution loop. There are {num_datasets} datasets.\n\n"
        + space_note
        + f"## RECENT AXIS COVERAGE (last 3 iterations): {recent_axes}\n\n"
        + "## INGEST POLICY (adaptive multi-pass)\n"
        + "You are given ONLY the raw video (a VideoStream). The single HARD "
        + "constraint is: a single VLM request may contain AT MOST 320 frames "
        + "(the framework raises if you exceed it -- see render_frames). Within "
        + "that cap, YOU decide the ingest rate, HOW MANY passes to sample, and "
        + "HOW MANY frames per pass via self.sample_ingest_frames(video, fps=..., "
        + "max_frames<=320). For long videos, prefer sampling in MULTIPLE passes "
        + "and aggregating what you see into a memory (text summaries, keyframes, "
        + "embeddings, or a hybrid) in build_memory, then answer from that memory "
        + "-- rather than a single uniform dump that misses key moments. Do not "
        + "use a fixed 32/48/64-frame pool as a default; small top-k values are "
        + "answer-time only.\n\n"
        + "You MUST satisfy the HARD EXPLORATION CONSTRAINT in SKILL.md: at least 1 of "
        + "your 3 candidates must move an axis NOT in the above list, with axis=\"exploration\".\n\n"
        + f"## Run directories\n"
        + f"All logs and results for this run are under `{LOGS_DIR}/`.\n"
        + f"- `{EVOLUTION_SUMMARY}` -- past results\n"
        + f"- `{FRONTIER_VAL}` -- frontier\n"
        + f"- `{LOGS_DIR / 'reports'}/` -- post-eval reports\n"
        + f"- Write pending_eval.json to: `{PENDING_EVAL}`"
    )


def count_iterations_from_summary():
    """Highest iteration number in evolution_summary.jsonl (for resume)."""
    if not EVOLUTION_SUMMARY.exists():
        return 0
    max_iter = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter


def propose_claude(task_prompt, iteration, timeout=2400):
    """Returns True if candidates were produced (pending_eval.json exists).

    Uses the Claude Code CLI via `claude_wrapper` with subscription auth
    (Claude Pro/Max login), NOT an API key. We pop ANTHROPIC_API_KEY so the
    CLI falls back to the logged-in subscription session, which avoids
    API rate limits.
    """
    os.environ.pop("CLAUDECODE", None)
    # Strip API key so claude CLI uses subscription auth (avoids rate limits)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    result = claude_wrapper.run(
        prompt=task_prompt,
        model=PROPOSER_MODEL,
        allowed_tools=PROPOSER_ALLOWED_TOOLS,
        skills=[str(EVOLVE_DIR / ".claude/skills/vl-harness")],
        cwd=str(EVOLVE_DIR),
        log_dir=str(LOGS_DIR / "claude_sessions"),
        name=f"iter{iteration}",
        timeout_seconds=timeout,
        effort="max",
        # Silence per-tool-call progress on the terminal; full traces are still
        # written to logs/<run>/claude_sessions/. Keeps stdout to candidate
        # names + benchmarking results only.
        progress=False,
    )
    # Restore API key
    if saved_key:
        os.environ["ANTHROPIC_API_KEY"] = saved_key
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    # Stay silent on success: the main loop prints the candidate names, and
    # full proposer traces are saved under logs/<run>/claude_sessions/.
    return PENDING_EVAL.exists()


def _snapshot_agent_mtimes():
    """Return {name: mtime} for every *.py in AGENTS_DIR (baselines included).

    Used to detect same-name collisions across iterations/runs: any candidate
    whose file mtime does not advance during a propose step was NOT written by
    this iteration and would silently reuse (or overwrite) a pre-existing file.
    """
    snapshot = {}
    if AGENTS_DIR.exists():
        for f in AGENTS_DIR.glob("*.py"):
            try:
                snapshot[f.stem] = f.stat().st_mtime
            except OSError:
                pass
    return snapshot


def dedupe_candidates(candidates, pre_snapshot):
    """Drop same-name duplicates and pre-existing (non-refreshed) collisions.

    - Within one iteration: keep the first occurrence of each name, drop later
      duplicates (proposer sometimes emits the same snake_case twice; the on-disk
      file would be overwritten and evaluation results would collapse to one).
    - Across iterations/runs: if `agents/<name>.py` already exists and its mtime
      did not advance during the propose step, the proposer did NOT write a new
      file for it -- treat as a naming collision and reject.
    """
    kept, dropped = [], []
    seen = set()
    post_snapshot = _snapshot_agent_mtimes()
    for c in candidates:
        name = c.get("name")
        if not name:
            dropped.append((c, "missing name"))
            continue
        if name in seen:
            dropped.append((c, "duplicate name within iteration"))
            continue
        pre_mtime = pre_snapshot.get(name)
        post_mtime = post_snapshot.get(name)
        if pre_mtime is not None and (post_mtime is None or post_mtime <= pre_mtime):
            dropped.append((c, f"name collides with existing agents/{name}.py"))
            continue
        seen.add(name)
        kept.append(c)
    for c, reason in dropped:
        print(f"    {_yellow('SKIP')} {c.get('name', '<no-name>')}: {reason}")
    return kept


def validate_candidates(candidates):
    """Import-check each candidate. Returns list of valid candidates."""
    valid = []
    for c in candidates:
        name = c["name"]
        result = run_cmd(
            [
                "uv",
                "run",
                "python",
                "-c",
                f"from vl_harness.agents.{name} import *; print('OK')",
            ],
            cwd=str(EVOLVE_DIR.parent),
            timeout=30,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print(f"    {_green('OK')} {name}")
            valid.append(c)
        else:
            print(f"    {_red('FAIL')} {name}")
            if result.stderr:
                print(f"      {_dim(result.stderr[:200])}")
    return valid


def validate_text_only_harnesses(candidates):
    """Reject answer-time visual paths in the text_only control condition."""
    forbidden_markers = ("render_frames(", "embed_images(", '"type": "image"', "'type': 'image'")
    valid = []
    for candidate in candidates:
        name = candidate["name"] if isinstance(candidate, dict) else candidate
        path = AGENTS_DIR / f"{name}.py"
        try:
            source = path.read_text()
        except OSError as exc:
            print(f"    {_red('TEXT-ONLY FAIL')} {name}: cannot read {path}: {exc}")
            continue
        violations = [marker for marker in forbidden_markers if marker in source]
        if violations:
            print(
                f"    {_red('TEXT-ONLY FAIL')} {name}: forbidden answer-time visual path "
                f"{', '.join(violations)}"
            )
            continue
        valid.append(candidate)
    return valid


def update_evolution_summary(
    iteration,
    candidates,
    val_scores,
    propose_time=None,
    bench_time=None,
    wall_time=None,
):
    """Append one JSONL row per candidate to evolution_summary.jsonl."""

    def _detect_execution_error(name):
        """Return True if this candidate crashed during val (not benchmarked at all,
        or ≥80% parse_fail, or >0 error entries).
        """
        # 1) Look for val.json under logs/<dataset>/<name>/<model>/val.json
        val_files = list(LOGS_DIR.rglob(f"*/{name}/*/val.json"))
        if not val_files:
            return True  # never wrote val at all → crashed
        for vf in val_files:
            try:
                d = json.loads(vf.read_text())
            except Exception:
                return True
            rs = d.get("results") or []
            if not rs:
                return True
            n = len(rs)
            pf = sum(1 for r in rs if r.get("parse_fail"))
            err = sum(1 for r in rs if r.get("error"))
            if err > 0:
                return True
            if pf / n >= 0.8:
                return True
        return False

    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    pareto = frontier.get("_pareto", [])
    best_val = pareto[0].get("val_accuracy", 0) if pareto else 0

    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            avg_val = val_scores.get(name, 0)
            crashed = _detect_execution_error(name)
            row = {
                "iteration": iteration,
                "system": name,
                "avg_val": round(avg_val, 1) if not crashed else None,
                "axis": c.get("axis", "?"),
                "hypothesis": c.get("hypothesis", ""),
                "delta": round(avg_val - best_val, 1) if (best_val and not crashed) else None,
                "outcome": (
                    "execution_error"
                    if crashed
                    else (
                        f"{avg_val:.1f}% ({avg_val - best_val:+.1f})"
                        if avg_val > 0
                        else "failed"
                    )
                ),
            }
            # Record which system the proposer chose as its base ("parent").
            # Used by analyze_parent_selection.py to measure implicit greedy bias.
            if "base_system" in c:
                row["base_system"] = c["base_system"]
            # Also record the champion-at-proposal-time so analyzer doesn't have
            # to re-derive it (avoids errors when candidates from the same iter
            # influence each other's frontier value).
            row["champion_at_proposal"] = pareto[0].get("system") if pareto else None
            row["champion_val_at_proposal"] = round(best_val, 1) if best_val else None
            row["search_space"] = _ACTIVE_SEARCH_SPACE
            # Real visual-token cost from this candidate's val.json (Pareto currency).
            try:
                _vt = None
                for _k, _v in load_results(LOGS_DIR, "val.json").items():
                    if _k[2] == name and _v.get("visual_tokens"):
                        _vt = _v.get("visual_tokens")
                        break
                row["visual_tokens"] = _vt
            except Exception:
                row["visual_tokens"] = None
            if "components" in c:
                row["components"] = c["components"]
            if i == 0 and wall_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1),
                    "wall": round(wall_time, 1),
                }
            f.write(json.dumps(row) + "\n")


def fresh_start():
    """Reset only the active run's outputs; never delete shared agent source files."""
    if LOGS_DIR.exists():
        shutil.rmtree(LOGS_DIR)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {_green('Fresh start')}: cleared only run logs at {LOGS_DIR}")


def finalize_run(baselines, datasets, model_short, candidate_concurrency=1):
    """Evaluate test once, after freezing this run against further evolution."""
    if not FRONTIER_VAL.exists():
        print(f"ERROR: no validation frontier for run at {LOGS_DIR}")
        raise SystemExit(1)

    if FINALIZED.exists():
        state = json.loads(FINALIZED.read_text())
        if state.get("status") == "complete":
            print(f"Run already finalized: {LOGS_DIR.name}")
            result = run_benchmark(["--results", "--test"])
            if result.stdout:
                print(result.stdout)
            return

    frontier = json.loads(FRONTIER_VAL.read_text())
    pareto = frontier.get("_pareto", [])
    test_systems = set(baselines)
    test_systems.update(entry["system"] for entry in pareto)
    for key, value in frontier.items():
        if not key.startswith("_") and isinstance(value, dict):
            if "best_system" in value:
                test_systems.add(value["best_system"])

    FINALIZED.write_text(
        json.dumps(
            {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "systems": sorted(test_systems),
            },
            indent=2,
        )
    )

    print(f"\n{_ts()} {_bold('Phase Final: Test evaluation')}")
    test_outcomes = run_memory_benchmarks(
        sorted(test_systems),
        ["--test"],
        candidate_concurrency,
        "test",
    )
    failed = len(test_outcomes) != len(test_systems) or any(
        result.returncode != 0 for result, _ in test_outcomes.values()
    )

    frontier_result = run_benchmark(["--frontier", "--test", "--model", model_short])
    failed = failed or frontier_result.returncode != 0

    result = run_benchmark(["--results", "--test"])
    failed = failed or result.returncode != 0
    if result.stdout:
        print(result.stdout)

    test_results = load_results(RESULTS_DIR, "test.json")
    missing = [
        (model_short, dataset, system)
        for system in test_systems
        for dataset in datasets
        if (model_short, dataset, system) not in test_results
    ]
    if missing:
        failed = True
        print(f"Missing {len(missing)} complete test result group(s).")

    if failed:
        print("Test finalization incomplete. Fix failures, then rerun --test.")
        raise SystemExit(1)

    state = json.loads(FINALIZED.read_text())
    state["status"] = "complete"
    state["completed_at"] = datetime.now().isoformat()
    FINALIZED.write_text(json.dumps(state, indent=2))
    print(f"\n{_ts()} {_bold('Test finalization complete.')}")


def run_evolve(args):
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    global RESULTS_DIR, FINALIZED

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    run_cfg = cfg.get("run", {}) or {}
    args.iterations = (
        args.iterations if args.iterations is not None else int(run_cfg.get("iterations", 20))
    )
    args.skip_baseline = args.skip_baseline or bool(run_cfg.get("skip_baseline", False))
    args.fresh = args.fresh or bool(run_cfg.get("fresh", False))
    if args.test and args.fresh:
        raise SystemExit("--test cannot be combined with run.fresh or --fresh")
    datasets = cfg["datasets"]
    search_space = cfg.get("search_space", "full")
    candidate_concurrency = max(1, int(cfg.get("evolution", {}).get("candidate_concurrency", 1)))
    global _ACTIVE_SEARCH_SPACE
    _ACTIVE_SEARCH_SPACE = search_space

    config_model_ids = [m["model"] for m in cfg.get("models", [])]
    if args.model not in config_model_ids:
        print(f"ERROR: --model {args.model} not in {CONFIG_PATH.name}: {config_model_ids}")
        sys.exit(1)

    model_short = get_model_short_name(args.model)

    # Isolate run outputs under run-name subdirs
    if args.run_name:
        run_name = args.run_name
    elif run_cfg.get("name"):
        run_name = str(run_cfg["name"])
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
    RESULTS_DIR = LOGS_DIR / "results"
    FINALIZED = LOGS_DIR / "finalized.json"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    baselines = list(cfg["memory_systems"]["baselines"])
    # Optional paper-protocol seed (Appendix B.8 prompt + thinking). Enabled by
    # default as a high-quality starting baseline; set false to disable.
    if cfg["memory_systems"].get("include_paper_prompt_seed", False):
        _paper_seed = "uniform_frames_qwen3_paper_prompt"
        if _paper_seed not in baselines:
            baselines.append(_paper_seed)
    if search_space == "text_only":
        checked_baselines = validate_text_only_harnesses(baselines)
        if len(checked_baselines) != len(baselines):
            raise SystemExit(
                "text_only requires Phase 0 baselines without answer-time visual paths"
            )
    if args.test:
        finalize_run(baselines, datasets, model_short, candidate_concurrency)
        return

    if (
        FINALIZED.exists()
        and json.loads(FINALIZED.read_text()).get("status") == "complete"
    ):
        print(
            f"ERROR: run '{run_name}' is finalized; use a new --run-name to continue evolution"
        )
        raise SystemExit(1)

    if args.fresh:
        fresh_start()

    print(
        f"{_ts()} {_bold('Evolution (memory systems)')}  "
        f"run={_cyan(run_name)}  model={_cyan(args.model)}  "
        f"iters={args.iterations}  datasets={datasets}  space={search_space}"
    )

    # ── Phase 0: Baselines ─────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n{_ts()} {_bold('Phase 0: Baselines')}  systems={baselines}")
        baseline_failed = False
        for bl in baselines:
            if _interrupted:
                break
            print(f"  {_ts()} benchmarking {_bold(bl)}...", flush=True)
            t0 = time.time()
            result = run_benchmark(["--memory", bl])
            elapsed = time.time() - t0
            if result.returncode != 0:
                baseline_failed = True
                detail = (result.stderr or result.stdout).strip()
                print(f"    {_red('FAIL')} {bl}: {detail[:500]}")
            else:
                print(f"    {_green('OK')} ({_elapsed(elapsed)})")

        if baseline_failed:
            print("Baseline evaluation failed; fix the reported error before evolution.")
            raise SystemExit(1)

        run_benchmark(["--frontier", "--model", model_short])

        # Show baseline results
        results = load_results(LOGS_DIR, "val.json")
        for bl in baselines:
            accs = [
                results[k]["accuracy"] * 100
                for ds in datasets
                for k in [(model_short, ds, bl)]
                if k in results and results[k].get("accuracy") is not None
            ]
            if accs:
                avg = sum(accs) / len(accs)
                print(f"    {_bold(bl)}: avg_val={_pct(avg)}")

    # ── Phase 1..N: Evolution ──────────────────────────────────
    start_iteration = count_iterations_from_summary() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()

        # Show frontier status
        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        pareto = frontier.get("_pareto", [])
        best_val = pareto[0].get("val_accuracy", 0) if pareto else 0
        best_sys = pareto[0].get("system", "none") if pareto else "none"

        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  "
            f"frontier={best_sys} @ {_pct(best_val * 100 if best_val <= 1 else best_val)}"
        )
        print(f"{'─' * 60}")

        task_prompt = render_task_prompt(iteration, len(datasets), search_space)

        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        # Snapshot agent file mtimes so we can detect same-name collisions
        # (proposer must write a fresh file for every candidate name).
        pre_snapshot = _snapshot_agent_mtimes()

        # Propose
        propose_start = time.time()
        print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
        ok = propose_claude(task_prompt, iteration, timeout=args.propose_timeout)
        propose_time = time.time() - propose_start

        if not ok:
            print(
                f"  {_red('FAIL')} proposer returned no candidates after {_elapsed(propose_time)}"
            )
            continue

        candidates = json.loads(PENDING_EVAL.read_text()).get("candidates", [])
        print(
            f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

        # Reject in-iteration duplicates and same-name collisions with existing
        # on-disk agents (files whose mtime did not advance during propose).
        candidates = dedupe_candidates(candidates, pre_snapshot)
        if not candidates:
            print(f"  {_red('0 unique')} candidates after dedupe, skipping iteration")
            continue

        # Validate
        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        valid_candidates = validate_candidates(candidates)
        if search_space == "text_only":
            valid_candidates = validate_text_only_harnesses(valid_candidates)

        if not valid_candidates:
            print(
                f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
            )
            update_evolution_summary(
                iteration, candidates, {}, propose_time=propose_time
            )
            continue
        print(
            f"  {_green(f'{len(valid_candidates)} valid')} out of {len(candidates)} candidates"
        )

        # Benchmark
        bench_start = time.time()
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid_candidates)} system(s) x {len(datasets)} datasets"
        )
        if not _interrupted:
            run_memory_benchmarks(
                [c["name"] for c in valid_candidates],
                [],
                candidate_concurrency,
                "val",
                timeout=args.candidate_timeout,
            )
        bench_time = time.time() - bench_start

        run_benchmark(["--frontier", "--model", model_short])

        # Compute scores and show results
        val_scores = {}
        results = load_results(LOGS_DIR, "val.json")
        for c in valid_candidates:
            name = c["name"]
            accs = [
                results[k]["accuracy"] * 100
                for ds in datasets
                for k in [(model_short, ds, name)]
                if k in results and results[k].get("accuracy") is not None
            ]
            val_scores[name] = sum(accs) / len(accs) if accs else 0
            delta = val_scores[name] - (best_val * 100 if best_val <= 1 else best_val)
            delta_str = f"{delta:+.1f}"
            delta_colored = (
                _green(delta_str)
                if delta > 0
                else (_red(delta_str) if delta < 0 else _dim(delta_str))
            )
            print(
                f"    {_bold(name)}: avg_val={_pct(val_scores[name])}  delta={delta_colored}"
            )

        wall_time = time.time() - iter_start
        update_evolution_summary(
            iteration,
            valid_candidates,
            val_scores,
            propose_time=propose_time,
            bench_time=bench_time,
            wall_time=wall_time,
        )

        # Show iteration summary
        improved = any(
            v > (best_val * 100 if best_val <= 1 else best_val)
            for v in val_scores.values()
        )
        status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )

    if _interrupted:
        return

    print(f"\n{_ts()} {_bold('Validation evolution complete.')}")
    print(
        f"Finalize once with: uv run python meta_harness.py --run-name {run_name} --test"
    )


def main():
    parser = argparse.ArgumentParser(description="Evolution loop for memory systems")
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Iteration count (default: run.iterations from the selected config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_k40.yaml",
        help="Config file to use (default: config_k40.yaml)",
    )
    args, remaining = parser.parse_known_args()

    global CONFIG_PATH
    CONFIG_PATH = EVOLVE_DIR / args.config
    os.environ["VL_HARNESS_CONFIG"] = str(CONFIG_PATH)

    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
    _default_model = _cfg["models"][0]["model"] if _cfg.get("models") else None

    parser.add_argument(
        "--model",
        default=_default_model,
        help=f"Solver model (default: {_default_model})",
    )
    parser.add_argument(
        "--propose-timeout",
        type=int,
        default=2400,
        help="Timeout per propose step (default: 2400s)",
    )
    parser.add_argument(
        "--candidate-timeout",
        type=int,
        default=3600,
        help="Timeout per candidate val bench in seconds; timed-out candidates are "
        "marked TIMEOUT and skipped from the frontier (default: 3600s = 1h)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for isolated output dirs. Auto-generated if not set.",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Clear proposed systems and reset logs"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Phase 0 baseline eval"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Finalize an existing named run with one held-out test evaluation",
    )
    args = parser.parse_args(remaining, args)

    if args.test and not args.run_name:
        parser.error("--test requires --run-name")
    if args.test and args.fresh:
        parser.error("--test cannot be combined with --fresh")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    run_evolve(args)


if __name__ == "__main__":
    main()
