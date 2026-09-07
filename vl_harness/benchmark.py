"""Sweep datasets x memory systems."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def load_config() -> dict:
    """Load config from config.yaml or VL_HARNESS_CONFIG env override."""
    env_path = os.environ.get("VL_HARNESS_CONFIG")
    if env_path:
        config_path = Path(env_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent.parent / "configs" / config_path
    else:
        config_path = Path(__file__).parent.parent / "configs" / "config_k40.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


# Load config
_CONFIG = load_config()
DATASETS = _CONFIG["datasets"]
MODELS = _CONFIG["models"]  # List of {model, api_base} dicts
SEEDS = _CONFIG["benchmark"]["seeds"]
CONCURRENCY = _CONFIG["benchmark"]["concurrency"]
INNER_MAX_WORKERS = max(1, int(_CONFIG["inner_loop"].get("max_workers", 7)))
INNER_MAX_TOKENS = max(8, int(_CONFIG["inner_loop"].get("max_tokens", 1024)))
_DS_DEFAULTS = {k: _CONFIG["dataset"][k] for k in ("num_train", "num_val", "num_test")}
_DS_OVERRIDES = _CONFIG["dataset"].get("overrides", {})

DEFAULT_SEED = 42
_SKIP_MEMORY_FILES = {"__init__"}


def discover_all_memory_systems() -> list[tuple[str, str]]:
    """Auto-discover all memory system .py files on disk."""
    base = Path(__file__).parent
    systems = []
    for f in sorted((base / "agents").glob("*.py")):
        name = f.stem
        if name in _SKIP_MEMORY_FILES:
            continue
        systems.append((name, f"agents/{name}.py"))
    return systems


def get_dataset_sizes(dataset: str) -> tuple[int, int, int]:
    """Return (num_train, num_val, num_test) for a dataset, applying overrides."""
    o = _DS_OVERRIDES.get(dataset, {})
    return (
        o.get("num_train", _DS_DEFAULTS["num_train"]),
        o.get("num_val", _DS_DEFAULTS["num_val"]),
        o.get("num_test", _DS_DEFAULTS["num_test"]),
    )


def _sanitize_filename(desc: str) -> str:
    return re.sub(r"[^\w\-.]", "_", desc)


def _print_failure(desc: str, log_path: Path) -> None:
    print(f"\nFAILED: {desc}")
    print(f"Log: {log_path}")
    try:
        lines = log_path.read_text().strip().split("\n")
    except OSError:
        return
    for line in lines[-8:]:
        print(f"  {line[:120]}")


async def _run_with_retries(
    cmd: list[str],
    log_path: Path,
    max_retries: int = 2,
    timeout: float = 2592000,  # 临时无上限（≈30天）。原值 18000。跑完 emergent 实验后改回。
) -> bool:
    cmd_str = " ".join(cmd)
    log_path.write_text(f"command: {cmd_str}\n\n")

    for attempt in range(max_retries + 1):
        if attempt > 0:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\nretry {attempt}\n{'=' * 60}\n")

        with log_path.open("a", encoding="utf-8") as f:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=f,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                code = await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                code = 124
            f.write(f"\nexit={code}\n")

        if code == 0:
            return True

    return False


async def run_all_jobs(
    runs: list[tuple[str, list[str]]],
    logs_dir: Path,
    concurrency: int,
    max_retries: int = 2,
) -> list[tuple[str, bool]]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def run_one(idx: int, desc: str, cmd: list[str]) -> tuple[str, bool]:
        async with sem:
            log_path = logs_dir / f"{idx:02d}_{_sanitize_filename(desc)}.log"
            ok = await _run_with_retries(cmd, log_path, max_retries=max_retries)
            if not ok:
                _print_failure(desc, log_path)
            return desc, ok

    tasks = [
        asyncio.create_task(run_one(idx, desc, cmd))
        for idx, (desc, cmd) in enumerate(runs)
    ]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# New hierarchical directory structure
# ---------------------------------------------------------------------------


def run_dir(
    base: Path, dataset: str, memory: str, model: str, seed: int = DEFAULT_SEED
) -> Path:
    """Construct hierarchical run directory path.

    logs/{dataset}/{memory}/{model}/          (default seed)
    logs/{dataset}/{memory}/{model}_seed{N}/  (non-default seed)
    """
    leaf = model if seed == DEFAULT_SEED else f"{model}_seed{seed}"
    return base / dataset / memory / leaf


def parse_run_path(base: Path, filepath: Path) -> dict | None:
    """Parse (dataset, memory, model, seed) from a result file under base."""
    try:
        rel = filepath.parent.relative_to(base)
        parts = rel.parts
        if len(parts) != 3:
            return None
        dataset, memory, model_leaf = parts
        m = re.match(r"^(.+)_seed(\d+)$", model_leaf)
        if m:
            model = m.group(1)
            seed = int(m.group(2))
        else:
            model = model_leaf
            seed = DEFAULT_SEED
        return {"dataset": dataset, "memory": memory, "model": model, "seed": seed}
    except (ValueError, IndexError):
        return None


def _has_usable_result(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and isinstance(data.get("accuracy"), (int, float))


# ── Shared baseline cache (run-name independent) ──────────────────────────────
# Seed memory systems (baselines) are deterministic given
# (dataset, memory agent code, model, seed, temperature, dataset sizes). Their
# results do NOT depend on the evolution --run-name, so instead of caching them
# under logs/<run-name>/ we cache them once under logs/_shared_baselines/. This
# makes switching run-names (or restarting) never re-run an already-succeeded
# baseline. Evolved candidate systems are never baselines, so they are untouched.
SHARED_BASELINE_ROOT = Path(__file__).parent / "logs" / "_shared_baselines"
_SHARED_COPIED_FILES = ("val.json", "log.jsonl", "memory.json")


def _baseline_names() -> set[str]:
    ms = _CONFIG.get("memory_systems", {}) or {}
    names = set(ms.get("baselines", []) or [])
    if ms.get("include_paper_prompt_seed"):
        names.add("uniform_frames_qwen3_paper_prompt")
    return names


def _is_baseline(mem_name: str) -> bool:
    return mem_name in _baseline_names()


def _baseline_fingerprint(
    mem_path: str, temperature, n_train: int, n_val: int, n_test: int
) -> str:
    """Hash of everything that can change a baseline result beyond the parts
    already encoded in the run_dir layout (dataset, model, seed)."""
    h = hashlib.sha256()
    h.update(b"v1")
    h.update(str(temperature).encode())
    h.update(f"{n_train},{n_val},{n_test}".encode())
    agent = Path(__file__).parent / mem_path
    try:
        h.update(agent.read_bytes())
    except OSError:
        pass
    return h.hexdigest()[:16]


def _shared_baseline_dir(
    dataset: str, mem_name: str, model_name: str, seed: int,
    mem_path: str, temperature, n_train: int, n_val: int, n_test: int,
) -> Path:
    leaf = model_name if seed == DEFAULT_SEED else f"{model_name}_seed{seed}"
    fp = _baseline_fingerprint(mem_path, temperature, n_train, n_val, n_test)
    return SHARED_BASELINE_ROOT / fp / dataset / mem_name / leaf


def _try_reuse_shared_baseline(
    logs_dir: Path, dataset: str, mem_name: str, model_name: str, seed: int,
    mem_path: str, temperature, n_train: int, n_val: int, n_test: int,
) -> bool:
    """If a usable baseline result exists in the shared cache, copy it into the
    run-specific dir and return True so the caller skips evaluation."""
    if not _is_baseline(mem_name):
        return False
    shared_rd = _shared_baseline_dir(
        dataset, mem_name, model_name, seed, mem_path, temperature,
        n_train, n_val, n_test,
    )
    shared_val = shared_rd / "val.json"
    if not (shared_val.exists() and _has_usable_result(shared_val)):
        return False
    rd = run_dir(logs_dir, dataset, mem_name, model_name, seed)
    rd.mkdir(parents=True, exist_ok=True)
    for fn in _SHARED_COPIED_FILES:
        src = shared_rd / fn
        if src.exists():
            shutil.copy2(src, rd / fn)
    return True


def _sync_baselines_to_shared(
    logs_dir: Path, results_dir: Path, memory_systems, datasets, models, temperature,
) -> None:
    """Copy freshly-produced usable baseline results into the shared cache for
    reuse by future runs under a different --run-name."""
    for model_cfg in models:
        model = model_cfg["model"]
        model_name = get_model_short_name(model)
        for dataset in datasets:
            n_train, n_val, n_test = get_dataset_sizes(dataset)
            for mem_name, mem_path in memory_systems:
                if not _is_baseline(mem_name):
                    continue
                for seed in SEEDS:
                    shared_rd = _shared_baseline_dir(
                        dataset, mem_name, model_name, seed, mem_path,
                        temperature, n_train, n_val, n_test,
                    )
                    rd = run_dir(logs_dir, dataset, mem_name, model_name, seed)
                    val_file = rd / "val.json"
                    if val_file.exists() and _has_usable_result(val_file):
                        shared_rd.mkdir(parents=True, exist_ok=True)
                        for fn in _SHARED_COPIED_FILES:
                            s = rd / fn
                            if s.exists():
                                shutil.copy2(s, shared_rd / fn)
                    rd_results = run_dir(results_dir, dataset, mem_name, model_name, seed)
                    test_file = rd_results / "test.json"
                    if test_file.exists() and _has_usable_result(test_file):
                        shared_rd.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(test_file, shared_rd / "test.json")


def _migrate_shared_cache(src_dir: Path) -> None:
    """One-off: copy usable baseline results from an existing run dir into the
    shared baseline cache (pre-warming it before switching run-names)."""
    src_dir = Path(src_dir).resolve()
    if not src_dir.exists():
        print(f"Error: {src_dir} not found")
        return
    migrated = 0
    temperature = _CONFIG["inner_loop"].get("temperature")
    for ds_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        dataset = ds_dir.name
        n_train, n_val, n_test = get_dataset_sizes(dataset)
        for mem_name, mem_path in discover_all_memory_systems():
            if not _is_baseline(mem_name):
                continue
            mem_dir = ds_dir / mem_name
            if not mem_dir.exists():
                continue
            for model_dir in sorted(p for p in mem_dir.iterdir() if p.is_dir()):
                m = re.match(r"^(.+)_seed(\d+)$", model_dir.name)
                if m:
                    model_name, seed = m.group(1), int(m.group(2))
                else:
                    model_name, seed = model_dir.name, DEFAULT_SEED
                rd = run_dir(src_dir, dataset, mem_name, model_name, seed)
                val_file = rd / "val.json"
                if not (val_file.exists() and _has_usable_result(val_file)):
                    continue
                shared_rd = _shared_baseline_dir(
                    dataset, mem_name, model_name, seed, mem_path,
                    temperature, n_train, n_val, n_test,
                )
                shared_rd.mkdir(parents=True, exist_ok=True)
                for fn in _SHARED_COPIED_FILES:
                    s = rd / fn
                    if s.exists():
                        shutil.copy2(s, shared_rd / fn)
                rd_test = run_dir(src_dir / "results", dataset, mem_name, model_name, seed)
                test_file = rd_test / "test.json"
                if test_file.exists() and _has_usable_result(test_file):
                    shutil.copy2(test_file, shared_rd / "test.json")
                migrated += 1
                print(f"  migrated {dataset}/{mem_name}/{model_name}")
    print(f"Migrated {migrated} baseline result(s) into shared cache.")


def load_results(base_dir: Path, filename: str = "val.json") -> dict:
    """Load results from hierarchical dir structure.

    Globs base_dir/**/filename, parses path to extract (dataset, memory, model, seed).
    Returns complete seed groups as (model, dataset, memory) -> aggregate data.
    """
    grouped = defaultdict(dict)
    for filepath in base_dir.rglob(filename):
        parsed = parse_run_path(base_dir, filepath)
        if not parsed:
            continue
        try:
            data = json.loads(filepath.read_text())
            if not isinstance(data, dict):
                continue
            key = (parsed["model"], parsed["dataset"], parsed["memory"])
            grouped[key][parsed["seed"]] = data
        except (json.JSONDecodeError, KeyError):
            continue

    results = {}
    required_seeds = sorted(set(SEEDS))
    summed_fields = (
        "runtime_seconds",
        "llm_calls",
        "llm_input_tokens",
        "llm_output_tokens",
        "llm_total_tokens",
    )
    for key, by_seed in grouped.items():
        if any(seed not in by_seed for seed in required_seeds):
            continue
        seed_results = [by_seed[seed] for seed in required_seeds]
        aggregate = dict(seed_results[0])
        aggregate.pop("seed", None)
        aggregate["seeds"] = required_seeds

        try:
            if all("correct" in data and "total" in data for data in seed_results):
                aggregate["correct"] = sum(data["correct"] for data in seed_results)
                aggregate["total"] = sum(data["total"] for data in seed_results)
                aggregate["accuracy"] = (
                    aggregate["correct"] / aggregate["total"]
                    if aggregate["total"]
                    else 0.0
                )
            else:
                aggregate.pop("correct", None)
                aggregate.pop("total", None)
                aggregate["accuracy"] = sum(
                    data["accuracy"] for data in seed_results
                ) / len(seed_results)
            aggregate["memory_context_chars"] = sum(
                data.get("memory_context_chars", 0) for data in seed_results
            ) / len(seed_results)
            aggregate["visual_tokens"] = sum(
                data.get("visual_tokens", 0) for data in seed_results
            ) / len(seed_results)
            aggregate["num_frames"] = sum(
                data.get("num_frames", 0) for data in seed_results
            ) / len(seed_results)
            for field in summed_fields:
                aggregate[field] = sum(data.get(field, 0) for data in seed_results)
        except (KeyError, TypeError, ZeroDivisionError):
            continue
        results[key] = aggregate
    return results


def compute_pareto_frontier(
    points: list[tuple[str, float, int]],
) -> list[tuple[str, float, int]]:
    """Compute Pareto frontier for (name, accuracy, ctx_tokens).

    A point is Pareto-optimal if no other point is at least as accurate and no
    more expensive, with at least one strict improvement.
    Returns points sorted by accuracy descending.
    """
    pareto = [
        point
        for point in points
        if not any(
            other_acc >= point[1]
            and other_tokens <= point[2]
            and (other_acc > point[1] or other_tokens < point[2])
            for _, other_acc, other_tokens in points
        )
    ]
    return sorted(pareto, key=lambda x: (-x[1], x[2]))


def _inner_loop_command() -> list[str]:
    """Return a cwd-independent command prefix for inner-loop workers.

    Uses the current interpreter by default so it works without a uv venv. Set
    VL_INNER_CMD (space-separated) to override, e.g. "uv run python".

    Critical env vars are forwarded explicitly (with sane defaults) because the
    spawned worker runs under `env PYTHONPATH=...` and may not inherit them from
    the launch shell. Without VL_HARNESS_DATA the loader falls back to
    ~/.cache/vl-harness/data and raises FileNotFoundError on the lvbench
    annotations (see DATASETS.md). Without NO_PROXY the local VLM gateway
    (127.0.0.1:8080) is routed through an intercepting proxy.
    """
    import shlex
    import sys as _sys

    project_dir = Path(__file__).resolve().parent
    prefix = ["env", f"PYTHONPATH={project_dir.parent}"]
    # Data root: forward if set, else default to <project>/data so the worker
    # never silently uses the wrong (empty) ~/.cache/vl-harness/data.
    vhd = os.environ.get("VL_HARNESS_DATA") or str(project_dir.parent.parent / "data")
    prefix += [f"VL_HARNESS_DATA={vhd}"]
    # Local VLM gateway must bypass any HTTP proxy.
    no_proxy = os.environ.get("NO_PROXY") or "localhost,127.0.0.1"
    prefix += [f"NO_PROXY={no_proxy}", f"no_proxy={no_proxy}"]
    # HuggingFace token (used by some loaders/downloader paths).
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val is not None:
            prefix += [f"{var}={val}"]
    override = os.environ.get("VL_INNER_CMD")
    if override:
        prefix += shlex.split(override)
    else:
        prefix += [_sys.executable]
    return prefix + ["-m", "vl_harness.inner_loop"]


def print_results(results: dict, metric_label: str = "val", pareto_only: bool = False):
    """Print one table per model: harnesses as rows, datasets as columns.

    Cost column is *visual tokens* (the VL-Harness Pareto currency), not chars.
    A ``*`` marks harnesses on the accuracy-vs-visual-token Pareto frontier.
    """
    if not results:
        print("No results found")
        return

    memory_names = sorted(set(mem for _, _, mem in results.keys()))
    models_in_results = sorted(set(m for m, _, _ in results.keys()))
    target_models = [get_model_short_name(m["model"]) for m in MODELS]
    models_to_show = [m for m in models_in_results if m in target_models] or models_in_results

    for model_name in models_to_show:
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}  [{metric_label}]  (cost = avg visual tokens/question)")
        print("=" * 80)

        rows = []
        for mem in memory_names:
            accs = []
            vtoks = []
            cells = []
            for ds in DATASETS:
                data = results.get((model_name, ds, mem))
                if data:
                    acc = data.get("accuracy")
                    vtoks.append(data.get("visual_tokens", 0))
                    if acc is not None:
                        cells.append(f"{acc * 100:.1f}")
                        accs.append(acc * 100)
                    else:
                        cells.append("-")
                else:
                    cells.append("-")
                    vtoks.append(0)
            avg_acc = sum(accs) / len(DATASETS) if DATASETS else 0
            non_zero = [t for t in vtoks if t > 0]
            avg_vtok = int(sum(non_zero) / len(non_zero)) if non_zero else 0
            rows.append((avg_acc, mem, cells, avg_vtok))

        rows.sort(key=lambda x: x[0])

        pareto_points = [(mem, avg_acc, avg_vtok) for avg_acc, mem, _, avg_vtok in rows]
        pareto_set = {name for name, _, _ in compute_pareto_frontier(pareto_points)}

        short_names = [d[:12] for d in DATASETS]
        col_w = 14
        header = (
            f"{'harness':<32}"
            + "".join(f"{d:>{col_w}}" for d in short_names)
            + f"{'avg':>7}{'vis_tok':>10}"
        )
        print(header)
        print("-" * len(header))

        for avg_acc, mem, cells, avg_vtok in rows:
            if pareto_only and mem not in pareto_set:
                continue
            marker = " *" if mem in pareto_set else ""
            vt_str = f"{avg_vtok:,}" if avg_vtok > 0 else "-"
            print(
                f"{mem + marker:<32}"
                + "".join(f"{c:>{col_w}}" for c in cells)
                + f"{avg_acc:>7.1f}{vt_str:>10}"
            )

        pareto_rows = compute_pareto_frontier(pareto_points)
        if len(pareto_rows) >= 1:
            print("\n  Pareto frontier (accuracy vs visual tokens):")
            for n, a, t in pareto_rows:
                print(f"    {n} ({a:.1f}%, {t:,} vis_tok)")


def build_val_runs(
    logs_dir: Path,
    memory_systems: list[tuple[str, str]],
    datasets: list[str],
    models: list[dict],
    mode: str = "online",
    num_epochs: int = 1,
    temperature: float | None = None,
) -> tuple[list[tuple[str, list[str]]], int, int]:
    """Build (description, command) pairs for val runs that need to run."""
    runs = []
    num_done = 0
    for model_cfg in models:
        model = model_cfg["model"]
        api_base = model_cfg.get("api_base")
        model_name = get_model_short_name(model)
        for dataset in datasets:
            n_train, n_val, n_test = get_dataset_sizes(dataset)
            for mem_name, mem_path in memory_systems:
                for seed in SEEDS:
                    rd = run_dir(logs_dir, dataset, mem_name, model_name, seed)
                    val_file = rd / "val.json"

                    if val_file.exists() and _has_usable_result(val_file):
                        num_done += 1
                        continue

                    if _try_reuse_shared_baseline(
                        logs_dir, dataset, mem_name, model_name, seed,
                        mem_path, temperature, n_train, n_val, n_test,
                    ):
                        num_done += 1
                        continue

                    rd.mkdir(parents=True, exist_ok=True)
                    desc = f"val/{dataset}/{mem_name}/{model_name}"
                    cmd = _inner_loop_command() + [
                        "--memory",
                        mem_path,
                        "--dataset",
                        dataset,
                        "--seed",
                        str(seed),
                        "--model",
                        model,
                        "--mode",
                        mode,
                        "--val-output",
                        str(val_file),
                        "--save-memory",
                        str(rd / "memory.json"),
                        "--log",
                        str(rd / "log.jsonl"),
                    ]
                    cmd.extend(
                        [
                            "--num-train",
                            str(n_train),
                            "--num-val",
                            str(n_val),
                            "--num-test",
                            str(n_test),
                        ]
                    )
                    if api_base:
                        cmd.extend(["--api-base", api_base])
                    if mode == "offline" and num_epochs > 1:
                        cmd.extend(["--num-epochs", str(num_epochs)])
                    if temperature is not None:
                        cmd.extend(["--temperature", str(temperature)])
                    cmd.extend(
                        [
                            "--max-workers",
                            str(INNER_MAX_WORKERS),
                            "--max-tokens",
                            str(INNER_MAX_TOKENS),
                        ]
                    )
                    if val_file.exists():
                        cmd.append("--force")
                    runs.append((desc, cmd))
    random.shuffle(runs)
    return runs, len(runs), num_done


def build_test_runs(
    logs_dir: Path,
    results_dir: Path,
    memory_systems: list[tuple[str, str]],
    datasets: list[str],
    models: list[dict],
    mode: str = "online",
    temperature: float | None = None,
) -> tuple[list[tuple[str, list[str]]], int, int]:
    """Build (description, command) pairs for test runs that need to run."""
    runs = []
    num_done = 0
    for model_cfg in models:
        model = model_cfg["model"]
        api_base = model_cfg.get("api_base")
        model_name = get_model_short_name(model)
        for dataset in datasets:
            n_train, n_val, n_test = get_dataset_sizes(dataset)
            for mem_name, mem_path in memory_systems:
                for seed in SEEDS:
                    rd_results = run_dir(
                        results_dir, dataset, mem_name, model_name, seed
                    )
                    test_file = rd_results / "test.json"

                    if test_file.exists() and _has_usable_result(test_file):
                        num_done += 1
                        continue

                    if _is_baseline(mem_name):
                        shared_rd = _shared_baseline_dir(
                            dataset, mem_name, model_name, seed, mem_path,
                            temperature, n_train, n_val, n_test,
                        )
                        shared_test = shared_rd / "test.json"
                        if shared_test.exists() and _has_usable_result(shared_test):
                            rd_results.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(shared_test, test_file)
                            rd_logs = run_dir(
                                logs_dir, dataset, mem_name, model_name, seed
                            )
                            if not (rd_logs / "memory.json").exists():
                                sm = shared_rd / "memory.json"
                                if sm.exists():
                                    rd_logs.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(sm, rd_logs / "memory.json")
                            num_done += 1
                            continue

                    # Need saved memory from val run
                    rd_logs = run_dir(logs_dir, dataset, mem_name, model_name, seed)
                    memory_file = rd_logs / "memory.json"
                    if not memory_file.exists():
                        print(
                            f"  WARNING: no memory.json for {dataset}/{mem_name}/{model_name} (run val first)"
                        )
                        continue

                    rd_results.mkdir(parents=True, exist_ok=True)
                    desc = f"test/{dataset}/{mem_name}/{model_name}"
                    cmd = _inner_loop_command() + [
                        "--memory",
                        mem_path,
                        "--dataset",
                        dataset,
                        "--seed",
                        str(seed),
                        "--model",
                        model,
                        "--mode",
                        mode,
                        "--load-memory",
                        str(memory_file),
                        "--test-output",
                        str(test_file),
                    ]
                    cmd.extend(
                        [
                            "--num-train",
                            str(n_train),
                            "--num-val",
                            str(n_val),
                            "--num-test",
                            str(n_test),
                        ]
                    )
                    if api_base:
                        cmd.extend(["--api-base", api_base])
                    if temperature is not None:
                        cmd.extend(["--temperature", str(temperature)])
                    cmd.extend(
                        [
                            "--max-workers",
                            str(INNER_MAX_WORKERS),
                            "--max-tokens",
                            str(INNER_MAX_TOKENS),
                        ]
                    )
                    if test_file.exists():
                        cmd.append("--force")
                    runs.append((desc, cmd))
    random.shuffle(runs)
    return runs, len(runs), num_done


def print_frontier(
    logs_dir: Path,
    results_dir: Path,
    model_filter: str | None = None,
    metric: str = "val",
):
    """Print frontier (best system per dataset) and write frontier JSON."""
    if metric == "test":
        base_dir = results_dir
        filename = "test.json"
    else:
        base_dir = logs_dir
        filename = "val.json"

    results = load_results(base_dir, filename)
    if not results:
        print("No results found")
        return

    if model_filter:
        results = {k: v for k, v in results.items() if k[0] == model_filter}
        if not results:
            print(f"No results for model: {model_filter}")
            return

    # Compute best system per dataset
    by_dataset = defaultdict(list)
    for (model, dataset, memory), data in results.items():
        acc = (data.get("accuracy") or 0) * 100
        ctx_len = data.get("visual_tokens", 0)
        by_dataset[dataset].append(
            {"memory": memory, "accuracy": acc, "ctx_len": ctx_len}
        )

    frontier = {}
    for dataset in DATASETS:
        if dataset in by_dataset:
            best = max(
                by_dataset[dataset], key=lambda x: (x["accuracy"], -x["ctx_len"])
            )
            frontier[dataset] = {
                "best_system": best["memory"],
                "accuracy": best["accuracy"],
                "ctx_len": best["ctx_len"],
            }

    # Print frontier
    print("\n" + "=" * 60)
    title = (
        f"FRONTIER [{metric}] (model: {model_filter})"
        if model_filter
        else f"FRONTIER [{metric}]"
    )
    print(title)
    print("=" * 60)
    for dataset in DATASETS:
        if dataset in frontier:
            info = frontier[dataset]
            acc = info["accuracy"]
            len_str = f", {info['ctx_len']:,} vis_tok" if info["ctx_len"] > 0 else ""
            print(f"  {dataset}: {info['best_system']} ({acc:.1f}%{len_str})")
        else:
            print(f"  {dataset}: (no results)")

    # Aggregate Pareto frontier
    by_memory = defaultdict(lambda: {"accs": [], "ctx_lens": []})
    for (model, dataset, memory), data in results.items():
        acc = (data.get("accuracy") or 0) * 100
        ctx_len = data.get("visual_tokens", 0)
        by_memory[memory]["accs"].append(acc)
        by_memory[memory]["ctx_lens"].append(ctx_len)

    points = []
    for mem, stats in by_memory.items():
        avg_acc = sum(stats["accs"]) / len(stats["accs"])
        non_zero = [t for t in stats["ctx_lens"] if t > 0]
        avg_len = int(sum(non_zero) / len(non_zero)) if non_zero else 0
        points.append((mem, avg_acc, avg_len))

    pareto = compute_pareto_frontier(points)
    print(f"\nPARETO FRONTIER [{metric}] (accuracy vs visual tokens):")
    print(f"  {'system':<28} {'acc':>7} {'vis_tok':>10}")
    print(f"  {'-' * 47}")
    for name, acc, length in pareto:
        len_str = f"{length:,}" if length > 0 else "0"
        print(f"  {name:<28} {acc:>7.1f} {len_str:>10}")

    # Write frontier json
    frontier["_pareto"] = [
        {
            "system": name,
            "val_accuracy" if metric == "val" else "test_accuracy": round(acc, 1),
            "ctx_len": length,
        }
        for name, acc, length in pareto
    ]
    frontier_filename = "frontier_val.json" if metric == "val" else "frontier.json"
    frontier_path = logs_dir / frontier_filename
    frontier_path.write_text(json.dumps(frontier, indent=2))
    print(f"\nWrote {frontier_path}")


def update_summary(logs_dir: Path):
    """Auto-update logs/summary.json with all val results aggregated."""
    results = load_results(logs_dir, "val.json")
    if not results:
        return

    summary = {}
    for (model, dataset, memory), data in results.items():
        if dataset not in summary:
            summary[dataset] = {}
        summary[dataset][memory] = {
            "accuracy": data.get("accuracy"),
            "memory_context_chars": data.get("memory_context_chars", 0),
            "model": model,
        }

    summary_path = logs_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


def print_summary(logs_dir: Path, results_dir: Path):
    """Print total token usage from completed runs."""
    total_tokens = 0
    for base, fn in [(logs_dir, "val.json"), (results_dir, "test.json")]:
        for data in load_results(base, fn).values():
            total_tokens += data.get("llm_input_tokens", 0) + data.get(
                "llm_output_tokens", 0
            )
    if total_tokens > 0:
        print(f"\nTotal tokens: {total_tokens:,}")


def print_missing(
    logs_dir: Path,
    memory_systems: list,
    datasets: list,
    metric: str = "val",
    results_dir: Path | None = None,
):
    """Print missing results."""
    if metric == "test":
        results = load_results(results_dir or logs_dir.parent / "results", "test.json")
    else:
        results = load_results(logs_dir, "val.json")
    all_memories = [n for n, _ in memory_systems]
    target_models = [get_model_short_name(m["model"]) for m in MODELS]

    missing = []
    for model in target_models:
        for ds in datasets:
            for mem in all_memories:
                if (model, ds, mem) not in results:
                    missing.append((model, ds, mem))

    if missing:
        print(f"\n{'=' * 60}")
        print(f"MISSING RESULTS ({len(missing)}) [{metric}]")
        print("=" * 60)
        for model, ds, mem in missing[:20]:
            print(f"  {model} / {ds} / {mem}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


async def main():
    parser = argparse.ArgumentParser(description="Sweep datasets x memory systems")
    parser.add_argument("--memory", type=str, help="Filter to one memory system")
    parser.add_argument("--dataset", type=str, help="Filter to one dataset")
    parser.add_argument("--model", type=str, help="Filter by model (for --frontier)")
    parser.add_argument(
        "--test", action="store_true", help="Run/show test mode (default: val)"
    )
    parser.add_argument(
        "--frontier", action="store_true", help="Print frontier + write analysis files"
    )
    parser.add_argument(
        "--results", action="store_true", help="Print results table only (no jobs)"
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Only show baselines + Pareto frontier systems",
    )
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default=_CONFIG["inner_loop"].get("mode", "online"),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=_CONFIG["inner_loop"].get("num_epochs", 1),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_CONFIG["inner_loop"].get("temperature"),
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Override logs directory (default: logs/)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override test results directory (default: results/)",
    )
    parser.add_argument(
        "--migrate-shared-cache",
        type=str,
        metavar="SRC_RUN_DIR",
        help="Copy usable baseline results from SRC_RUN_DIR into the shared "
        "baseline cache (run-name independent), then exit.",
    )
    args = parser.parse_args()

    if args.migrate_shared_cache:
        _migrate_shared_cache(Path(args.migrate_shared_cache))
        return

    base = Path(__file__).parent
    logs_dir = Path(args.logs_dir).resolve() if args.logs_dir else base / "logs"
    results_dir = (
        Path(args.results_dir).resolve() if args.results_dir else base / "results"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    metric = "test" if args.test else "val"

    if args.frontier:
        print_frontier(logs_dir, results_dir, model_filter=args.model, metric=metric)
        if metric == "test":
            print_results(
                load_results(results_dir, "test.json"),
                metric_label="test",
                pareto_only=args.pareto,
            )
        else:
            print_results(
                load_results(logs_dir, "val.json"),
                metric_label="val",
                pareto_only=args.pareto,
            )
        return

    if args.results or args.pareto:
        if args.test:
            print_results(
                load_results(results_dir, "test.json"),
                metric_label="test",
                pareto_only=args.pareto,
            )
        else:
            results = load_results(logs_dir, "val.json")
            print_results(results, metric_label="val", pareto_only=args.pareto)
            if not args.pareto:
                update_summary(logs_dir)
        return

    # Auto-discover all memory systems on disk
    memory_systems = discover_all_memory_systems()

    if args.memory:
        name = Path(args.memory).stem
        memory_systems = [(n, p) for n, p in memory_systems if n == name]
        if not memory_systems:
            print(f"Error: '{args.memory}' not found on disk.")
            return

    datasets = DATASETS
    if args.dataset:
        datasets = [
            d for d in DATASETS if d == args.dataset or d.endswith(f"/{args.dataset}")
        ]
        if not datasets:
            print(f"Error: '{args.dataset}' not found. Available: {DATASETS}")
            return

    # Run from project root
    os.chdir(Path(__file__).parent.parent.parent)

    if args.test:
        runs, num_pending, num_done = build_test_runs(
            logs_dir,
            results_dir,
            memory_systems,
            datasets,
            MODELS,
            args.mode,
            args.temperature,
        )
    else:
        runs, num_pending, num_done = build_val_runs(
            logs_dir,
            memory_systems,
            datasets,
            MODELS,
            args.mode,
            args.num_epochs,
            args.temperature,
        )
    n_total = num_pending + num_done

    model_names = [get_model_short_name(m["model"]) for m in MODELS]
    mode_str = f"mode={args.mode}" + (
        f" epochs={args.num_epochs}" if args.mode == "offline" else ""
    )
    print(f"Status: {num_done}/{n_total} done, {num_pending} pending [{metric}]")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Datasets: {len(datasets)}, Memory: {len(memory_systems)}, Seeds: {SEEDS}")
    print(f"  {mode_str}")

    if args.test:
        print_results(load_results(results_dir, "test.json"), metric_label="test")
    else:
        results = load_results(logs_dir, "val.json")
        print_results(results, metric_label="val")
        update_summary(logs_dir)

    if num_pending == 0:
        print("\nAll done!")
        return

    print(f"\nLaunching {num_pending} jobs (concurrency={CONCURRENCY})...")

    launcher_logs = logs_dir / ".launcher"
    job_results = await run_all_jobs(
        runs=runs,
        logs_dir=launcher_logs,
        concurrency=CONCURRENCY,
        max_retries=2,
    )

    # Persist any freshly-produced baseline results into the run-name-independent
    # shared cache so a future run under a different --run-name can reuse them.
    _sync_baselines_to_shared(
        logs_dir, results_dir, memory_systems, datasets, MODELS, args.temperature
    )

    succeeded = sum(1 for _, ok in job_results if ok)
    print(f"\nCompleted: {succeeded}/{len(job_results)}")

    print_summary(logs_dir, results_dir)
    print_missing(
        logs_dir,
        memory_systems,
        datasets,
        metric=metric,
        results_dir=results_dir,
    )

    if args.test:
        print_results(load_results(results_dir, "test.json"), metric_label="test")
    else:
        results = load_results(logs_dir, "val.json")
        print_results(results, metric_label="val")
        update_summary(logs_dir)

    return 0 if succeeded == len(job_results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
