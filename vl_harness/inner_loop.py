"""Inner Loop: Online and offline training with memory systems."""

import json
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from .harness import VideoMemoryHarness as MemorySystem
from .harness import parse_time_reference


class JSONLLogger:
    """Append-only JSONL logger. Thread-safe."""

    def __init__(
        self, path: str | None = None, checkpoint_steps: set[int] | None = None
    ):
        self.path = Path(path) if path else None
        self.start_time = time.time()
        self.checkpoint_steps = checkpoint_steps or set()
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def log(self, type: str, **data):
        """Write a log entry. All logging goes through this method."""
        if not self.path:
            return
        entry = {"type": type, "t": round(time.time() - self.start_time, 2), **data}
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def checkpoint(self, step: int, memory_state: str):
        if step in self.checkpoint_steps:
            self.log("checkpoint", step=step, memory_state=memory_state)


def _get_eval_kwargs(ex: dict[str, Any]) -> dict[str, Any]:
    """Extract evaluation kwargs from example, handling raw_input -> input_nums mapping."""
    kwargs = {k: v for k, v in ex.items() if k not in ("input", "target")}
    if "raw_input" in ex:
        kwargs["input_nums"] = ex["raw_input"]
    return kwargs


def _unpack_eval_result(raw) -> tuple[bool, dict]:
    """Normalize evaluator output to (ok, metrics).

    Evaluators return either:
    - bool: simple correct/incorrect
    - dict: {"was_correct": bool, "metrics": {...}}
    """
    if isinstance(raw, dict):
        return raw["was_correct"], raw.get("metrics", {})
    return bool(raw), {}


def compute_micro_f1(predictions: list[dict]) -> float:
    """Compute Micro-F1 from predictions with tp/fp/fn metrics.

    Sums tp/fp/fn across all predictions that have them, computes global F1.
    Returns 0.0 if no tp/fp/fn data found.
    """
    total_tp = total_fp = total_fn = 0
    has_data = False
    for p in predictions:
        m = p.get("metrics", {})
        if "tp" in m:
            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]
            has_data = True
    if not has_data:
        return 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def make_result(preds: list[dict]) -> dict:
    """Build result dict from predictions, including micro_f1/avg_f1 when available."""
    correct = sum(1 for p in preds if p["was_correct"])
    parse_fail = sum(1 for p in preds if p.get("parse_fail", p.get("prediction") == "?"))
    valid = len(preds) - parse_fail
    prediction_counts: dict[str, int] = {}
    for p in preds:
        label = str(p.get("prediction", "?")).strip().upper() or "?"
        prediction_counts[label] = prediction_counts.get(label, 0) + 1
    result = {
        "accuracy": correct / len(preds) if preds else 0.0,
        "correct": correct,
        "total": len(preds),
        "parse_fail": parse_fail,
        "valid_predictions": valid,
        "valid_accuracy": correct / valid if valid else 0.0,
        "prediction_counts": prediction_counts,
    }
    # Compute rich metrics if any prediction has them
    f1_values = [
        p["metrics"]["f1"] for p in preds if p.get("metrics", {}).get("f1") is not None
    ]
    if f1_values:
        result["avg_f1"] = sum(f1_values) / len(f1_values)
        result["micro_f1"] = compute_micro_f1(preds)
    else:
        result["avg_f1"] = None
        result["micro_f1"] = None
    return result


def _run_offline_loop(
    memory: MemorySystem,
    examples: list[dict[str, Any]],
    check_answer: Callable[..., bool],
    num_epochs: int = 1,
    batch_size: int = 1,
    max_workers: int = 32,
    logger: JSONLLogger | None = None,
    step_offset: int = 0,
    collect_trajectory: bool = True,
    val_examples: list[dict[str, Any]] | None = None,
    skip_train_eval: bool = False,
) -> dict[str, Any]:
    """Run offline training: train with ground truth visible, then evaluate.

    In offline mode:
    1. Train phase: batch examples → learn_from_batch, multiple epochs
    2. If val_examples provided: eval on val after each epoch, keep best checkpoint
    3. Eval phase: predict on all examples to measure final accuracy (no updates)
       (skipped when skip_train_eval=True, e.g. val-only evolve runs)

    Returns accuracy measured AFTER training (not during).
    """
    trajectory = [] if collect_trajectory else None
    total_steps = num_epochs * len(examples)

    # Early stopping state
    best_val_acc = -1.0
    best_state = None
    best_epoch = 0

    # Training phase: batch-based learning with ground truth visible
    step = 0
    for epoch in range(num_epochs):
        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start : batch_start + batch_size]

            # Create batch_results with ground truth as "prediction"
            batch_results = []
            for ex in batch:
                r = {
                    "input": ex["input"],
                    "prediction": ex["target"],  # Ground truth visible
                    "ground_truth": ex["target"],
                    "was_correct": True,
                }
                # Forward extra fields (e.g. raw_question) for memory systems
                for k, v in ex.items():
                    if k not in ("input", "target") and k not in r:
                        r[k] = v
                batch_results.append(r)

            t0 = time.time()
            memory.learn_from_batch(batch_results)
            train_ms = int((time.time() - t0) * 1000)

            if logger:
                global_idx = step_offset + step
                logger.log(
                    "train_batch",
                    step=global_idx,
                    epoch=epoch,
                    batch_size=len(batch),
                    train_ms=train_ms,
                )
                logger.checkpoint(global_idx, memory.get_state())

            step += len(batch)

        # Val eval after each epoch for early stopping
        if val_examples:
            val_result = evaluate_memory(
                memory, val_examples, check_answer, max_workers
            )
            val_acc = val_result["accuracy"]
            if logger:
                logger.log(
                    "val_epoch",
                    epoch=epoch,
                    val_acc=round(val_acc, 4),
                    val_correct=val_result["correct"],
                    val_total=val_result["total"],
                )
            print(
                f"  epoch {epoch}: val={val_acc:.1%} ({val_result['correct']}/{val_result['total']})",
                flush=True,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = memory.get_state()
                best_epoch = epoch

    # Restore best checkpoint if we did early stopping
    if best_state is not None and best_epoch < num_epochs - 1:
        print(
            f"  early stopping: restoring epoch {best_epoch} (val={best_val_acc:.1%})",
            flush=True,
        )
        memory.set_state(best_state)
        if logger:
            logger.log(
                "early_stop", best_epoch=best_epoch, best_val_acc=round(best_val_acc, 4)
            )

    # Evaluation phase: predict on all examples (no updates)
    # Skip when skip_train_eval=True (val-only mode — saves ~15 min per system)
    if skip_train_eval:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": len(examples),
            "trajectory": trajectory,
            "num_epochs": num_epochs,
        }

    def predict_one(idx: int, ex: dict[str, Any]) -> tuple:
        pred, meta = memory.predict(ex["input"])
        prompt_info = memory.get_last_prompt_info()
        raw = check_answer(pred, ex["target"], **_get_eval_kwargs(ex))
        ok, metrics = _unpack_eval_result(raw)
        return idx, ex, pred, meta, ok, metrics, prompt_info

    results = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(examples)))) as exe:
        futures = {exe.submit(predict_one, i, ex): i for i, ex in enumerate(examples)}
        for future in as_completed(futures):
            idx, ex, pred, meta, ok, metrics, prompt_info = future.result()
            results[idx] = (ex, pred, meta, ok, metrics, prompt_info)

    correct = 0
    for idx, (ex, pred, meta, ok, metrics, prompt_info) in enumerate(results):
        global_idx = step_offset + total_steps + idx
        correct += int(ok)

        if logger:
            logger.log(
                "eval_step",
                step=global_idx,
                input_preview=ex["input"][:200],
                pred=pred,
                tgt=ex["target"],
                ok=ok,
                prompt_len=prompt_info["prompt_len"],
                prompt_hash=prompt_info["prompt_hash"],
            )

        # Optional per-sample stderr progress. Enabled by TC_VERBOSE env var:
        #   TC_VERBOSE=1  -> compact one-liner per sample
        #   TC_VERBOSE=2  -> also print truncated full pred/target
        if os.environ.get("TC_VERBOSE"):
            _level = os.environ.get("TC_VERBOSE", "1")
            _mark = "OK " if ok else "MISS"
            _inp = (ex["input"] or "").replace("\n", " ")[:60]
            _pred = str(pred).replace("\n", " ")[:40]
            _tgt = str(ex["target"]).replace("\n", " ")[:40]
            print(
                f"    [eval {global_idx + 1}/{len(examples)}] {_mark} "
                f"tgt={_tgt!r:<42} pred={_pred!r:<42} | {_inp}",
                file=sys.stderr,
                flush=True,
            )
            if _level >= "2":
                print(f"      raw={str(meta)[:200]}", file=sys.stderr, flush=True)

        if trajectory is not None:
            trajectory.append(
                {
                    "step": global_idx,
                    "input": ex["input"],
                    "prediction": pred,
                    "target": ex["target"],
                    "was_correct": ok,
                    "metrics": metrics,
                    "metadata": meta,
                }
            )

    return {
        "accuracy": correct / len(examples) if examples else 0.0,
        "correct": correct,
        "total": len(examples),
        "trajectory": trajectory,
        "num_epochs": num_epochs,
    }


def run_inner_loop(
    memory: MemorySystem,
    examples: list[dict[str, Any]],
    check_answer: Callable[..., bool],
    batch_size: int = 1,
    max_workers: int = 32,
    logger: JSONLLogger | None = None,
    step_offset: int = 0,
    collect_trajectory: bool = True,
    mode: str = "online",
    num_epochs: int = 1,
    val_examples: list[dict[str, Any]] | None = None,
    skip_train_eval: bool = False,
) -> dict[str, Any]:
    """Run training with memory system.

    Args:
        memory: Memory system to train. Must have thread-safe predict() if batch_size > 1.
        examples: List of examples with {input, target} (and optional raw_input)
        check_answer: Function (prediction, target, **kwargs) -> bool
        batch_size: Number of examples to predict before updating (1=fully online)
        max_workers: Max parallel workers for batch predictions
        logger: JSONLLogger for structured logging
        step_offset: Starting step number (for chunked training)
        collect_trajectory: Whether to collect full trajectory (disable for memory efficiency)
        mode: "online" or "offline"
            - online: predict first, then update with feedback (single pass)
            - offline: train with ground truth visible, can run multiple epochs
        num_epochs: Number of epochs for offline mode (ignored in online mode)
        val_examples: Validation examples for early stopping in offline mode
        skip_train_eval: Skip final train eval in offline mode (val-only evolve runs)
    """
    if mode == "offline":
        return _run_offline_loop(
            memory=memory,
            examples=examples,
            check_answer=check_answer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            max_workers=max_workers,
            logger=logger,
            step_offset=step_offset,
            collect_trajectory=collect_trajectory,
            val_examples=val_examples,
            skip_train_eval=skip_train_eval,
        )
    # Online mode (default): predict batch → learn from batch
    correct = 0
    trajectory = [] if collect_trajectory else None

    def predict_one(idx: int, ex: dict[str, Any]) -> tuple:
        t0 = time.time()
        pred, meta = memory.predict(ex["input"])
        prompt_info = memory.get_last_prompt_info()
        return idx, ex, pred, meta, prompt_info, time.time() - t0

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]

        # PHASE 1: PREDICT (parallel within batch)
        if batch_size == 1:
            pred_results = [predict_one(0, batch[0])]
        else:
            pred_results = []
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as exe:
                futures = {
                    exe.submit(predict_one, i, ex): i for i, ex in enumerate(batch)
                }
                for future in as_completed(futures):
                    pred_results.append(future.result())
            pred_results.sort(key=lambda x: x[0])

        # Build batch_results for learn_from_batch
        batch_results = []
        for idx, ex, pred, meta, prompt_info, _predict_s in pred_results:
            global_idx = step_offset + batch_start + idx
            inp, tgt = ex["input"], ex["target"]
            raw = check_answer(pred, tgt, **_get_eval_kwargs(ex))
            ok, metrics = _unpack_eval_result(raw)
            correct += int(ok)

            result = {
                "input": inp,
                "prediction": pred,
                "ground_truth": tgt,
                "was_correct": ok,
                "metadata": meta,
            }
            if metrics:
                result["metrics"] = metrics
            # Forward extra fields (e.g. raw_question) for memory systems
            for k, v in ex.items():
                if k not in ("input", "target") and k not in result:
                    result[k] = v
            batch_results.append(result)

            # Log individual step
            if logger:
                logger.log(
                    "step",
                    step=global_idx,
                    input_preview=inp[:200],
                    pred=pred,
                    tgt=tgt,
                    ok=ok,
                    prompt_len=prompt_info["prompt_len"],
                    prompt_hash=prompt_info["prompt_hash"],
                )

            if trajectory is not None:
                trajectory.append(
                    {
                        "step": global_idx,
                        "input": inp,
                        "prediction": pred,
                        "target": tgt,
                        "was_correct": ok,
                        "metrics": metrics,
                        "metadata": meta,
                    }
                )

        # PHASE 2: LEARN FROM BATCH
        t0 = time.time()
        memory.learn_from_batch(batch_results)
        learn_ms = int((time.time() - t0) * 1000)

        if logger:
            batch_idx = batch_start // batch_size
            logger.log(
                "learn_batch",
                batch_idx=batch_idx,
                batch_size=len(batch_results),
                learn_ms=learn_ms,
            )
            logger.checkpoint(
                step_offset + batch_start + len(batch) - 1, memory.get_state()
            )

    return {
        "accuracy": correct / len(examples) if examples else 0.0,
        "correct": correct,
        "total": len(examples),
        "trajectory": trajectory,
    }


def evaluate_memory(
    memory: MemorySystem,
    examples: list[dict[str, Any]],
    check_answer: Callable[..., bool],
    max_workers: int = 32,
) -> dict[str, Any]:
    """Evaluate without updating (parallel)."""
    if not examples:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "predictions": [],
            "avg_prompt_len": 0,
        }

    def predict_one(idx: int, ex: dict[str, Any]) -> tuple:
        pred, meta = memory.predict(ex["input"])
        meta = dict(meta or {})
        prompt_info = memory.get_last_prompt_info()
        prompt_len = prompt_info.get("prompt_len") or 0
        prompt_text = prompt_info.get("prompt_text") or ""
        # Injected context = full prompt - test input (remainder is template + memory context)
        context_len = max(0, prompt_len - len(ex["input"])) if prompt_len else 0
        # VL-Harness Pareto currency: visual tokens shown to answer this question.
        visual_tokens = prompt_info.get("visual_tokens") or 0
        num_frames = prompt_info.get("num_frames") or 0
        raw = check_answer(pred, ex["target"], **_get_eval_kwargs(ex))
        ok, metrics = _unpack_eval_result(raw)
        result = {
            "prediction": pred,
            "target": ex["target"],
            "was_correct": ok,
            "prompt_len": prompt_len,
            "context_len": context_len,
            "prompt_text": prompt_text,
            "visual_tokens": visual_tokens,
            "num_frames": num_frames,
            "parse_fail": bool(meta.get("parse_fail", pred == "?")),
            "raw_response": meta.get("raw"),
        }
        # Did the model ever see the moment the answer is annotated to live in?
        # Recorded per question so any run can be stratified by evidence
        # visibility and by how wide the annotation is, without re-running.
        window = parse_time_reference((ex.get("meta") or {}).get("time_reference"))
        if window:
            t0, t1 = window
            shown = prompt_info.get("frame_times") or []
            # 17% of LVBench windows are zero-width ("at this instant"), which no
            # sampled frame can ever land inside. Recording the DISTANCE from the
            # nearest shown frame to the window, rather than a boolean, keeps the
            # tolerance a post-hoc analysis choice instead of silently forcing
            # those questions into the "evidence missed" group forever.
            gap = min((max(0.0, t0 - t, t - t1) for t in shown), default=None)
            result["evidence_window_s"] = round(t1 - t0, 2)
            result["evidence_gap_s"] = None if gap is None else round(gap, 2)
            result["evidence_visible_strict"] = any(t0 <= t <= t1 for t in shown)
            result["evidence_frames_shown"] = sum(1 for t in shown if t0 <= t <= t1)
        if metrics:
            result["metrics"] = metrics
        return idx, result

    results = [None] * len(examples)
    total = len(examples)
    t_start = time.time()
    done = 0
    # Progress is printed as it happens because this function otherwise writes
    # nothing until all examples finish. A long run then looks identical to a
    # hung one, and every wrong call made about the 1009-frame line today came
    # from having to guess which it was.
    every = 1 if total <= 20 else max(1, total // 50)
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(examples)))) as exe:
        futures = {exe.submit(predict_one, i, ex): i for i, ex in enumerate(examples)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            done += 1
            if done % every == 0 or done == total:
                el = time.time() - t_start
                rate = el / done
                print(
                    f"  [eval] {done}/{total}  {el / 60:.1f} min elapsed"
                    f"  {rate:.1f} s/q  eta {(total - done) * rate / 60:.0f} min",
                    flush=True,
                )

    correct = sum(1 for r in results if r["was_correct"])
    context_lens = [r["context_len"] for r in results]
    avg_context_len = int(sum(context_lens) / len(context_lens)) if context_lens else 0
    vtoks = [r.get("visual_tokens", 0) for r in results]
    avg_visual_tokens = int(sum(vtoks) / len(vtoks)) if vtoks else 0
    nframes = [r.get("num_frames", 0) for r in results]
    avg_num_frames = round(sum(nframes) / len(nframes), 2) if nframes else 0
    return {
        "accuracy": correct / len(examples),
        "correct": correct,
        "total": len(examples),
        "predictions": results,
        "avg_context_len": avg_context_len,
        "avg_visual_tokens": avg_visual_tokens,
        "avg_num_frames": avg_num_frames,
    }


def load_memory_system(
    path: str,
    vlm,
    embedder=None,
    target_side: int | None = None,
    frame_budget: int | None = None,
) -> MemorySystem:
    """Load a video-memory harness from a file path.

    Accepts paths like:
    - 'agents/uniform_frames_no_memory.py'
    - 'agents/my_candidate.py'
    - 'uniform_frames_no_memory' (searches agents/)
    """
    import importlib
    import inspect

    # Handle short names (without directory)
    if "/" not in path and not path.endswith(".py"):
        try:
            return load_memory_system(
                f"agents/{path}.py", vlm, embedder, target_side, frame_budget
            )
        except (ModuleNotFoundError, ValueError):
            raise ValueError(f"Harness '{path}' not found in agents") from None

    module_path = path.replace("/", ".").replace(".py", "")
    module = importlib.import_module(f".{module_path}", package="vl_harness")

    # getmembers sorts by name, so a harness that subclasses another one silently
    # ran its imported parent whenever the parent's name sorted first -- the two
    # modality ablations and both _maxbudget arms produced byte-identical requests
    # to their parents and were scored as if they were real variants. Classes
    # defined in this file win; an imported base is only a fallback.
    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, MemorySystem) and obj is not MemorySystem
    ]
    own = [obj for obj in candidates if obj.__module__ == module.__name__]
    if len(own) > 1:
        raise ValueError(
            f"{path} defines {len(own)} harness classes "
            f"({', '.join(o.__name__ for o in own)}); it must define exactly one"
        )

    for obj in own or candidates:
        # VideoMemoryHarness subclasses take (vlm, embedder).
        ms = obj(vlm=vlm, embedder=embedder)
        if target_side is not None:
            ms.target_side = int(target_side)
        if frame_budget:
            ms.FRAME_BUDGET = int(frame_budget)
        return ms

    raise ValueError(f"No VideoMemoryHarness subclass found in {path}")


def load_config() -> dict:
    """Load the config selected by VL_HARNESS_CONFIG, or config_k40.yaml."""
    import os
    import yaml

    config_path = Path(os.environ.get("VL_HARNESS_CONFIG", "config_k40.yaml"))
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import argparse

    from .data import ALL_TASKS, load_dataset_splits, load_dataset_splits_3way

    # Load config from YAML
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Run inner loop with memory system")
    parser.add_argument("--memory", default="agents/no_memory.py")
    parser.add_argument("--dataset", required=True, help=f"Options: {ALL_TASKS}")
    parser.add_argument("--seed", type=int, default=cfg["inner_loop"]["seed"])
    parser.add_argument("--model", default=None, help="Model to use (overrides config)")
    parser.add_argument(
        "--api-base", default=None, help="API base URL (overrides config)"
    )
    parser.add_argument(
        "--mode",
        default=cfg["inner_loop"].get("mode", "online"),
        choices=["online", "offline"],
        help="Training mode: online (predict->feedback->update) or offline (train with labels->eval)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=cfg["inner_loop"].get("num_epochs", 1),
        help="Number of epochs for offline mode (ignored in online mode)",
    )
    parser.add_argument(
        "--num-train", type=int, default=None, help="Override num_train from config"
    )
    parser.add_argument(
        "--num-val", type=int, default=None, help="Override num_val from config"
    )
    parser.add_argument(
        "--num-test", type=int, default=None, help="Override num_test from config"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature (overrides model default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=cfg["inner_loop"].get("max_tokens", 64),
        help="Max completion tokens per VLM call.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并发请求数（本地单服务建议1，避免超时）",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=cfg["inner_loop"].get("enable_thinking", False),
        help="Enable Qwen3-VL thinking mode (default from inner_loop.enable_thinking in config).",
    )
    # New output args: split val/test into separate files
    parser.add_argument(
        "--save-memory",
        default=None,
        help="Write memory state to this path after training",
    )
    parser.add_argument(
        "--load-memory",
        default=None,
        help="Load memory state from this path (skip training)",
    )
    parser.add_argument(
        "--val-output", default=None, help="Write val results JSON here"
    )
    parser.add_argument(
        "--test-output", default=None, help="Write test results JSON here"
    )
    parser.add_argument(
        "--frame-budget",
        type=int,
        default=None,
        help=(
            "Frames per VLM request for this sweep arm (overrides "
            "video.frame_budget). Clamps ingest sampling too, so uniform "
            "baselines become uniform-K automatically."
        ),
    )
    parser.add_argument("--log", default=None, help="Path for JSONL training log")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Skip if all requested outputs already exist
    outputs_to_check = [p for p in [args.val_output, args.test_output] if p]
    if (
        outputs_to_check
        and all(Path(p).exists() for p in outputs_to_check)
        and not args.force
    ):
        print(f"Already complete, skipping: {outputs_to_check}")
        exit(0)

    # Resolve dataset sizes: CLI args > per-dataset overrides > defaults
    ds = cfg["dataset"]
    ds_overrides = ds.get("overrides", {}).get(args.dataset, {})
    num_train = (
        args.num_train
        if args.num_train is not None
        else ds_overrides.get("num_train", ds["num_train"])
    )
    num_val = (
        args.num_val
        if args.num_val is not None
        else ds_overrides.get("num_val", ds["num_val"])
    )
    num_test = (
        args.num_test
        if args.num_test is not None
        else ds_overrides.get("num_test", ds["num_test"])
    )

    eval_val = args.val_output is not None
    eval_test = args.test_output is not None

    print(f"Loading dataset: {args.dataset}", flush=True)
    if num_val > 0:
        train_examples, val_examples, test_examples, evaluator = (
            load_dataset_splits_3way(
                args.dataset,
                num_train=num_train,
                num_val=num_val,
                num_test=num_test,
                shuffle_seed=args.seed,
            )
        )
        print(
            f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}",
            flush=True,
        )
    else:
        train_examples, test_examples, evaluator = load_dataset_splits(
            args.dataset,
            num_train=num_train,
            num_test=num_test,
            shuffle_seed=args.seed,
        )
        val_examples = []
        print(f"Train: {len(train_examples)}, Test: {len(test_examples)}", flush=True)

    from .vlm import VLM, MultimodalEmbedder, make_stub_vlm

    # Resolve model/api_base: CLI args > first entry in config models list
    if args.model:
        model = args.model
    elif cfg.get("models"):
        model = cfg["models"][0]["model"]
    else:
        raise ValueError(
            "No model specified. Use --model or set 'models' in config.yaml"
        )
    if args.api_base:
        api_base = args.api_base
    elif cfg.get("models"):
        api_base = cfg["models"][0].get("api_base")
    else:
        api_base = None

    # "stub" model => dependency-free plumbing VLM (offline smoke tests).
    if model == "stub":
        vlm = make_stub_vlm()
    else:
        vlm = VLM(
            model=model,
            api_base=api_base,
            temperature=args.temperature if args.temperature is not None else 0.0,
            max_tokens=max(8, int(args.max_tokens)),
            enable_thinking=args.enable_thinking,
            timeout=float(cfg.get("inner_loop", {}).get("vlm_timeout", 1800.0)),
        )

    # Embedder: stub unless config asks for a real backend.
    emb_cfg = cfg.get("embedder", {}) or {}
    embedder = MultimodalEmbedder(
        backend=emb_cfg.get("backend", "stub"),
        dim=emb_cfg.get("dim", 256),
        model=emb_cfg.get("model"),
        api_base=emb_cfg.get("api_base"),
        timeout=emb_cfg.get("timeout", 60.0),
    )

    video_cfg = cfg.get("video", {}) or {}
    target_side = video_cfg.get("target_side")
    frame_budget = (
        args.frame_budget
        if args.frame_budget is not None
        else video_cfg.get("frame_budget")
    )
    memory = load_memory_system(
        path=args.memory,
        vlm=vlm,
        embedder=embedder,
        target_side=target_side,
        frame_budget=frame_budget,
    )
    if frame_budget:
        print(f"[frame-budget] {frame_budget} frames/request (sweep arm)", flush=True)
    memory_name = Path(args.memory).stem

    il = cfg["inner_loop"]
    eval_interval = (
        il["eval_interval"] if il["eval_interval"] > 0 else len(train_examples) + 1
    )
    checkpoint_steps = set(range(0, len(train_examples) + 1, eval_interval))
    checkpoint_steps.add(len(train_examples) - 1)
    logger = JSONLLogger(args.log, checkpoint_steps=checkpoint_steps)
    logger.log(
        "meta",
        dataset=args.dataset,
        memory=memory_name,
        model=model,
        seed=args.seed,
        mode=args.mode,
        num_epochs=args.num_epochs if args.mode == "offline" else None,
        start_time=datetime.now().isoformat(),
    )

    train_correct = 0
    run_start = time.time()

    if args.load_memory:
        # Skip training — load saved memory state
        state = Path(args.load_memory).read_text()
        memory.set_state(state)
        print(f"Loaded memory state from {args.load_memory}", flush=True)
        train_acc = 0.0
    else:
        # Training loop
        mode_str = f"mode={args.mode}" + (
            f" epochs={args.num_epochs}" if args.mode == "offline" else ""
        )
        print(
            f"[0/{len(train_examples)}] {mode_str} {time.time() - run_start:.1f}s",
            flush=True,
        )

        if args.mode == "offline":
            chunk_results = run_inner_loop(
                memory,
                train_examples,
                evaluator,
                batch_size=il["batch_size"],
                logger=logger,
                step_offset=0,
                mode="offline",
                num_epochs=args.num_epochs,
                # A single epoch has no earlier checkpoint to select, and the
                # requested val/test evaluation below is authoritative. Avoid
                # decoding and querying the entire validation set twice.
                val_examples=val_examples if args.num_epochs > 1 else None,
                skip_train_eval=not eval_test,
                max_workers=max(1, int(args.max_workers)),
            )
            train_correct = chunk_results["correct"]
            print(
                f"[{len(train_examples)}/{len(train_examples)}] {time.time() - run_start:.1f}s",
                flush=True,
            )
        else:
            for chunk_start in range(0, len(train_examples), eval_interval):
                chunk_end = min(chunk_start + eval_interval, len(train_examples))
                chunk_results = run_inner_loop(
                    memory,
                    train_examples[chunk_start:chunk_end],
                    evaluator,
                    batch_size=il["batch_size"],
                    logger=logger,
                    step_offset=chunk_start,
                    mode="online",
                    max_workers=max(1, int(args.max_workers)),
                )
                train_correct += chunk_results["correct"]
                print(
                    f"[{chunk_end}/{len(train_examples)}] {time.time() - run_start:.1f}s",
                    flush=True,
                )

        train_acc = train_correct / len(train_examples) if train_examples else 0.0

        # Save memory state after training
        if args.save_memory:
            Path(args.save_memory).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_memory).write_text(memory.get_state())
            print(f"Saved memory state to {args.save_memory}", flush=True)

    # Eval: only run what's requested
    val_preds = []
    test_preds = []
    avg_context_len = 0
    avg_visual_tokens = 0
    avg_num_frames = 0

    if eval_val and eval_test:
        combined = evaluate_memory(
            memory,
            val_examples + test_examples,
            evaluator,
            max_workers=max(1, int(args.max_workers)),
        )
        avg_context_len = combined["avg_context_len"]
        avg_visual_tokens = combined["avg_visual_tokens"]
        avg_num_frames = combined["avg_num_frames"]
        val_preds = combined["predictions"][: len(val_examples)]
        test_preds = combined["predictions"][len(val_examples) :]
    elif eval_val:
        result = evaluate_memory(
            memory,
            val_examples,
            evaluator,
            max_workers=max(1, int(args.max_workers)),
        )
        avg_context_len = result["avg_context_len"]
        avg_visual_tokens = result["avg_visual_tokens"]
        avg_num_frames = result["avg_num_frames"]
        val_preds = result["predictions"]
    elif eval_test:
        result = evaluate_memory(
            memory,
            test_examples,
            evaluator,
            max_workers=max(1, int(args.max_workers)),
        )
        avg_context_len = result["avg_context_len"]
        avg_visual_tokens = result["avg_visual_tokens"]
        avg_num_frames = result["avg_num_frames"]
        test_preds = result["predictions"]

    val_result = make_result(val_preds) if val_preds else None
    test_result = make_result(test_preds) if test_preds else None

    val_acc = val_result["accuracy"] if val_result else None
    test_acc = test_result["accuracy"] if test_result else None

    runtime = time.time() - run_start
    llm_usage = vlm.get_usage()

    logger.log(
        "done",
        train_acc=round(train_acc, 4),
        train_correct=train_correct,
        train_total=len(train_examples),
        val_acc=round(val_acc, 4) if val_acc is not None else None,
        test_acc=round(test_acc, 4) if test_acc is not None else None,
        runtime_seconds=round(runtime, 2),
        memory_context_chars=avg_context_len,
        visual_tokens=avg_visual_tokens,
        num_frames=avg_num_frames,
        llm_calls=llm_usage.get("calls", 0),
        llm_input_tokens=llm_usage.get("input_tokens", 0),
        llm_output_tokens=llm_usage.get("output_tokens", 0),
    )

    # Print summary
    summary = f"Done: train={train_acc:.0%}"
    if val_acc is not None:
        summary += f" val={val_acc:.0%}"
    if test_acc is not None:
        summary += f" test={test_acc:.0%}"
    summary += f" time={runtime:.1f}s"
    print(summary, flush=True)

    # Build common metadata for output JSON
    def _build_output(result_dict, preds=None):
        out = {
            "accuracy": result_dict["accuracy"],
            "correct": result_dict["correct"],
            "total": result_dict["total"],
            "parse_fail": result_dict.get("parse_fail", 0),
            "valid_predictions": result_dict.get("valid_predictions", result_dict["total"]),
            "valid_accuracy": result_dict.get("valid_accuracy", result_dict["accuracy"]),
            "prediction_counts": result_dict.get("prediction_counts", {}),
            "dataset": args.dataset,
            "memory": args.memory,
            "model": model,
            "seed": args.seed,
            "mode": args.mode,
            "enable_thinking": args.enable_thinking,
            "num_epochs": args.num_epochs if args.mode == "offline" else None,
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(runtime, 2),
            "memory_context_chars": avg_context_len,
            "visual_tokens": avg_visual_tokens,
            "num_frames": avg_num_frames,
            "llm_calls": llm_usage.get("calls", 0),
            "llm_input_tokens": llm_usage.get("input_tokens", 0),
            "llm_output_tokens": llm_usage.get("output_tokens", 0),
            "llm_total_tokens": llm_usage.get("input_tokens", 0) + llm_usage.get("output_tokens", 0),
        }
        # Write-time cost. The harness has always tracked this via
        # account_ingest, but nothing ever read it back, so "matched budget"
        # only held per REQUEST: a harness that captions 320 frames to choose 40
        # looked as cheap as one that just shows 40. Reporting ingest alongside
        # answer-time frames is what makes the budget claim total rather than
        # per-request.
        try:
            vc = memory.get_visual_cost() or {}
        except Exception:
            vc = {}
        n_q = max(1, result_dict.get("total") or 1)
        n_vid = vc.get("ingest_videos") or 0
        out["ingest_videos"] = n_vid
        out["ingest_frames_per_video"] = round(vc.get("ingest_frames_avg", 0), 1)
        out["ingest_frames_per_question"] = round(
            vc.get("ingest_frames_avg", 0) * n_vid / n_q, 2
        )
        out["total_frames_per_question"] = round(
            (avg_num_frames or 0) + vc.get("ingest_frames_avg", 0) * n_vid / n_q, 2
        )
        if preds is not None:
            out["results"] = [
                {
                    "prediction": p.get("prediction"),
                    "target": p.get("target"),
                    "was_correct": p.get("was_correct"),
                    "visual_tokens": p.get("visual_tokens"),
                    "num_frames": p.get("num_frames"),
                    "parse_fail": p.get("parse_fail", str(p.get("prediction")).strip() == "?"),
                    "raw_response": p.get("raw_response"),
                    # Evidence-visibility fields are whitelisted here too, or the
                    # per-question instrumentation is silently dropped on the way
                    # to disk and every run has to be reconstructed offline.
                    "evidence_window_s": p.get("evidence_window_s"),
                    "evidence_gap_s": p.get("evidence_gap_s"),
                    "evidence_visible_strict": p.get("evidence_visible_strict"),
                    "evidence_frames_shown": p.get("evidence_frames_shown"),
                }
                for p in preds
            ]
        return out

    if args.val_output and val_result:
        Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.val_output, "w") as f:
            json.dump(_build_output(val_result, val_preds), f, indent=2)
        print(f"Saved val results to {args.val_output}", flush=True)

    if args.test_output and test_result:
        Path(args.test_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.test_output, "w") as f:
            json.dump(_build_output(test_result, test_preds), f, indent=2)
        print(f"Saved test results to {args.test_output}", flush=True)

