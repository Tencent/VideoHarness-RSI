---
name: vl-harness
description: Run one iteration of VL-Harness evolution — evolve video-memory-harness code for long-video MCQ. Called by meta_harness.py or interactively via /vl-harness.
---

# VL-Harness (Video-Memory Harness Evolution)

Run ONE iteration of video-memory-harness evolution. Do all work in the main
session — do NOT delegate to subagents. The base VLM is FROZEN; you only evolve
the code around it (what to store, how to retrieve, what to show).

**You do NOT run benchmarks.** You analyze results + prediction traces, prototype
changes, and implement new harnesses. The outer loop (`meta_harness.py`) runs
benchmarks separately.

## CRITICAL CONSTRAINTS

- You MUST implement exactly 1 new harness every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly 1 candidate per iteration. Parent is **always the current
  frontier F_t** (the task prompt names it). Copy `agents/<F_t>.py` and splice;
  do not start from a different file.
- Write the hypothesis from traces **before** any source edit.
- The outer loop **injects an archive** into the task prompt: historical
  harness source, scores, and traces. Use it. If a section is truncated,
  `Read` the on-disk `val.json` / `val_contexts.jsonl` / `agents/*.py`.
  The file you write must still copy F_t.
- **Held-out is off-limits.** Do NOT read `manifests/lvbench_split_seed42.json`
  `test` entries, any `target` / answer letter on held-out items, or any
  `test.json` / official-test dump. Evolution is scored on val only.

## THE OBJECTIVE: maximize ACCURACY (cost is reporting-only)

**Your objective is to MAXIMIZE MCQ accuracy.** Do NOT optimize for visual-token
cost — cost is measured only to *report* an accuracy-vs-cost Pareto frontier
afterwards, it is NOT part of your objective and there is NO equal-budget
success criterion. Pick whatever representation / retrieval / packing gives the
best accuracy.

Soft guidance only: avoid *gratuitous* frame dumping (e.g. showing dozens of
near-duplicate frames when a few well-chosen ones do as well) — not to save
budget per se, but because redundant frames rarely help accuracy and often hurt
it. When accuracy is equal, the cheaper harness is preferred for reporting, but
never trade real accuracy for a lower token count.

Visual tokens are measured at the serving model (the REAL tokens the vision
tower consumes), so the reported frontier is faithful; you do not need to reason
about token counts yourself.

## Anti-parameter-tuning rules

The most common failure mode is harnesses that are parameter variants (frame
count, top-k, resolution). Check `evolution_summary.jsonl` — sweeps almost always
regress or tie. **Good candidates change a fundamental mechanism.**

## INGEST POLICY (per-request window)

The framework hard cap is a **per-request** window: one VLM call may contain at
most `frame_budget()` frames (K=40 on the paper protocol; otherwise
`MAX_REQUEST_FRAMES`). `render_frames` raises if you exceed it. That is **not**
an ingest-pool size. How many frames you ingest, at what fps, and how many
passes you use, is part of the hypothesis. Do not treat a 320-frame / 2 fps
pool as mandatory. Small top-k values remain answer-time packing choices.

## SEARCH-SPACE MODE (read this first — it constrains what you may propose)

The task prompt for this iteration contains a **SEARCH SPACE** section. Obey it
strictly — it defines a controlled experiment:

- **FULL**: you may use any modality (caption text, keyframe images, visual
  embeddings, structured memory) and any cross-modal retrieval routing
  (text->text / text->image / image->image / fusion). All 7 axes below are open.
- **TEXT-ONLY** (control condition): you MUST restrict every candidate to
  **textual memory + text->text retrieval only**. HARD CONSTRAINTS: at ANSWER
  time show NO images/frames to the VLM and use NO image / visual-embedding
  retrieval to decide what the VLM sees. You MAY caption frames at INGEST time,
  but only captions may enter the answer-time prompt. This emulates a text-domain
  harness (the "Meta-Harness in the video domain" baseline). Axes B/D/F below
  are effectively closed; move A/C/E/G within the text modality.

The accuracy difference between FULL and TEXT-ONLY runs quantifies the value of
making memory/retrieval **modality** a searchable object — do not break the
TEXT-ONLY constraint, or the ablation is invalidated.

## The 7 video-specific search axes

Pick a candidate that moves a DIFFERENT axis than the last 3 iterations when
that still improves the frontier; otherwise exploitation of F_t is fine.

- **A. Ingestion** — uniform vs shot-boundary vs motion-adaptive sampling;
  eager vs lazy captioning; ingest density.
- **B. Representation** — captions / keyframe images / visual embeddings /
  structured (entity tracks, scene graph) / mixtures.
- **C. Granularity & hierarchy** — frame / shot / scene levels; frame->shot->scene
  summaries; near-duplicate dedup.
- **D. Retrieval modality router** — text->text / text->image / image->image /
  fusion; route by question type (appearance vs reasoning vs localization).
- **E. Retrieval algorithm** — dense kNN / temporal windows / MMR diversity /
  coarse-to-fine (locate scene then pull frames) / temporal grounding ("when did X").
- **F. Budget packing** — under a fixed visual-token budget, N images vs M captions;
  resolution tiers; ordering; which frames actually earn their tokens.
- **G. Answering** — single pass / re-watch verification (retrieve, then look again
  to confirm) / multi-retrieval self-consistency.

## HARD EXPLORATION CONSTRAINT (optional at N=1)

Read `evolution_summary.jsonl` and take **the last 3 iterations' candidates**
(if fewer, use all available). Collect the union of every candidate's
`components` axis tags (e.g. `axisA-ingest`, `axisD-router`, ...).

**Constraint**: N=1, so the forced-exploration slot is optional. Prefer an
unused axis when it still improves the frontier; otherwise exploitation is
fine. If you do take a new axis, set `axis` to `"exploration"`.

If every axis has already appeared in the recent-3 union (rare), you MAY
instead switch to a fundamentally different **combination**.

If an exploration candidate loses on val, that is FINE and EXPECTED — the
value of exploration is priced into the multi-iteration search, not into the
single-iteration accuracy.

Prior hand-designed systems (WorldMM: episodic/semantic/visual memory + adaptive
routing; Homer: hierarchical perceptual/entity/event memory + verify-and-correct +
runtime skill accumulation) are FAIR GAME to reimplement and combine as candidates
— the point is for search to rediscover/surpass them automatically.

## Anti-overfitting rules (STRICT — we report on held-out sets)

- **No dataset-specific hints.** Never hard-code knowledge about specific videos,
  questions, or answer distributions.
- **Never mention dataset names** in code, prompts, or comments.
- **No answer leakage.** Do not bias toward any option letter or exploit MCQ format
  regularities. General strategies ("prioritize high-motion frames", "route
  appearance questions to images") are fine — they apply broadly.

## WORKFLOW (do ALL steps yourself)

### Step 1: Analyze traces, then write H
From the run directory in the task prompt, read F_t's `val.json` (which
questions are already correct) and `val_contexts.jsonl` (what F_t actually
showed the VLM: frames, timestamps, prompt text). Also read
`evolution_summary.jsonl` and `frontier_val.json`.

Formulate **one** falsifiable hypothesis from those traces, then implement.

- Rows with `outcome: "execution_error"` are INFRASTRUCTURE CRASHES, NOT rejected
  hypotheses. Do not update your prior based on them; do not avoid the mechanism
  they used. If a promising axis has only execution_error rows, it is fair game
  (and often high-value) to retry with a simpler / more robust implementation.

### Step 2: Prototype — MANDATORY
Write a throwaway script in `/tmp/` exercising the core retrieval/packing logic
on a few real episodes pulled from logs. Try 2-3 variants; keep the best.
Delete scripts when done.

### Step 3: Implement
1. Pick a globally unique snake_case `name` (check `evolution_summary.jsonl`;
   append `_iter{N}` on collision).
2. Copy `agents/<F_t>.py` to `agents/<name>.py`, then splice the new mechanism.
3. **Self-critique:** if `build_memory`/`answer_question` differ from F_t only
   in constants, REWRITE with a genuinely new mechanism.
4. Validate: `python -c "from vl_harness.agents.<name> import *; print('OK')"`

Do not edit `config.yaml` to register candidates — `agents/` is auto-discovered.

### Step 4: Write pending_eval.json
Write to the path given in the task prompt:

```json
{
  "iteration": <N>,
  "candidates": [
    {"name": "<snake>", "file": "agents/<name>.py", "hypothesis": "<claim>",
     "axis": "exploitation|exploration", "base_system": "<F_t>",
     "components": ["axisD-router", "axisF-packing"]}
  ]
}
```

Output: `CANDIDATES: <name>`

## VideoMemoryHarness interface

```python
class VideoMemoryHarness(ABC):
    def __init__(self, vlm, embedder=None, target_side=224): ...
    def build_memory(self, video: VideoStream) -> Any: ...        # ingest once per video
    def answer_question(self, memory, question, options) -> tuple[str, dict]: ...

# helpers you SHOULD use:
#   self.caption_frame(frame)            -> caption (mock: preset; real: VLM, write-time)
#   self.embed_texts([...]) / self.embed_images([...])
#   self.topk_indices(query_vec, matrix, k)
#   self.render_frames(frames)           -> content parts + charges visual tokens
#   self.ask_vlm(parts)                  -> VLM response text
#   self.account_ingest(video_id, n_frames, visual_tokens)   # write-time cost
#   normalize_choice(pred, options), format_options(options), extract_json_field(resp, "final_answer")
```

- `build_memory` runs ONCE per unique video (cached, thread-safe).
- `answer_question` must work for MCQ; return the option letter + metadata.
- Visual tokens shown in `answer_question` are the Pareto currency — spend them wisely.

## Directory structure
- Val results: `<run>/ <dataset>/<harness>/<model>/val.json`
- Traces: `<run>/<dataset>/<harness>/<model>/log.jsonl`
- Context traces (what the VLM saw): `<run>/<dataset>/<harness>/<model>/val_contexts.jsonl`
- Test results: `results/<dataset>/<harness>/<model>/test.json` (never seen during evolution)
