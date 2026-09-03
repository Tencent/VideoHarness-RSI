"""Cardinality ledger: enumerate the occurrences, then read the total off the list.

Parent: ``agents/pointer_ladder.py`` (frontier F_t). Its ingest, its coarse->fine
pointer ladder, its stated-clock bypass, its burst decoder, its AKS anchor top-up
and its verbatim answer prompt are all inherited. Nothing about WHERE to look
changes. What changes is what the harness is allowed to do with what it sees --
and only for questions that ask for an exact number.

Six iterations of this line have all bought localisation, and the frontier's own
val traces say that on one large slice of its wrong-set, buying more of it is
actively counter-productive:

    shown frames inside the answer window      non-count acc      count acc
        0                                          54.2%            57.1%
        1-10                                       54.0%            11.1%
        11+                                        76.0%            12.5%

Non-count questions ride the parent's asset -- 54% -> 76% as the evidence
arrives. Counting runs the other way. It is the one place where the harness sees
more and does worse.

Isolating the slice: a question is in it when it asks for a cardinality AND its
options are bare numbers. Both conjuncts matter, and each was checked alone:

    cardinality phrase AND >=3 numeric options    3/26 = 11.5%
    cardinality phrase, non-numeric options       6/9  = 66.7%
    numeric options, no cardinality phrase        5/6  = 83.3%
    everything else                             198/324 = 61.1%   (Fisher p=9.3e-7)

11.5% is below the 25% a coin scores -- P(<=3 of 26 | p=0.25) = 0.080 -- and the
errors are unsigned (15 under-counts vs 9 over-counts, binomial p=0.31). Neither
conjunct is the problem on its own; only together do they collapse. That is the
signature of a guess whose magnitude is plausible and whose value is arbitrary,
not of a model that looked in the wrong place.

Mechanism (axis B representation + axis G answering): an exact cardinality is
CONSTRUCTED, not recognised. A count of N things that are never co-visible cannot
be read off a frame set the way an appearance or an identity can -- it needs each
occurrence enumerated, each one judged new or a re-sighting of one already
counted, and only then a total. F_t gives that process nowhere to happen: all 350
of its answer calls reply with exactly one character, so the tally must complete
implicitly inside a single forward pass over 40 frames, and what emerges is a
magnitude-shaped guess.

Three traces say the deficit is not evidence and not aiming:

* 'How many Asian speakers appear in the video?' -- 85 frames inside the answer
  window, 0% of adjacent shown frames within 3s of each other, so coverage was
  genuinely spread. Answered 3; truth 5.
* 'How many hosts have changed?' -- 53 in-window frames over 3660s. Answered 1;
  truth 6.
* 'How many steak on the left side of the table at 21:00?' -- takes the
  stated-clock bypass, the route that scores 66.7% overall, with 12 frames inside
  a 22s window. Answered 1; truth 3. The proven localiser fires correctly and the
  count is still wrong.

So the repair is an accumulator, not a better aim. On a gated question the same
selected frames are shown to the frozen VLM with a different job: list the
distinct occurrences, one per line, each with the timestamp where it was seen.
That list is the ledger F_t lacks. The option is then chosen from the ledger's
own contents, in a text-only second call -- the arithmetic leaves the forward
pass and becomes text the model can inspect and re-read.

Why the frame selection is deliberately NOT touched: the rival reading is that
counting fails because the burst packs near-duplicates (median gated question has
45% of adjacent shown frames under 3s apart, with a median largest unwatched gap
of 708s). That rival is already weakened by the table above -- the 0-frames
bucket outscores both dense buckets -- and re-packing frames is the trap this run
has paid for twice (temporal_zoom_refine -2.3, legible_timeline -1.1). Holding
the pixels fixed makes this a clean test: same evidence, added ledger.

Containment. The gate reads only the question's own words and its option format;
no video, corpus or answer-distribution knowledge enters it. When it is False --
324 of 350 questions, 198 of them already correct -- ``answer_question``
delegates straight to the parent, so those cannot change behaviour at all. When
it is True, every failure path falls back to the parent's own answer call, run
lazily so a successful ledger does not pay for a letter it discards: an empty or
unparseable ledger, a ledger the model declines to build because the number is
spoken rather than shown, a reconciliation that names no option. That last guard
matters -- two of the three gated questions F_t already answers correctly are
narrated or general-knowledge numbers with zero in-window frames, and enumerating
pixels is the wrong instrument for them, so the ledger is allowed to abstain.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..harness import extract_json_field, format_options, normalize_choice
from .aks import aks_select
from .pointer_ladder import PointerLadder, _Mem
from .stated_interval_router import parse_stated_interval
from .survey_then_commit import (
    BURST_SHARE,
    PAPER_INSTRUCT_PROMPT,
    _clamp,
)

AGENT_NAME = "cardinality_ledger"

# ---- the gate --------------------------------------------------------------
# A cardinality request in the question's own words. Deliberately about how the
# question is PHRASED, not about any subject matter.
_CARDINALITY_PHRASE = re.compile(
    r"\bhow many\b|\bhow much\b|\bnumber of\b|\bhow often\b|\bhow long\b",
    re.IGNORECASE,
)
# An option that is a bare quantity: "3", "12", "70", "$416", "60%". Anything
# with words in it ("3 minutes", "a gold coin", "1 to 3") is NOT a bare quantity,
# so relational and unit-bearing options stay on the parent's path. That is what
# keeps the gate narrow: the phrase alone scores 66.7% and must not be captured.
_BARE_QUANTITY = re.compile(r"^\$?\s*(\d+(?:\.\d+)?)\s*%?$")
# How many of the four options must be bare quantities for the answer to be a
# number the harness could in principle tally to. Three of four means one
# free-form distractor is tolerated without opening the gate to prose options.
MIN_NUMERIC_OPTIONS = 3

# ---- the ledger -----------------------------------------------------------
# Room for the enumeration to actually be written out. The parent caps its
# pointer replies at 48 tokens because it only needs timestamps; a ledger needs
# one line per occurrence, so it gets real room. This is an output-token budget,
# not a visual one -- it adds no images.
LEDGER_MAX_TOKENS = 420
# Beyond this many listed lines the reply has stopped enumerating and started
# describing; treat it as unusable and let the parent answer. Kept well above any
# plausible option value so it only catches runaway generations.
MAX_LEDGER_ITEMS = 60
# The token the ledger call is told to emit when the quantity is not something it
# can see and tally -- a spoken figure, a number written on screen, a total that
# was announced. Those are direct reads, and the parent already handles them.
ABSTAIN = "NONE"

LEDGER_PROMPT = (
    "You are given {n} frames from a video, in order, each preceded by its "
    "elapsed time as [seconds].\n\n"
    "Question that will be asked next: {question}\n{options}\n\n"
    "Do not answer it yet, and do not pick an option. This question asks for an "
    "exact quantity, so first build a list.\n"
    "List every DISTINCT thing that the question asks you to count and that you "
    "can actually see in these frames. One per line, in the form:\n"
    "  <index>. [<time>s] <short description of this specific one>\n"
    "Rules for the list:\n"
    "- Each line must be a NEW one. If you see the same one again in a later "
    "frame, do not add another line for it.\n"
    "- Two that look alike but are clearly separate occasions DO get separate "
    "lines; say briefly what makes them separate.\n"
    "- Only list what is visible in these frames. Do not infer extras that are "
    "probably there off-screen or between frames.\n"
    "- If the quantity is not something you can see and tally here -- for "
    "example it is spoken, or written as a figure on screen, or announced as a "
    "total -- then reply with the single word {abstain} and nothing else.\n"
    "Write only the list, or only {abstain}."
)

RECONCILE_PROMPT = (
    "You examined frames from a video and listed the distinct ones you could "
    "see, to answer this question.\n\n"
    "Question: {question}\n{options}\n\n"
    "Your list:\n{ledger}\n\n"
    "That list has {n_items} entries.\n"
    "Now choose the option. Use the list as your evidence: if the entries are "
    "each genuinely distinct and you found them all, the answer is how many "
    "there are. If you can tell the list is incomplete -- the frames skip time "
    "where more would have occurred -- then the true quantity is higher than "
    "{n_items}, and you should pick accordingly rather than picking {n_items}.\n"
    "Respond with only the letter (A, B, C, or D) of the correct option."
)

_ITEM_LINE = re.compile(r"^\s*(?:\d{1,2}\s*[.)\]]|[-*•])\s*\S")


def option_quantities(options: list[str]) -> list[float]:
    """Bare-quantity values among the options, in option order."""
    out: list[float] = []
    for opt in options:
        body = re.sub(r"^\s*[A-Da-d]\s*[.)]\s*", "", str(opt)).strip()
        m = _BARE_QUANTITY.match(body)
        if m:
            out.append(float(m.group(1)))
    return out


def is_cardinality_question(question: str, options: list[str]) -> bool:
    """The gate: a cardinality request whose options are bare quantities.

    Both conjuncts are required. Measured on the frontier's own traces, the
    phrase alone scores 66.7% and bare-quantity options alone score 83.3%; only
    together do they fall to 11.5%, so either half on its own would capture
    questions the parent already answers well.
    """
    if not _CARDINALITY_PHRASE.search(question or ""):
        return False
    return len(option_quantities(options)) >= MIN_NUMERIC_OPTIONS


def parse_ledger(reply: str) -> list[str]:
    """Enumerated lines from a ledger reply; empty means unusable or abstained."""
    text = (reply or "").strip()
    if not text:
        return []
    # An abstention anywhere in an otherwise-empty reply: the model was told the
    # quantity is not tallyable from these frames.
    if re.fullmatch(rf"\W*{ABSTAIN}\W*", text, re.IGNORECASE):
        return []
    items = [ln.strip() for ln in text.splitlines() if _ITEM_LINE.match(ln)]
    if not items or len(items) > MAX_LEDGER_ITEMS:
        return []
    return items


class CardinalityLedger(PointerLadder):
    """Give counting an accumulator; leave every other question to the parent."""

    # -- ingest and localisation are the parent's, untouched ---------------

    def _select_frames(
        self, memory: _Mem, question: str, options: list[str], k: int
    ) -> tuple[list[Any], str, int, Any, dict[str, Any]]:
        """The parent's answer-time frame selection, reproduced for the gated path.

        Only gated questions reach this; ungated ones delegate to the parent
        wholesale, so the 324 questions outside the gate are guaranteed
        untouched. Kept in step with the parent by using its own helpers
        (``_ladder``, ``_burst``, ``aks_select``) and its own constants rather
        than re-deriving any of them.
        """
        qv = np.asarray(self.embed_texts([question])[0], dtype=np.float32)
        scores = np.asarray(memory.img_emb, dtype=np.float32) @ qv.reshape(-1)

        span = parse_stated_interval(question, memory.duration)
        info: dict[str, Any] = {}
        if span is not None:
            centers = [0.5 * (span[0] + span[1])]
            route = "stated"
        else:
            centers, info = self._ladder(memory, question, options)
            route = f"ladder{info.get('ladder_levels', 0)}" if centers else "parent"

        if not centers:
            return _clamp([memory.frames[i] for i in aks_select(scores, k)], k), route, 0, span, info

        want = max(1, int(round(k * BURST_SHARE)))
        if span is not None and memory.video is not None:
            try:
                burst = memory.video.sample_time_range(span[0], span[1], want)
            except Exception:
                burst = []
        else:
            burst = self._burst(memory, centers, want)

        merged = {int(f.index): f for f in burst}
        n_burst = len(merged)
        for i in aks_select(scores, max(1, k - n_burst)):
            merged.setdefault(int(memory.frames[i].index), memory.frames[i])
        return _clamp([merged[i] for i in sorted(merged)], k), route, n_burst, span, info

    def _ask_letter(self, parts: list[dict[str, Any]], options: list[str]) -> tuple[str, str]:
        resp = self.ask_vlm(parts)
        return normalize_choice(
            extract_json_field(resp, "final_answer") or resp, options
        ), (resp or "")

    def answer_question(
        self, memory: _Mem, question: str, options: list[str]
    ) -> tuple[str, dict[str, Any]]:
        # Outside the gate the parent answers, unmodified. This is the whole
        # containment argument: no shared code runs before the delegation, so
        # nothing about those questions can drift.
        if not is_cardinality_question(question, options):
            return super().answer_question(memory, question, options)

        k = self.frame_budget()
        if not memory.frames:
            return "?", {"error": "no frames", "strategy": AGENT_NAME}

        chosen, route, n_burst, span, info = self._select_frames(
            memory, question, options, k
        )

        # -- the ledger: the parent's pixels, a different job ---------------
        # Asked FIRST so that the parent's single-letter call is only paid for
        # when the ledger does not produce a usable answer. Rendering the same
        # 40 frames twice would double this question's visual tokens for a
        # letter that is then discarded.
        ledger_parts = self.render_frames(chosen, timestamps=True)
        ledger_parts.append(
            {
                "type": "text",
                "text": LEDGER_PROMPT.format(
                    n=len(chosen),
                    question=question,
                    options=format_options(options),
                    abstain=ABSTAIN,
                ),
            }
        )
        try:
            ledger_reply = self.ask_vlm(ledger_parts, max_tokens=LEDGER_MAX_TOKENS)
        except Exception:
            ledger_reply = ""
        items = parse_ledger(ledger_reply)

        # -- reconcile from the ledger, text only --------------------------
        # No images: the tally is the evidence, and re-showing the frames here
        # would just invite the single-pass guess the ledger exists to replace.
        letter, raw, source = "", "", "parent_direct"
        if items:
            recon = [
                {
                    "type": "text",
                    "text": RECONCILE_PROMPT.format(
                        question=question,
                        options=format_options(options),
                        ledger="\n".join(items),
                        n_items=len(items),
                    ),
                }
            ]
            try:
                cand, cand_raw = self._ask_letter(recon, options)
            except Exception:
                cand, cand_raw = "", ""
            # Only a reply that actually names an option is allowed to answer.
            if cand and cand in {chr(ord("A") + i) for i in range(len(options))}:
                letter, raw, source = cand, cand_raw, "ledger"

        # -- fallback: the parent's own answer call, verbatim ---------------
        # Reached when the ledger abstained, was unparseable, errored, or named
        # nothing. The question then costs exactly what it costs on F_t plus the
        # refused ledger call.
        if not letter:
            base_parts = self.render_frames(chosen, timestamps=True)
            base_parts.append(
                {
                    "type": "text",
                    "text": PAPER_INSTRUCT_PROMPT.format(
                        question=question, options=format_options(options)
                    ),
                }
            )
            letter, raw = self._ask_letter(base_parts, options)
            source = "parent_direct"

        meta = {
            "strategy": AGENT_NAME,
            "sampled": len(chosen),
            "pool": len(memory.frames),
            "budget": k,
            "route": route,
            "burst_frames": n_burst,
            "gated": True,
            "answer_source": source,
            "ledger_items": len(items),
            "ledger_abstained": not items,
            "option_quantities": option_quantities(options),
            "stated_interval": (
                [round(span[0], 1), round(span[1], 1)] if span else None
            ),
            "raw": (raw or "")[:200],
        }
        meta.update(info)
        return letter, meta
