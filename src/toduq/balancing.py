"""Class balancing for the seed set — equalize the positive and negative classes.

The seed set applies many *violation / uncertainty* operators but only a single
*control* operator per turn, so the raw output is heavily skewed toward the
positive (should-flag / should-abstain) class. Two levers bring it to balance:

  1. control_multiplier — emit N distinct control (paraphrase) variants per turn
     so the negative class grows to meet the positives. With a live sampling model
     each variant differs; the split is realized at generation time
     (see `run_dialogue`). `auto_control_multiplier` sizes it from the operator mix.
  2. balance() — a final, deterministic trim of whichever class is still the
     majority down to `ratio * minority`, so the shipped set is exactly balanced.

Nothing is silently discarded: balance() returns the dropped rows plus a report,
and generate_seed keeps the full (unbalanced) accepted set alongside the balanced
one (records_all.jsonl).
"""
from __future__ import annotations

import random
from typing import Any, Callable


def auto_control_multiplier(operators: list, is_control: Callable[[Any], bool]) -> int:
    """Pick a control multiplier so control sites ≈ violation sites per turn.

    E.g. 11 violation operators + 1 control operator → 11, so each turn yields ~11
    control variants to match its ~11 violation samples. The exact split is then
    guaranteed by `balance`; this only gets the two classes into the same ballpark
    cheaply (control edits are label-preserving paraphrases)."""
    n_ctrl = sum(1 for o in operators if is_control(o))
    n_other = sum(1 for o in operators if not is_control(o))
    return max(1, round(n_other / max(1, n_ctrl)))


def balance(records: list, *, is_positive: Callable[[Any], bool],
            ratio: float = 1.0, seed: int = 0) -> tuple[list, list, dict[str, Any]]:
    """Undersample the majority class so |majority| ≤ round(ratio · |minority|).

    ratio=1.0 gives an exact 1:1 split. Deterministic given `seed` (a seeded
    shuffle picks which majority rows to keep). Returns (kept, dropped, report);
    if either class is empty nothing is dropped.
    """
    pos = [r for r in records if is_positive(r)]
    neg = [r for r in records if not is_positive(r)]
    report: dict[str, Any] = {"positive": len(pos), "negative": len(neg),
                              "target_ratio": ratio}
    if not pos or not neg:
        report.update(majority_class="none", dropped=0, kept_positive=len(pos),
                      kept_negative=len(neg), balanced_total=len(records))
        return list(records), [], report

    if len(pos) >= len(neg):
        majority, minority, maj_name = pos, neg, "positive"
    else:
        majority, minority, maj_name = neg, pos, "negative"
    cap = min(len(majority), int(round(ratio * len(minority))))
    rng = random.Random(seed)
    order = list(range(len(majority)))
    rng.shuffle(order)
    keep = set(order[:cap])
    kept_majority = [m for i, m in enumerate(majority) if i in keep]
    dropped = [m for i, m in enumerate(majority) if i not in keep]
    kept = minority + kept_majority
    report.update(
        majority_class=maj_name, dropped=len(dropped),
        kept_positive=(cap if maj_name == "positive" else len(pos)),
        kept_negative=(cap if maj_name == "negative" else len(neg)),
        balanced_total=len(kept),
    )
    return kept, dropped, report
