"""Evaluation harness (milestone 5).

Given a system-under-test's per-turn output (abstain?, action, [clarification]),
score against the gold labels:

  - abstention_accuracy : abstain iff gold.should_abstain
  - routing_accuracy    : predicted action == gold.action (given abstention)
  - calibration         : ECE / AUROC on should-abstain (needs numpy/sklearn)
  - uncertainty_bleed   : for multi-domain turns, does perturbing service A shift
                          the model's confidence/state in service B?

Deterministic metrics live here; LLM-judged metrics (clarification quality) call
the judge. Kept dependency-light: calibration functions import numpy lazily.
"""
from __future__ import annotations

from typing import Iterable


def abstention_accuracy(preds: Iterable[bool], gold: Iterable[bool]) -> float:
    preds, gold = list(preds), list(gold)
    if not gold:
        return 0.0
    return sum(p == g for p, g in zip(preds, gold)) / len(gold)


def routing_accuracy(pred_actions: Iterable[str], gold_actions: Iterable[str]) -> float:
    """Accuracy over turns where the gold action is an abstention (!= answer)."""
    pairs = [(p, g) for p, g in zip(pred_actions, gold_actions) if g != "answer"]
    if not pairs:
        return 0.0
    return sum(p == g for p, g in pairs) / len(pairs)


def over_abstention_rate(preds: Iterable[bool], gold: Iterable[bool]) -> float:
    """Fraction of should-answer turns where the model wrongly abstained."""
    pairs = [(p, g) for p, g in zip(preds, gold) if g is False]
    if not pairs:
        return 0.0
    return sum(1 for p, _ in pairs if p) / len(pairs)


__all__ = ["abstention_accuracy", "routing_accuracy", "over_abstention_rate"]
