"""Calibration + uncertainty metrics (Milestone 5). Pure stdlib — no numpy.

Given a system-under-test's per-turn output against the gold labels:
  - expected_calibration_error : confidence vs. correctness (binned ECE)
  - auroc                       : ranking quality of a should-abstain score
  - semantic_entropy            : dispersion across N samples (v2 prediction UQ)
  - uncertainty_bleed           : does perturbing service A move the model's
                                  state/confidence in service B? (multi-domain)
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> float:
    """Binned ECE: weighted mean |accuracy - confidence| across confidence bins."""
    n = len(confidences)
    if n == 0:
        return 0.0
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        idx = min(n_bins - 1, max(0, int(c * n_bins)))
        bins[idx].append(i)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(1 for i in b if correct[i]) / len(b)
        conf = sum(confidences[i] for i in b) / len(b)
        ece += (len(b) / n) * abs(acc - conf)
    return ece


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUROC via the Mann-Whitney U statistic (handles ties). `labels` True =
    the positive class (should-abstain). Returns 0.5 when only one class present."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return 0.5
    ranked = sorted(zip(scores, labels), key=lambda t: t[0])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based average rank for ties
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_pos_ranks = sum(r for k, r in ranks.items() if ranked[k][1])
    u = sum_pos_ranks - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def semantic_entropy(samples: Iterable[str], *, normalize: bool = True) -> float:
    """Shannon entropy over *semantic clusters* of N sampled responses.

    v1 uses exact-string clustering as a stand-in; v2 swaps in an entailment /
    embedding clustering step. High entropy => the model is unstable on this turn
    (prediction uncertainty).
    """
    import math

    counts = Counter(s.strip() for s in samples)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = -sum((c / total) * math.log(c / total) for c in counts.values())
    if normalize and len(counts) > 1:
        ent /= math.log(len(counts))
    return ent


def uncertainty_bleed(base_state: dict, perturbed_state: dict, *, perturbed_service: str) -> float:
    """Fraction of slots that changed in services OTHER than the perturbed one.

    In a multi-domain dialogue we inject uncertainty into `perturbed_service`;
    a well-behaved model should leave the *other* frames untouched. Nonzero bleed
    means uncertainty leaked across services. Compares two belief states (each
    service -> {slot: value}).
    """
    changed = total = 0
    services = set(base_state) | set(perturbed_state)
    for svc in services:
        if svc == perturbed_service:
            continue
        b = base_state.get(svc, {}) or {}
        p = perturbed_state.get(svc, {}) or {}
        for slot in set(b) | set(p):
            total += 1
            if b.get(slot) != p.get(slot):
                changed += 1
    return changed / total if total else 0.0
