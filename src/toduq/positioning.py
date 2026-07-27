"""Where in the dialogue to inject — site enumeration + positional sampling.

A dialogue has many user turns. We do NOT always perturb the same turn: each
sample places the uncertainty at a *different* site so, across the dataset, the
injection is spread over early / middle / late positions. This lets us measure
whether abstention degrades with dialogue depth and, in multi-domain dialogues,
whether position relative to the service switch matters.

Flow:  user_turns + operators -> enumerate_sites (applicable only)
                              -> select_sites (stratified by position)
                              -> one Record per chosen site (see passes.pipeline)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from toduq.operators.base import Operator
from toduq.schema import Position, Turn

Band = Literal["early", "middle", "late"]
Policy = Literal["all", "one_per_turn", "stratified_position", "n_per_dialogue"]


def position_of(ordinal: int, num_user_turns: int) -> Position:
    """Compute the position of a user turn among all user turns in the dialogue."""
    n = max(1, num_user_turns)
    rel = 0.0 if n <= 1 else ordinal / (n - 1)
    band: Band = "early" if rel < 1 / 3 else "middle" if rel < 2 / 3 else "late"
    return Position(
        user_turn_ordinal=ordinal,
        num_user_turns=num_user_turns,
        relative_position=round(rel, 4),
        band=band,
    )


@dataclass
class Site:
    """One candidate injection point: a user turn + an applicable operator."""
    ordinal: int          # index among USER turns (0-based)
    turn_idx: int         # absolute SGD turn index (== ordinal if not supplied)
    operator: Operator
    position: Position


def enumerate_sites(
    user_turns: Sequence[Turn],
    operators: Sequence[Operator],
    *,
    turn_indices: Optional[Sequence[int]] = None,
) -> list[Site]:
    """All (turn, operator) pairs where the operator is applicable to the turn."""
    n = len(user_turns)
    abs_idx = list(turn_indices) if turn_indices is not None else list(range(n))
    sites: list[Site] = []
    for ordinal, turn in enumerate(user_turns):
        pos = position_of(ordinal, n)
        for op in operators:
            if op.is_applicable(turn):
                sites.append(Site(ordinal=ordinal, turn_idx=abs_idx[ordinal],
                                  operator=op, position=pos))
    return sites


def select_sites(
    sites: Sequence[Site],
    *,
    policy: Policy = "stratified_position",
    k: Optional[int] = None,
    seed: int = 0,
) -> list[Site]:
    """Pick which candidate sites become samples, spreading across positions.

    - `all`                : every applicable site (exhaustive).
    - `one_per_turn`       : at most one operator per user turn (round-robin).
    - `stratified_position`: draw evenly across early/middle/late bands so the
                             dataset is positionally balanced. `k` caps the count.
    - `n_per_dialogue`     : exactly `k` sites, maximally spread across bands.
    """
    rng = random.Random(seed)
    pool = list(sites)
    rng.shuffle(pool)  # deterministic given seed; breaks positional ordering bias

    if policy == "all":
        return sorted(pool, key=_order_key)

    if policy == "one_per_turn":
        seen: set[int] = set()
        chosen = [s for s in pool if (s.ordinal not in seen and not seen.add(s.ordinal))]
        return sorted(chosen, key=_order_key)

    # Bucket by position band for the stratified policies.
    buckets: dict[Band, list[Site]] = {"early": [], "middle": [], "late": []}
    for s in pool:
        buckets[s.position.band].append(s)

    target = k if k is not None else sum(len(v) for v in buckets.values())
    chosen: list[Site] = []
    # Round-robin across non-empty bands so no band dominates.
    order: list[Band] = ["early", "middle", "late"]
    while len(chosen) < target and any(buckets[b] for b in order):
        for b in order:
            if buckets[b] and len(chosen) < target:
                chosen.append(buckets[b].pop())
    return sorted(chosen, key=_order_key)


def _order_key(s: Site) -> tuple[int, str]:
    return (s.ordinal, s.operator.id)
