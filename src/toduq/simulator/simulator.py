"""TODUQ Simulator.

Takes a single TODUQ sample (a Record whose target turn carries injected /
paraphrased / perturbed uncertainty), reconstructs the perturbed dialogue,
replays it turn-by-turn through a chatbot module, scores each user turn with a UQ
metric, and reports whether the metric flags uncertainty at the injected turn.

    result = simulate_record(record, original_user_turns, bot, metric)
    result.identified   # did the metric peak at the injected turn?
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from toduq.schema import Turn
from toduq.simulator.bot import Chatbot
from toduq.simulator.metrics import UQMetric


@dataclass
class TurnScore:
    ordinal: int
    utterance: str
    score: float
    is_injected: bool


@dataclass
class SimResult:
    metric: str
    operator: str
    injected_ordinal: int
    should_abstain: bool               # False for coherent controls
    turn_scores: list[TurnScore] = field(default_factory=list)
    predicted_ordinal: Optional[int] = None
    rank_of_injected: Optional[int] = None
    identified: bool = False           # metric peaked at the injected turn (and it should)
    mode: str = "history"              # "immediate" (turn alone) | "history" (turn + prior context)


def perturbed_user_turns(original: list[Turn], record) -> list[Turn]:
    """Original user turns with the target turn replaced by the sample's final
    (perturbed) utterance."""
    turns = copy.deepcopy(original)
    ordinal = record.position.user_turn_ordinal
    turns[ordinal] = Turn(
        utterance=record.passes_edit.final_utterance,
        belief_state=turns[ordinal].belief_state,
    )
    return turns


def simulate_record(record, original_user_turns: list[Turn], bot: Chatbot,
                    metric: UQMetric, *, mode: str = "history") -> SimResult:
    """Replay a sample and score each turn's uncertainty.

    mode:
      - "history"   : the metric sees the turn PLUS the prior conversation history
                      (the accumulated context, if any).
      - "immediate" : the metric sees only the current turn, in isolation.
    """
    if mode not in ("history", "immediate"):
        raise ValueError(f"mode must be 'history' or 'immediate', got {mode!r}")
    turns = perturbed_user_turns(original_user_turns, record)
    injected = record.position.user_turn_ordinal
    should_abstain = record.gold.should_abstain

    history: list[str] = []
    scores: list[TurnScore] = []
    for i, t in enumerate(turns):
        context = history if mode == "history" else []
        s = metric.score(bot, t.utterance, context)
        scores.append(TurnScore(i, t.utterance, round(s, 4), i == injected))
        # advance the dialogue regardless, so history mode has context to use
        resp = bot.respond(t.utterance, history)
        history += [f"User: {t.utterance}", f"Assistant: {resp}"]

    # rank the injected turn by score (1 = highest); argmax = predicted injection
    ordered = sorted(scores, key=lambda x: x.score, reverse=True)
    predicted = ordered[0].ordinal if ordered else None
    rank = next((r for r, ts in enumerate(ordered, 1) if ts.is_injected), None)
    # a control (should_abstain False) is "identified correctly" iff NO turn spikes
    if should_abstain:
        identified = predicted == injected and ordered[0].score > 0
    else:
        identified = all(ts.score == 0 for ts in scores)

    return SimResult(metric=metric.name, operator=record.operator,
                     injected_ordinal=injected, should_abstain=should_abstain,
                     turn_scores=scores, predicted_ordinal=predicted,
                     rank_of_injected=rank, identified=identified, mode=mode)
