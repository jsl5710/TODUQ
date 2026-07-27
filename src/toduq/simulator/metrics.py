"""Uncertainty-quantification metrics for the simulator.

Each metric scores ONE user turn (given the bot + history) with an uncertainty
value in [0, 1]. Metrics never see the injected-turn label — the simulator scores
every turn and then checks whether the metric peaks at the injected turn.

- LexicalUncertaintyMetric : input-based, offline. Scores the USER turn's surface
  for hedges / underspecification / dangling referents. Catches INPUT-type
  injections (slot_drop, referential_ambig, multi_value, underspecify) without a
  model, and demonstrably fails on parameter/reasoning injections — which is the
  point: UQ-metric coverage varies by uncertainty type.
- SemanticEntropyMetric   : response-based. Samples the bot N times and computes
  semantic entropy over the responses (needs a live model; identical samples ->
  entropy 0, so EchoBot is degenerate here by design).
- VerbalizedConfidenceMetric : asks the model its confidence; score = 1 - conf.
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from toduq.eval.metrics import semantic_entropy
from toduq.simulator.bot import Chatbot

_HEDGE_WORDS = {"somewhere", "something", "anywhere", "someone", "somehow",
                "maybe", "or", "whatever", "wherever"}
_HEDGE_PHRASES = ("that one", "not sure", "over there", "or something", "or else")


@runtime_checkable
class UQMetric(Protocol):
    name: str
    def score(self, bot: Chatbot, turn: str, history: list[str]) -> float: ...


class LexicalUncertaintyMetric:
    """Offline, input-based. Fraction-weighted count of uncertainty markers in the
    user turn, squashed to [0, 1]. Deterministic — good for reproducible demos."""
    name = "lexical"

    def score(self, bot: Chatbot, turn: str, history: list[str]) -> float:
        low = turn.lower()
        words = set(re.findall(r"[a-z']+", low))
        hits = len(words & _HEDGE_WORDS)
        hits += sum(1 for p in _HEDGE_PHRASES if p in low)
        return min(1.0, hits / 2.0)


class SemanticEntropyMetric:
    """Response-based. Semantic entropy over N bot samples (needs a live model)."""
    name = "semantic_entropy"

    def __init__(self, n: int = 5):
        self.n = n

    def score(self, bot: Chatbot, turn: str, history: list[str]) -> float:
        samples = bot.sample(turn, history, self.n)
        return semantic_entropy(samples)  # already normalized to [0, 1] when >1 cluster


class VerbalizedConfidenceMetric:
    """Ask the model how confident it is; score = 1 - confidence (needs a model)."""
    name = "verbalized_confidence"

    _PROMPT = ("On a scale from 0.0 (no idea) to 1.0 (certain), how confident are you "
               "that you can fully answer this user turn without more information? "
               "Reply with only the number.\nUser: {turn}")

    def score(self, bot: Chatbot, turn: str, history: list[str]) -> float:
        raw = bot.llm.generate(self._PROMPT.format(turn=turn))
        m = re.search(r"[0-1](?:\.\d+)?", raw)
        conf = float(m.group()) if m else 0.5
        return max(0.0, min(1.0, 1.0 - conf))
