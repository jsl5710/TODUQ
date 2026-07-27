"""Simulator UQ metrics — thin adapters over the shared `toduq.uq` layer.

The metric interface the simulator expects is `score(bot, turn, history) -> float`.
Every metric here delegates to a UQ method loaded from the shared registry
(`toduq.uq.load_uq`), so the SAME method implementations are used here and in
TODUQ-MoA's gate. Pick any method with `load_metric(name)`.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from toduq.simulator.bot import Chatbot
from toduq.uq import UQMethod, load_uq


@runtime_checkable
class UQMetric(Protocol):
    name: str
    def score(self, bot: Chatbot, turn: str, history: list[str]) -> float: ...


class _MetricAdapter:
    """Wraps a shared UQ method as a simulator metric (bot.llm is the client)."""
    def __init__(self, method: UQMethod):
        self._method = method
        self.name = method.name

    def score(self, bot: Chatbot, turn: str, history: list[str]) -> float:
        client = getattr(bot, "llm", None)
        return self._method.score(turn, context=history, client=client).score


def load_metric(name: str, **kwargs) -> _MetricAdapter:
    """Load any UQ metric of choice by name (see toduq.uq.available())."""
    return _MetricAdapter(load_uq(name, **kwargs))


# --- Back-compat constructors (same names as before; now backed by the registry)
def LexicalUncertaintyMetric() -> _MetricAdapter:
    return load_metric("lexical")


def SemanticEntropyMetric(n: int = 5) -> _MetricAdapter:
    return load_metric("semantic_entropy", n=n)


def VerbalizedConfidenceMetric() -> _MetricAdapter:
    return load_metric("verbalized_confidence")


def SelfConsistencyMetric(n: int = 5) -> _MetricAdapter:
    return load_metric("self_consistency", n=n)
