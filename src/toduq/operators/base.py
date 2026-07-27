"""Operator contract: the deterministic edit that owns the gold label.

Each operator implements the four passes. `analyse` and `document` are pure and
deterministic (they set the label). `apply` may call the LLM (paraphrasing) and
`confirm` may call the judge (validation). See docs/injection_operators.md.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional

from toduq.runners.base import LLMClient
from toduq.schema import (
    AnalysePass,
    ApplyPass,
    BeliefState,
    DocumentPass,
    Family,
    Turn,
    UncertaintyType,
)


class Operator(ABC):
    id: str
    family: Family
    uncertainty_type: UncertaintyType

    @abstractmethod
    def is_applicable(self, turn: Turn) -> bool:
        """Cheap check: can this operator fire on this turn's belief state?"""

    @abstractmethod
    def analyse(self, turn: Turn) -> AnalysePass:
        """Pass 1 — locate the injection site. Deterministic."""

    @abstractmethod
    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        """Pass 2 — specify from->to + gold label. Deterministic (owns the label)."""

    @abstractmethod
    def apply(self, turn: Turn, spec: DocumentPass, llm: Optional[LLMClient]) -> ApplyPass:
        """Pass 3 — realize the edit; optionally add LLM paraphrase variants."""

    # ---- helpers shared by operators -------------------------------------
    @staticmethod
    def _clone_state(state: BeliefState) -> BeliefState:
        return copy.deepcopy(state)

    @staticmethod
    def _first_filled_slot(turn: Turn) -> Optional[tuple[str, str, object]]:
        """Return (service, slot, value) of the first filled slot, or None."""
        for service, frame in turn.belief_state.items():
            for slot, value in frame.slot_values.items():
                return service, slot, value
        return None

    @staticmethod
    def _verbalized_slot(turn: Turn) -> Optional[tuple[str, str, object]]:
        """Return (service, slot, value) for a slot whose value is actually spoken
        in THIS turn's utterance — the slot this turn introduces/mentions, not one
        carried over from an earlier turn. Prefers a verbalized slot; falls back to
        None so operators can decline turns that only reference prior state.
        """
        utt = turn.utterance.lower()
        for service, frame in turn.belief_state.items():
            for slot, value in frame.slot_values.items():
                if value is not None and str(value).lower() in utt:
                    return service, slot, value
        return None
