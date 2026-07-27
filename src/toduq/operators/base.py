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
        carried over from an earlier turn.

        Prefers the SGD ingest's ground-truth `verbalized_slots` provenance; falls
        back to a substring heuristic when provenance is absent (hand-built turns).
        Returns None so operators can decline turns that only reference prior state.
        """
        for service, slots in turn.verbalized_slots.items():
            frame = turn.belief_state.get(service)
            for slot in slots:
                if frame and frame.slot_values.get(slot) is not None:
                    return service, slot, frame.slot_values[slot]
        # Fallback: substring match when no provenance is available.
        if not turn.verbalized_slots:
            utt = turn.utterance.lower()
            for service, frame in turn.belief_state.items():
                for slot, value in frame.slot_values.items():
                    if value is not None and str(value).lower() in utt:
                        return service, slot, value
        return None

    @staticmethod
    def _requested_slot(turn: Turn) -> Optional[tuple[str, str]]:
        """Return (service, requested_slot) for the first per-turn requested slot,
        or None. Request turns are the natural sites for parameter/RAG operators."""
        for service, frame in turn.belief_state.items():
            if frame.requested_slots:
                return service, frame.requested_slots[0]
        return None

    @staticmethod
    def _active(turn: Turn) -> Optional[tuple[str, Optional[str]]]:
        """Return (service, active_intent) for the first frame, or None."""
        for service, frame in turn.belief_state.items():
            return service, frame.active_intent
        return None

    @staticmethod
    def _carried_slot(turn: Turn) -> Optional[tuple[str, str, object]]:
        """A filled slot carried over from an earlier turn (present in state but
        NOT introduced this turn) — the target for a cross-turn contradiction."""
        for service, frame in turn.belief_state.items():
            introduced = set(turn.introduced_slots.get(service, []))
            for slot, value in frame.slot_values.items():
                if value is not None and slot not in introduced:
                    return service, slot, value
        return None

    def _maybe_paraphrase(self, text: str, llm: Optional[LLMClient], n: int = 2):
        """Return (method, variants). Template-only when no LLM is wired."""
        if llm is None:
            return "template", []
        from toduq.prompts import render_paraphrase_prompt
        out = llm.generate(render_paraphrase_prompt(text, n=n))
        variants = [ln.strip("-• ").strip() for ln in out.splitlines() if ln.strip()][:n]
        return "template+llm_paraphrase", variants
