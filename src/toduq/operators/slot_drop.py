"""slot_drop — remove a required filled slot to induce INPUT uncertainty.

Perturbation family. Drops a slot's value from both the utterance and the belief
state, leaving the intent underspecified. Gold action = clarify.
"""
from __future__ import annotations

from typing import Optional

from toduq.operators.base import Operator
from toduq.prompts import render_paraphrase_prompt
from toduq.routing.gold_action import derive_action, derive_severity
from toduq.runners.base import LLMClient
from toduq.schema import AnalysePass, ApplyPass, DocumentPass, Turn


class SlotDrop(Operator):
    id = "slot_drop"
    family = "perturbation"
    uncertainty_type = "input"

    def is_applicable(self, turn: Turn) -> bool:
        return self._first_filled_slot(turn) is not None

    def analyse(self, turn: Turn) -> AnalysePass:
        hit = self._first_filled_slot(turn)
        if hit is None:
            return AnalysePass(modifiable=False, rationale="No filled slot to drop.")
        service, slot, value = hit
        intent = turn.belief_state[service].active_intent
        return AnalysePass(
            modifiable=True,
            target_service=service,
            target_slot=slot,
            target_intent=intent,
            candidate_operators=["slot_drop", "referential_ambig"],
            rationale=(
                f"Turn fills required slot {slot}={value!r}. Removing it leaves "
                f"{intent} underspecified — canonical input-uncertainty site "
                f"resolvable by a single clarification."
            ),
        )

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        service = analysis.target_service
        slot = analysis.target_slot
        assert service and slot, "analyse must have found a slot before document"
        before = turn.belief_state[service].slot_values.get(slot)
        severity = derive_severity(self.id)
        action = derive_action(self.id, self.uncertainty_type, severity)
        return DocumentPass(
            operator=self.id,
            change_from=turn.utterance,
            # The canonical (template) rewrite drops the slot mention. The LLM
            # paraphrase pass produces natural variants of this in `apply`.
            change_to=_template_drop(turn.utterance, str(before)),
            slot_delta={slot: {"before": before, "after": None}},
            intended_uncertainty=self.uncertainty_type,
            expected_severity=severity,
            gold_action=action,
            gold_clarification_question=f"Which {slot.replace('_', ' ')} should I use?",
            gold_query=None,
        )

    def apply(self, turn: Turn, spec: DocumentPass, llm: Optional[LLMClient]) -> ApplyPass:
        new_state = self._clone_state(turn.belief_state)
        (slot, delta), = spec.slot_delta.items()
        for frame in new_state.values():
            frame.slot_values.pop(slot, None)

        variants: list[str] = []
        method = "template"
        if llm is not None:
            prompt = render_paraphrase_prompt(spec.change_to, n=2)
            out = llm.generate(prompt)
            variants = [ln.strip("-• ").strip() for ln in out.splitlines() if ln.strip()][:2]
            method = "template+llm_paraphrase"

        return ApplyPass(
            modified_utterance=spec.change_to,
            method=method,
            paraphrase_variants=variants,
            new_belief_state=new_state,
        )


def _template_drop(utterance: str, value: str) -> str:
    """Best-effort deterministic removal of the slot value mention.

    Kept simple on purpose: the natural rewrite comes from the LLM paraphrase
    pass; this is the reproducible fallback that anchors the label.
    """
    if value and value in utterance:
        stripped = utterance.replace(value, "somewhere").strip()
        return stripped
    return "I would like to find somewhere to eat."
