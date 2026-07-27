"""Input-uncertainty operators (aleatoric) → gold action `clarify`.

All three target a slot VERBALIZED in the turn and make it ambiguous or
underspecified without changing the intent, so a single clarification resolves it.
"""
from __future__ import annotations

from typing import Optional

from toduq.operators.base import Operator
from toduq.routing.gold_action import derive_action, derive_severity
from toduq.runners.base import LLMClient
from toduq.schema import AnalysePass, ApplyPass, DocumentPass, Turn

# Referent phrasing by rough slot family, for a natural surface form.
_REFERENT = {"city": "there", "location": "there", "cuisine": "that type",
             "category": "that kind", "date": "then"}
# A plausible alternate value per slot family, for multi-value ambiguity.
_ALT_VALUE = {"cuisine": "Italian", "category": "Sports", "city": "Oakland"}
# A vague generalization per slot family, for underspecification.
_VAGUE = {"city": "somewhere nearby", "location": "somewhere around there",
          "cuisine": "some kind of food", "date": "sometime soon"}


class _InputSlotOp(Operator):
    """Shared analyse/scaffolding for input operators that edit a verbalized slot."""
    family = "perturbation"
    uncertainty_type = "input"

    def is_applicable(self, turn: Turn) -> bool:
        return self._verbalized_slot(turn) is not None

    def analyse(self, turn: Turn) -> AnalysePass:
        hit = self._verbalized_slot(turn)
        if hit is None:
            return AnalysePass(modifiable=False, rationale="No slot verbalized in this turn.")
        service, slot, value = hit
        return AnalysePass(
            modifiable=True, target_service=service, target_slot=slot,
            target_intent=turn.belief_state[service].active_intent,
            candidate_operators=[self.id],
            rationale=f"Turn verbalizes {slot}={value!r}; {self.id} makes it ambiguous → clarify.",
        )

    def _clarify_q(self, slot: str) -> str:
        return f"Which {slot.replace('_', ' ')} do you mean?"

    def _document(self, turn: Turn, analysis: AnalysePass, change_to: str,
                  after) -> DocumentPass:
        slot = analysis.target_slot
        service = analysis.target_service
        before = turn.belief_state[service].slot_values.get(slot)
        severity = derive_severity(self.id)
        return DocumentPass(
            operator=self.id, change_from=turn.utterance, change_to=change_to,
            slot_delta={slot: {"before": before, "after": after}},
            intended_uncertainty=self.uncertainty_type, expected_severity=severity,
            gold_action=derive_action(self.id, self.uncertainty_type, severity),
            gold_clarification_question=self._clarify_q(slot), gold_query=None,
        )

    def apply(self, turn: Turn, spec: DocumentPass, llm: Optional[LLMClient]) -> ApplyPass:
        new_state = self._clone_state(turn.belief_state)
        (slot, delta), = spec.slot_delta.items()
        after = delta["after"]
        for frame in new_state.values():
            if slot in frame.slot_values:
                if after is None:
                    frame.slot_values.pop(slot, None)
                else:
                    frame.slot_values[slot] = after
        method, variants = self._maybe_paraphrase(spec.change_to, llm)
        return ApplyPass(modified_utterance=spec.change_to, method=method,
                         paraphrase_variants=variants, new_belief_state=new_state)


class ReferentialAmbig(_InputSlotOp):
    id = "referential_ambig"

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        slot, value = analysis.target_slot, turn.belief_state[analysis.target_service].slot_values[analysis.target_slot]
        ref = _REFERENT.get(slot, "that one")
        change_to = turn.utterance.replace(str(value), ref)
        return self._document(turn, analysis, change_to, after=None)  # unresolved referent


class MultiValue(_InputSlotOp):
    id = "multi_value"

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        slot = analysis.target_slot
        value = turn.belief_state[analysis.target_service].slot_values[slot]
        alt = _ALT_VALUE.get(slot, "something else")
        change_to = turn.utterance.replace(str(value), f"{value} or {alt}")
        doc = self._document(turn, analysis, change_to, after=[value, alt])  # ambiguous multi
        doc.gold_clarification_question = f"Would you prefer {value} or {alt}?"
        return doc


class Underspecify(_InputSlotOp):
    id = "underspecify"

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        slot = analysis.target_slot
        value = turn.belief_state[analysis.target_service].slot_values[slot]
        vague = _VAGUE.get(slot, "something like that")
        change_to = turn.utterance.replace(str(value), vague)
        doc = self._document(turn, analysis, change_to, after=None)
        doc.gold_clarification_question = f"Could you be more specific about the {slot.replace('_', ' ')}?"
        return doc
