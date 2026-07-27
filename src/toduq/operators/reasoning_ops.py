"""Reasoning-uncertainty operators (mixed) → gold action `handoff_llm`.

These need multi-step reasoning to resolve: a contradiction to reconcile, an
implicit constraint to infer, or a cross-service dependency to track. A single
clarification usually won't do — escalate to a stronger reasoner.
"""
from __future__ import annotations

from typing import Optional

from toduq.operators.base import Operator
from toduq.operators.parameter_ops import _InjectionOp
from toduq.routing.gold_action import derive_action, derive_severity
from toduq.runners.base import LLMClient
from toduq.schema import AnalysePass, ApplyPass, DocumentPass, Turn


class CrossTurnContra(Operator):
    """Contradict a slot value carried over from an earlier turn → handoff_llm."""
    id = "cross_turn_contra"
    family = "perturbation"
    uncertainty_type = "reasoning"

    def is_applicable(self, turn: Turn) -> bool:
        return self._carried_slot(turn) is not None

    def analyse(self, turn: Turn) -> AnalysePass:
        hit = self._carried_slot(turn)
        if hit is None:
            return AnalysePass(modifiable=False, rationale="No carried-over slot to contradict.")
        service, slot, value = hit
        return AnalysePass(modifiable=True, target_service=service, target_slot=slot,
                           target_intent=turn.belief_state[service].active_intent,
                           candidate_operators=[self.id],
                           rationale=f"Contradict carried {slot}={value!r}; two conflicting values must be reconciled → handoff.")

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        service, slot = analysis.target_service, analysis.target_slot
        before = turn.belief_state[service].slot_values[slot]
        conflict = f"__conflict__:{before}!=elsewhere"
        change_to = turn.utterance.rstrip(".?! ") + f". Actually, forget {before} — somewhere else entirely."
        severity = derive_severity(self.id)
        return DocumentPass(
            operator=self.id, change_from=turn.utterance, change_to=change_to,
            slot_delta={slot: {"before": before, "after": conflict}},
            intended_uncertainty=self.uncertainty_type, expected_severity=severity,
            gold_action=derive_action(self.id, self.uncertainty_type, severity),
            gold_clarification_question=None, gold_query=None,
        )

    def apply(self, turn: Turn, spec: DocumentPass, llm: Optional[LLMClient]) -> ApplyPass:
        new_state = self._clone_state(turn.belief_state)
        (slot, delta), = spec.slot_delta.items()
        for frame in new_state.values():
            if slot in frame.slot_values:
                frame.slot_values[slot] = delta["after"]  # mark as conflicted
        method, variants = self._maybe_paraphrase(spec.change_to, llm)
        return ApplyPass(modified_utterance=spec.change_to, method=method,
                         paraphrase_variants=variants, new_belief_state=new_state)


class ImplicitConstraint(_InjectionOp):
    """Add a constraint needing world-knowledge inference → handoff_llm."""
    id = "implicit_constraint"
    family = "injection"
    uncertainty_type = "reasoning"

    def is_applicable(self, turn: Turn) -> bool:
        return self._active(turn) is not None

    def analyse(self, turn: Turn) -> AnalysePass:
        service, intent = self._active(turn)
        return AnalysePass(modifiable=True, target_service=service, target_intent=intent,
                           candidate_operators=[self.id],
                           rationale="Add an implicit constraint requiring inference → handoff.")

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        change_to = turn.utterance.rstrip(".?! ") + ", somewhere I can bring my service dog and get gluten-free options."
        return self._document_injection(turn, change_to, service=analysis.target_service,
                                        intent=analysis.target_intent)


class CrossServiceDep(_InjectionOp):
    """At a service switch, tie the new intent to the other frame → handoff_llm.

    Needs a multi-domain turn (>1 active service) — the doc's flagged
    perturbation site. Lets us measure whether uncertainty bleeds across frames.
    """
    id = "cross_service_dep"
    family = "injection"
    uncertainty_type = "reasoning"

    def is_applicable(self, turn: Turn) -> bool:
        return len(turn.belief_state) > 1

    def analyse(self, turn: Turn) -> AnalysePass:
        services = list(turn.belief_state.keys())
        return AnalysePass(modifiable=True, target_service=services[-1],
                           target_intent=turn.belief_state[services[-1]].active_intent,
                           candidate_operators=[self.id],
                           rationale=f"Tie {services[-1]} to {services[0]} across the switch → cross-service dependency.")

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        others = [s for s in turn.belief_state if s != analysis.target_service]
        other = others[0] if others else "the other one"
        change_to = turn.utterance.rstrip(".?! ") + f" — and make sure it fits around what I already set up in {other}."
        return self._document_injection(turn, change_to, service=analysis.target_service,
                                        intent=analysis.target_intent)
