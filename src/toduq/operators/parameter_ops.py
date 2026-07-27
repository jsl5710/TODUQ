"""Parameter-uncertainty operators (epistemic) → RAG / HITL.

These probe knowledge gaps: entities/facts outside the model's or the service
DB's knowledge. Structured-answerable → rag_structured (with a gold query);
needs free text → rag_unstructured; unknowable → hitl (major).
"""
from __future__ import annotations

from typing import Optional

from toduq.operators.base import Operator
from toduq.routing.gold_action import derive_action, derive_severity
from toduq.runners.base import LLMClient
from toduq.schema import AnalysePass, ApplyPass, DocumentPass, GoldQuery, Turn

# A rare / long-tail substitute per slot family (DB likely lacks it).
_LONG_TAIL = {"cuisine": "Ainu", "city": "Zapata", "category": "Kabaddi"}


class _InjectionOp(Operator):
    """Base for operators that ADD content (no slot edit); belief state unchanged."""

    def _document_injection(self, turn: Turn, change_to: str, *, service: str,
                            intent, gold_query=None, clarify=None) -> DocumentPass:
        severity = derive_severity(self.id)
        return DocumentPass(
            operator=self.id, change_from=turn.utterance, change_to=change_to,
            slot_delta={},  # injection: no slot edit
            intended_uncertainty=self.uncertainty_type, expected_severity=severity,
            gold_action=derive_action(self.id, self.uncertainty_type, severity),
            gold_clarification_question=clarify, gold_query=gold_query,
        )

    def apply(self, turn: Turn, spec: DocumentPass, llm: Optional[LLMClient]) -> ApplyPass:
        method, variants = self._maybe_paraphrase(spec.change_to, llm)
        return ApplyPass(modified_utterance=spec.change_to, method=method,
                         paraphrase_variants=variants,
                         new_belief_state=self._clone_state(turn.belief_state))


class OutOfKbEntity(_InjectionOp):
    """Ask about a specific entity absent from the service DB → rag_structured."""
    id = "out_of_kb_entity"
    family = "injection"
    uncertainty_type = "parameter"

    def is_applicable(self, turn: Turn) -> bool:
        return self._requested_slot(turn) is not None or self._active(turn) is not None

    def analyse(self, turn: Turn) -> AnalysePass:
        service, intent = self._active(turn)
        return AnalysePass(modifiable=True, target_service=service, target_intent=intent,
                           candidate_operators=[self.id],
                           rationale="Inject a specific out-of-KB entity → DB lookup (rag_structured).")

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        service, intent = analysis.target_service, analysis.target_intent
        entity = "Zorka's Ethiopian Kitchen on 5th"
        change_to = turn.utterance.rstrip(".?! ") + f", specifically {entity}?"
        query = GoldQuery(service=service, intent=intent or "",
                          constraints={"entity_name": entity}, retrieval_kind="structured")
        return self._document_injection(turn, change_to, service=service, intent=intent,
                                        gold_query=query)


class OutOfSchemaReq(_InjectionOp):
    """Ask for information the service schema has no slot for → rag_unstructured."""
    id = "out_of_schema_req"
    family = "injection"
    uncertainty_type = "parameter"

    def is_applicable(self, turn: Turn) -> bool:
        return self._active(turn) is not None

    def analyse(self, turn: Turn) -> AnalysePass:
        service, intent = self._active(turn)
        return AnalysePass(modifiable=True, target_service=service, target_intent=intent,
                           candidate_operators=[self.id],
                           rationale="Ask for out-of-schema info (needs free text) → rag_unstructured.")

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        service, intent = analysis.target_service, analysis.target_intent
        change_to = turn.utterance.rstrip() + " Also, is the place wheelchair accessible?"
        query = GoldQuery(service=service, intent=intent or "",
                          constraints={"question": "wheelchair_accessible"}, retrieval_kind="unstructured")
        return self._document_injection(turn, change_to, service=service, intent=intent,
                                        gold_query=query)


class UnknowableFact(_InjectionOp):
    """Ask something no source can answer (future/private) → hitl (major)."""
    id = "unknowable_fact"
    family = "injection"
    uncertainty_type = "parameter"

    def is_applicable(self, turn: Turn) -> bool:
        return self._active(turn) is not None

    def analyse(self, turn: Turn) -> AnalysePass:
        service, intent = self._active(turn)
        return AnalysePass(modifiable=True, target_service=service, target_intent=intent,
                           candidate_operators=[self.id],
                           rationale="Unanswerable/high-stakes request → abstain to human (hitl).")

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        change_to = turn.utterance.rstrip() + " Will it be busy there next Friday at 8pm?"
        return self._document_injection(turn, change_to, service=analysis.target_service,
                                        intent=analysis.target_intent)


class LongTailEntity(Operator):
    """Swap a verbalized slot value for a rare/long-tail one → rag_structured."""
    id = "long_tail_entity"
    family = "perturbation"
    uncertainty_type = "parameter"

    def is_applicable(self, turn: Turn) -> bool:
        hit = self._verbalized_slot(turn)
        return hit is not None and hit[1] in _LONG_TAIL

    def analyse(self, turn: Turn) -> AnalysePass:
        hit = self._verbalized_slot(turn)
        if hit is None or hit[1] not in _LONG_TAIL:
            return AnalysePass(modifiable=False, rationale="No verbalized slot with a long-tail variant.")
        service, slot, value = hit
        return AnalysePass(modifiable=True, target_service=service, target_slot=slot,
                           target_intent=turn.belief_state[service].active_intent,
                           candidate_operators=[self.id],
                           rationale=f"Swap {slot}={value!r} for a rare value the DB likely lacks → rag_structured.")

    def document(self, turn: Turn, analysis: AnalysePass) -> DocumentPass:
        service, slot = analysis.target_service, analysis.target_slot
        before = turn.belief_state[service].slot_values[slot]
        rare = _LONG_TAIL[slot]
        change_to = turn.utterance.replace(str(before), rare)
        severity = derive_severity(self.id)
        query = GoldQuery(service=service, intent=analysis.target_intent or "",
                          constraints={slot: rare}, retrieval_kind="structured")
        return DocumentPass(
            operator=self.id, change_from=turn.utterance, change_to=change_to,
            slot_delta={slot: {"before": before, "after": rare}},
            intended_uncertainty=self.uncertainty_type, expected_severity=severity,
            gold_action=derive_action(self.id, self.uncertainty_type, severity),
            gold_clarification_question=None, gold_query=query,
        )

    def apply(self, turn: Turn, spec: DocumentPass, llm: Optional[LLMClient]) -> ApplyPass:
        new_state = self._clone_state(turn.belief_state)
        (slot, delta), = spec.slot_delta.items()
        for frame in new_state.values():
            if slot in frame.slot_values:
                frame.slot_values[slot] = delta["after"]
        method, variants = self._maybe_paraphrase(spec.change_to, llm)
        return ApplyPass(modified_utterance=spec.change_to, method=method,
                         paraphrase_variants=variants, new_belief_state=new_state)
