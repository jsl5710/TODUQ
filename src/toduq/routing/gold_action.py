"""Derive the gold routing action from (operator, uncertainty_type, severity).

The gold label is *derived by rule*, never chosen by an LLM, so labels stay
consistent across the dataset. Pass 2 (Document) calls `derive_action`; Pass 4
(Confirm) re-derives and checks the realized edit still maps to the same action.
"""
from __future__ import annotations

from toduq.schema import Action, Severity, UncertaintyType

# Operator -> intended action. This is the single source of truth for labels.
OPERATOR_ACTION: dict[str, Action] = {
    # Input uncertainty -> clarify
    "slot_drop": "clarify",
    "referential_ambig": "clarify",
    "multi_value": "clarify",
    "underspecify": "clarify",
    # Parameter uncertainty -> RAG / HITL
    "out_of_kb_entity": "rag_structured",
    "out_of_schema_req": "rag_unstructured",
    "long_tail_entity": "rag_structured",
    "unknowable_fact": "hitl",
    # Reasoning uncertainty -> handoff / clarify
    "cross_turn_contra": "handoff_llm",
    "implicit_constraint": "handoff_llm",
    "cross_service_dep": "handoff_llm",
    # Controls -> answer
    "paraphrase": "answer",
}

# Operators whose severity is fixed regardless of context.
OPERATOR_SEVERITY: dict[str, Severity] = {
    "paraphrase": "none",
    "unknowable_fact": "major",
}


def derive_action(operator: str, uncertainty_type: UncertaintyType, severity: Severity) -> Action:
    """Return the gold action for an operator, with a severity override for HITL.

    A `major` severity escalates any abstaining action to human-in-the-loop,
    reflecting the spec: major issue -> HITL, minor issue -> automatic route.
    """
    if operator not in OPERATOR_ACTION:
        raise KeyError(f"Unknown operator: {operator!r}")
    base = OPERATOR_ACTION[operator]
    if base == "answer":
        return base
    if severity == "major":
        return "hitl"
    return base


def derive_severity(operator: str, *, override: Severity | None = None) -> Severity:
    if override is not None:
        return override
    return OPERATOR_SEVERITY.get(operator, "minor")


def should_abstain(action: Action) -> bool:
    return action != "answer"
