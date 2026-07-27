"""Operator registry — v1 scope: Input, Parameter, Reasoning (+ paraphrase control).

Prediction-uncertainty operators are v2 (measured, not injected). See
docs/injection_operators.md for the full contract and per-operator behavior.
"""
from toduq.operators.base import Operator
from toduq.operators.input_ops import MultiValue, ReferentialAmbig, Underspecify
from toduq.operators.parameter_ops import (
    LongTailEntity,
    OutOfKbEntity,
    OutOfSchemaReq,
    UnknowableFact,
)
from toduq.operators.paraphrase import Paraphrase
from toduq.operators.reasoning_ops import CrossServiceDep, CrossTurnContra, ImplicitConstraint
from toduq.operators.slot_drop import SlotDrop

_OPERATORS: list[type[Operator]] = [
    # Input → clarify
    SlotDrop, ReferentialAmbig, MultiValue, Underspecify,
    # Parameter → rag_* / hitl
    OutOfKbEntity, OutOfSchemaReq, LongTailEntity, UnknowableFact,
    # Reasoning → handoff_llm
    CrossTurnContra, ImplicitConstraint, CrossServiceDep,
    # Control → answer
    Paraphrase,
]

REGISTRY: dict[str, type[Operator]] = {op.id: op for op in _OPERATORS}


def get_operator(op_id: str) -> Operator:
    if op_id not in REGISTRY:
        raise KeyError(f"Unknown operator {op_id!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[op_id]()


def all_operators() -> list[Operator]:
    """Instantiate every registered operator (handy for run_dialogue)."""
    return [cls() for cls in _OPERATORS]


__all__ = ["Operator", "REGISTRY", "get_operator", "all_operators"]
