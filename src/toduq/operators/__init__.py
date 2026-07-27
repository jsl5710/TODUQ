"""Operator registry. v1 ships slot_drop + paraphrase as reference operators;
the remaining operators in docs/injection_operators.md follow the same contract.
"""
from toduq.operators.base import Operator
from toduq.operators.paraphrase import Paraphrase
from toduq.operators.slot_drop import SlotDrop

REGISTRY: dict[str, type[Operator]] = {
    SlotDrop.id: SlotDrop,
    Paraphrase.id: Paraphrase,
}


def get_operator(op_id: str) -> Operator:
    if op_id not in REGISTRY:
        raise KeyError(f"Unknown operator {op_id!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[op_id]()


__all__ = ["Operator", "REGISTRY", "get_operator", "SlotDrop", "Paraphrase"]
