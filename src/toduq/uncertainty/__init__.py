"""Uncertainty taxonomy constants (see docs/taxonomy.md).

The dimension -> source mapping and the dimension -> default-action mapping are
the reference other modules import so the taxonomy has one home.
"""
DIMENSIONS = ("input", "reasoning", "parameter", "prediction")

DIMENSION_SOURCE = {
    "input": "aleatoric",
    "reasoning": "mixed",
    "parameter": "epistemic",
    "prediction": "mixed",
}

DIMENSION_DEFAULT_ACTION = {
    "input": "clarify",
    "reasoning": "handoff_llm",
    "parameter": "rag_structured",
    "prediction": "handoff_llm",
}

__all__ = ["DIMENSIONS", "DIMENSION_SOURCE", "DIMENSION_DEFAULT_ACTION"]
