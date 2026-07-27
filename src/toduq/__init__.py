"""TODUQ — Task-Oriented Dialogue Uncertainty Quantification.

Controlled uncertainty injection over the Schema-Guided Dialogue dataset via a
deterministic 4-pass chain (analyse → document → apply → confirm), emitting one
JSON record per turn in which each pass is retrievable by key.
"""

__version__ = "0.1.0"

ACTIONS = (
    "answer",
    "clarify",
    "rag_structured",
    "rag_unstructured",
    "handoff_llm",
    "hitl",
)

UNCERTAINTY_TYPES = ("input", "reasoning", "parameter", "prediction")
FAMILIES = ("paraphrase", "perturbation", "injection")
SEVERITIES = ("none", "minor", "major")
