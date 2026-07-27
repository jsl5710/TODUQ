"""UQ registry — load any method of choice by name.

    from <pkg>.uq import load_uq, available
    uq = load_uq("semantic_entropy", n=8)
    uq.score("book a table somewhere", context=[...], client=my_llm)
"""
from __future__ import annotations

from .base import UQMethod
from .methods import (
    LexicalUQ,
    SelfConsistencyUQ,
    SemanticEntropyUQ,
    VerbalizedConfidenceUQ,
)

_BUILDERS = {
    "lexical": LexicalUQ,
    "semantic_entropy": SemanticEntropyUQ,
    "self_consistency": SelfConsistencyUQ,
    "verbalized_confidence": VerbalizedConfidenceUQ,
}


def available() -> list[str]:
    return sorted(_BUILDERS)


def load_uq(name: str, **kwargs) -> UQMethod:
    if name not in _BUILDERS:
        raise KeyError(f"Unknown UQ method {name!r}. Available: {available()}")
    return _BUILDERS[name](**kwargs)
