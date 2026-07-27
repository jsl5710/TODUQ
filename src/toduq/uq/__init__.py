"""Shared UQ layer — one implementation, loadable by name, used across the program.

Vendored identically into TODUQ and TODUQ-MoA (destined for a shared `tod-core`).
Pick any method with `load_uq(name)`:

    from <pkg>.uq import load_uq, available          # available() lists methods
    uq = load_uq("lexical")                          # offline, no model
    uq = load_uq("semantic_entropy", n=8)            # response-based, needs a client
    result = uq.score(text, context=history, client=llm)   # -> UQScore(score, type, ...)
"""
from .base import Client, EchoClient, UQMethod, UQScore
from .methods import (
    LexicalUQ,
    SelfConsistencyUQ,
    SemanticEntropyUQ,
    VerbalizedConfidenceUQ,
)
from .registry import available, load_uq

__all__ = [
    "UQScore", "UQMethod", "Client", "EchoClient",
    "LexicalUQ", "SemanticEntropyUQ", "SelfConsistencyUQ", "VerbalizedConfidenceUQ",
    "load_uq", "available",
]
