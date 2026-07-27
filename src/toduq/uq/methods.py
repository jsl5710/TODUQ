"""UQ method implementations. Add a class here and register it in registry.py.

- LexicalUQ           : input-based, offline. Hedge/underspecification markers +
                        unknowable-fact markers. Types: input / parameter.
- SemanticEntropyUQ   : response-based. Normalized entropy over N sampled responses
                        (semantic clustering; exact-string stand-in for v1).
- SelfConsistencyUQ   : response-based. 1 - agreement (top-cluster fraction) over N
                        samples. Cheaper cousin of semantic entropy.
- VerbalizedConfidenceUQ : response-based. Asks the model its confidence; 1 - conf.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional, Sequence

from .base import Client, UQScore

_HEDGE_WORDS = {"somewhere", "something", "anywhere", "someone", "somehow",
                "maybe", "or", "whatever", "wherever"}
_HEDGE_PHRASES = ("that one", "not sure", "over there", "or something", "or else")
_UNKNOWABLE = ("will it", "next week", "next friday", "tomorrow", "in the future", "guarantee")


def _tod_prompt(text: str, context: Sequence[str]) -> str:
    ctx = "\n".join(list(context)[-6:])
    return (f"You are a task-oriented dialogue assistant.\nDialogue so far:\n{ctx}\n\n"
            f"User: {text}\nAssistant:")


def _clusters(samples) -> Counter:
    """Semantic clustering stand-in (exact-string). Swap for entailment/embedding."""
    return Counter(s.strip() for s in samples)


class LexicalUQ:
    name = "lexical"
    requires_model = False

    def score(self, text: str, *, context: Sequence[str] = (),
              client: Optional[Client] = None) -> UQScore:
        low = text.lower()
        words = set(re.findall(r"[a-z']+", low))
        hits = len(words & _HEDGE_WORDS) + sum(1 for p in _HEDGE_PHRASES if p in low)
        utype = "input" if hits else None
        if any(u in low for u in _UNKNOWABLE):
            hits = max(hits, 2)
            utype = "parameter"
        return UQScore(min(1.0, hits / 2.0), self.name, utype, {"hits": hits})


class SemanticEntropyUQ:
    name = "semantic_entropy"
    requires_model = True

    def __init__(self, n: int = 5):
        self.n = n

    def score(self, text: str, *, context: Sequence[str] = (),
              client: Optional[Client] = None) -> UQScore:
        if client is None:
            return UQScore(0.0, self.name, "prediction", {"error": "no client"})
        samples = client.sample(_tod_prompt(text, context), self.n)
        counts = _clusters(samples)
        total = sum(counts.values())
        if total == 0:
            return UQScore(0.0, self.name, "prediction", {"clusters": 0})
        ent = -sum((c / total) * math.log(c / total) for c in counts.values())
        if len(counts) > 1:
            ent /= math.log(len(counts))
        return UQScore(min(1.0, ent), self.name, "prediction", {"clusters": len(counts)})


class SelfConsistencyUQ:
    name = "self_consistency"
    requires_model = True

    def __init__(self, n: int = 5):
        self.n = n

    def score(self, text: str, *, context: Sequence[str] = (),
              client: Optional[Client] = None) -> UQScore:
        if client is None:
            return UQScore(0.0, self.name, "prediction", {"error": "no client"})
        samples = client.sample(_tod_prompt(text, context), self.n)
        counts = _clusters(samples)
        total = sum(counts.values()) or 1
        agreement = max(counts.values()) / total if counts else 1.0
        return UQScore(1.0 - agreement, self.name, "prediction", {"agreement": round(agreement, 3)})


class VerbalizedConfidenceUQ:
    name = "verbalized_confidence"
    requires_model = True

    _PROMPT = ("On a scale from 0.0 (no idea) to 1.0 (certain), how confident are you "
               "that you can fully answer this user turn without more information? "
               "Reply with only the number.\nUser: {text}")

    def score(self, text: str, *, context: Sequence[str] = (),
              client: Optional[Client] = None) -> UQScore:
        if client is None:
            return UQScore(0.0, self.name, None, {"error": "no client"})
        raw = client.generate(self._PROMPT.format(text=text))
        m = re.search(r"[0-1](?:\.\d+)?", raw)
        conf = float(m.group()) if m else 0.5
        return UQScore(max(0.0, min(1.0, 1.0 - conf)), self.name, None, {"confidence": conf})
