"""Shared UQ core — types and the client protocol.

This package is SELF-CONTAINED (stdlib only, no imports from a parent package),
so the identical files are vendored into both TODUQ and TODUQ-MoA. It is the
single source of truth for uncertainty-quantification methods used across the
program; a future `tod-core` package would host it once.

A UQ method scores a piece of text (a user turn) in context and returns a
`UQScore`. Response-based methods take a `Client` (any object with generate /
sample); input-based methods ignore it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable


@dataclass
class UQScore:
    score: float                              # [0, 1] uncertainty magnitude
    method: str
    uncertainty_type: Optional[str] = None    # input | reasoning | parameter | prediction
    detail: dict = field(default_factory=dict)


@runtime_checkable
class Client(Protocol):
    """Minimal provider-agnostic model surface (matches TODUQ / MoA runners)."""
    def generate(self, prompt: str) -> str: ...
    def sample(self, prompt: str, n: int) -> list[str]: ...


class EchoClient:
    """Offline stub so input-based methods run with no model."""
    model_id = "echo"

    def generate(self, prompt: str) -> str:
        return prompt.strip().splitlines()[-1] if prompt.strip() else ""

    def sample(self, prompt: str, n: int) -> list[str]:
        return [self.generate(prompt) for _ in range(n)]


@runtime_checkable
class UQMethod(Protocol):
    name: str
    requires_model: bool
    def score(self, text: str, *, context: Sequence[str] = (),
              client: Optional[Client] = None) -> UQScore: ...
