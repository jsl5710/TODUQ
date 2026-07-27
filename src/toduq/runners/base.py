"""Provider-agnostic LLM interface.

One protocol, many adapters. Closed models (Claude, OpenAI) and open models
(vLLM / HF / Ollama) all satisfy `LLMClient`. `sample(n)` exists from v1 so the
Prediction-uncertainty work (semantic entropy over N runs) drops in without a
refactor, even though v1 does not inject prediction uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class GenConfig:
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: Optional[int] = None
    stop: Optional[list[str]] = None


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface every model adapter implements."""

    model_id: str

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        """Single completion."""
        ...

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        """N independent completions — the basis for prediction-uncertainty metrics."""
        ...


class EchoClient:
    """Deterministic offline stub. Lets the pipeline + tests run with no network
    or API keys. Real adapters live in claude.py / openai.py / open_source.py."""

    model_id = "echo-stub"

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        return prompt.strip().splitlines()[-1] if prompt.strip() else ""

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        return [self.generate(prompt, system=system, cfg=cfg) for _ in range(n)]
