"""ModelPool — split the generation workload across several models.

For a server run with multiple open models (Qwen, Llama 3.1, Mistral, ...) served
on different endpoints, a `ModelPool` hands out a client per generation site so
the work is divided **evenly** (round-robin) — or by explicit weights. Assignment
is deterministic (site order is seeded), and every record records which model
produced it (via `provenance.generator_model`).

    pool = ModelPool([qwen, llama, mistral])   # strategy="round_robin"
    client = pool.next()                        # cycles: qwen, llama, mistral, qwen, ...
    ...
    pool.summary()   # {'Qwen/...': 34, 'meta-llama/...': 33, 'mistral': 33}
"""
from __future__ import annotations

import random
from typing import Optional

from toduq.runners.base import GenConfig, LLMClient


class ModelPool:
    def __init__(self, clients: list[LLMClient], *, strategy: str = "round_robin",
                 weights: Optional[list[float]] = None, seed: int = 0):
        if not clients:
            raise ValueError("ModelPool needs at least one client")
        if strategy not in ("round_robin", "random", "weighted"):
            raise ValueError(f"unknown strategy {strategy!r}")
        if strategy == "weighted" and (not weights or len(weights) != len(clients)):
            raise ValueError("weighted strategy needs one weight per client")
        self.clients = clients
        self.labels = [getattr(c, "model_id", f"model-{i}") for i, c in enumerate(clients)]
        self.strategy = strategy
        self.weights = weights
        self._i = 0
        self._rng = random.Random(seed)
        self.counts: dict[str, int] = {label: 0 for label in self.labels}

    # The pool doubles as an LLMClient so it can drop into any `llm=` slot; each
    # generate/sample call advances the rotation (one call per record's paraphrase).
    model_id = "model-pool"

    def next(self) -> LLMClient:
        if self.strategy == "round_robin":
            idx = self._i % len(self.clients)
            self._i += 1
        elif self.strategy == "weighted":
            idx = self._rng.choices(range(len(self.clients)), weights=self.weights, k=1)[0]
        else:  # random (uniform)
            idx = self._rng.randrange(len(self.clients))
        self.counts[self.labels[idx]] += 1
        return self.clients[idx]

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        return self.next().generate(prompt, system=system, cfg=cfg)

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        return self.next().sample(prompt, n, system=system, cfg=cfg)

    def summary(self) -> dict[str, int]:
        return dict(self.counts)

    def reset_counts(self) -> None:
        self.counts = {label: 0 for label in self.labels}
