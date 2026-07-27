"""Anthropic (Claude) adapter for the LLMClient protocol.

Uses the official `anthropic` SDK. Current Claude models (Opus 5 / Sonnet 5 /
Opus 4.8) reject `temperature`/`top_p`/`top_k` and `budget_tokens`, so this
adapter deliberately omits sampling params and steers depth with
`output_config.effort`. Refusals surface as `stop_reason == "refusal"` on an
HTTP 200 — handled here rather than crashing.

Credentials resolve the SDK's normal way (ANTHROPIC_API_KEY, or an
`ant auth login` profile); do not hardcode a key.
"""
from __future__ import annotations

from typing import Optional

from toduq.runners.base import GenConfig

DEFAULT_MODEL = "claude-opus-5"  # latest Claude; override via config


class ClaudeClient:
    def __init__(self, model_id: str = DEFAULT_MODEL, *, effort: str = "medium",
                 max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("pip install 'toduq[closed]' (needs `anthropic`).") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()   # resolves creds from env/profile
        self.model_id = model_id
        self.effort = effort
        self.max_tokens = max_tokens

    def _call(self, prompt: str, system: str, cfg: Optional[GenConfig]) -> str:
        max_tokens = cfg.max_tokens if cfg else self.max_tokens
        kwargs = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
        )
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            return ""  # judge/pipeline treats empty as needs_review, never a pass
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        return self._call(prompt, system, cfg)

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        # Current Claude models don't expose temperature; sampling variance comes
        # from the model's own stochasticity across independent calls. This is the
        # basis for v2 prediction-uncertainty (semantic entropy over N runs).
        return [self._call(prompt, system, cfg) for _ in range(n)]
