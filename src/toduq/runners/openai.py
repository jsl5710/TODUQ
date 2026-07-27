"""OpenAI adapter for the LLMClient protocol (closed-source option).

Uses the official `openai` SDK. Kept minimal; sampling params are supported here
(unlike the Claude adapter) since OpenAI models accept temperature.
"""
from __future__ import annotations

from typing import Optional

from toduq.runners.base import GenConfig


class OpenAIClient:
    def __init__(self, model_id: str = "gpt-4o", *, max_tokens: int = 1024):
        try:
            import openai
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("pip install 'toduq[closed]' (needs `openai`).") from e
        self.client = openai.OpenAI()  # resolves OPENAI_API_KEY from env
        self.model_id = model_id
        self.max_tokens = max_tokens

    def _messages(self, prompt: str, system: str) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        cfg = cfg or GenConfig()
        resp = self.client.chat.completions.create(
            model=self.model_id, messages=self._messages(prompt, system),
            max_tokens=cfg.max_tokens, temperature=cfg.temperature, top_p=cfg.top_p,
        )
        return resp.choices[0].message.content or ""

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        cfg = cfg or GenConfig()
        resp = self.client.chat.completions.create(
            model=self.model_id, messages=self._messages(prompt, system),
            max_tokens=cfg.max_tokens, temperature=cfg.temperature, top_p=cfg.top_p, n=n,
        )
        return [c.message.content or "" for c in resp.choices]
