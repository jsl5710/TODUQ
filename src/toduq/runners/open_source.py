"""Open-source model adapters via OpenAI-compatible HTTP endpoints.

vLLM (`--api-server`) and Ollama both expose an OpenAI-compatible `/v1` API, so a
single adapter covers open-weight models (Llama, Qwen, Mistral, ...) served
locally. This keeps the system-under-test swappable between closed and open
models with no pipeline changes.
"""
from __future__ import annotations

from typing import Optional

from toduq.runners.base import GenConfig


class OpenAICompatClient:
    """Talks to any OpenAI-compatible endpoint (vLLM / Ollama / TGI)."""

    def __init__(self, model_id: str, *, endpoint: str, api_key: str = "not-needed",
                 max_tokens: int = 1024):
        try:
            import openai
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("pip install 'toduq[open]' or `openai` for the HTTP client.") from e
        self.client = openai.OpenAI(base_url=endpoint, api_key=api_key)
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
        return [self.generate(prompt, system=system, cfg=cfg) for _ in range(n)]


def VLLMClient(model_id: str, endpoint: str = "http://localhost:8000/v1", **kw) -> OpenAICompatClient:
    return OpenAICompatClient(model_id, endpoint=endpoint, **kw)


def OllamaClient(model_id: str, endpoint: str = "http://localhost:11434/v1", **kw) -> OpenAICompatClient:
    return OpenAICompatClient(model_id, endpoint=endpoint, **kw)
