"""JHU AI Gateway adapter — one OpenAI-compatible endpoint, many providers.

The gateway proxies OpenAI, Anthropic, and others behind a single bearer key
(`GATEWAY_KEY`) at `{base}/compat/chat/completions`, using provider-prefixed
model ids like `openai/gpt-4o-mini` or `anthropic/claude-haiku-4.5`. It differs
from the stock OpenAI API in two ways this adapter handles:

  * auth is a bearer token from an env var (default `GATEWAY_KEY`), and
  * the token budget field is `max_completion_tokens`, not `max_tokens`.

Uses `requests` directly (no `openai` SDK) so it matches the gateway's tested
contract exactly. `send_temperature=False` for reasoning models that reject it.
"""
from __future__ import annotations

import os
from typing import Optional

from toduq.runners.base import GenConfig

DEFAULT_BASE = "https://gateway.engineering.jhu.edu/gateway"


class GatewayClient:
    def __init__(self, model_id: str, *, base: Optional[str] = None,
                 api_key_env: str = "GATEWAY_KEY", max_tokens: int = 1024,
                 timeout: float = 60.0, send_temperature: bool = True):
        self.model_id = model_id
        self.base = (base or os.environ.get("GATEWAY_BASE") or DEFAULT_BASE).rstrip("/")
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.send_temperature = send_temperature

    @property
    def url(self) -> str:
        return self.base + "/compat/chat/completions"

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set — export your JHU gateway key.")
        return key

    def _messages(self, prompt: str, system: str) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _post(self, messages: list[dict], cfg: GenConfig) -> str:
        import requests
        body: dict = {"model": self.model_id, "messages": messages,
                      "max_completion_tokens": cfg.max_tokens}
        if self.send_temperature:
            body["temperature"] = cfg.temperature
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self._key()}",
                     "Content-Type": "application/json"},
            json=body, timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    def generate(self, prompt: str, *, system: str = "", cfg: Optional[GenConfig] = None) -> str:
        return self._post(self._messages(prompt, system), cfg or GenConfig())

    def sample(self, prompt: str, n: int, *, system: str = "", cfg: Optional[GenConfig] = None) -> list[str]:
        cfg = cfg or GenConfig()
        return [self.generate(prompt, system=system, cfg=cfg) for _ in range(n)]
