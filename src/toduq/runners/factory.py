"""Build model clients by role from configs/models.yaml.

Roles: generator (paraphrase, Pass 3), judge (validation, Pass 4), and
system_under_test (evaluated in M5). Falls back to the offline EchoClient when
no config or SDK is available, so the pipeline always runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from toduq.runners.base import EchoClient, LLMClient

_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "models" / "models.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_client(spec: dict[str, Any]) -> LLMClient:
    """Instantiate a client from a single {adapter, model_id, ...} spec."""
    adapter = spec.get("adapter", "")
    model_id = spec.get("model_id", "")
    if adapter.endswith("claude:ClaudeClient"):
        from toduq.runners.claude import ClaudeClient
        return ClaudeClient(model_id or "claude-opus-5")
    if adapter.endswith("openai:OpenAIClient"):
        from toduq.runners.openai import OpenAIClient
        return OpenAIClient(model_id or "gpt-4o")
    if "open_source" in adapter:
        from toduq.runners.open_source import OllamaClient, VLLMClient
        endpoint = spec.get("endpoint", "http://localhost:8000/v1")
        ctor = OllamaClient if "Ollama" in adapter else VLLMClient
        return ctor(model_id, endpoint=endpoint)
    raise ValueError(f"Unknown adapter: {adapter!r}")


def client_for_role(role: str, config_path: Optional[Path] = None) -> LLMClient:
    """Resolve the client for a role, or EchoClient if config/SDK is missing."""
    path = config_path or _CONFIG
    if not path.exists():
        return EchoClient()
    try:
        cfg = _load_yaml(path)
        group = cfg["roles"][role]                       # e.g. "claude" or "open"
        # group names a section+key, e.g. closed.claude or open.<first>
        section, _, key = group.partition(".")
        specs = cfg.get(section, {})
        spec = specs[key] if key else next(iter(specs.values()))
        return build_client(spec)
    except Exception:
        return EchoClient()
