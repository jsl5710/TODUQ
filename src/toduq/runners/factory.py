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


def build_generator_pool(config_path: Optional[Path] = None):
    """Build a ModelPool from the config's `generators` list (splits generation
    evenly across models). Returns None if no `generators` are configured.

    config:
      generators:
        - {adapter: ...open_source:VLLMClient, model_id: Qwen/..., endpoint: http://gpu0:8000/v1}
        - {adapter: ...open_source:VLLMClient, model_id: meta-llama/Llama-3.1-8B-Instruct, endpoint: http://gpu1:8000/v1}
        - {adapter: ...open_source:OllamaClient, model_id: mistral, endpoint: http://gpu2:11434/v1}
      generation:
        split: round_robin        # round_robin | random | weighted
        weights: [1, 1, 1]        # weighted only
        seed: 0
    """
    from toduq.runners.pool import ModelPool
    path = config_path or _CONFIG
    if not path.exists():
        return None
    cfg = _load_yaml(path)
    specs = cfg.get("generators")
    if not specs:
        return None
    clients = [build_client(s) for s in specs]
    gen = cfg.get("generation", {}) or {}
    return ModelPool(clients, strategy=gen.get("split", "round_robin"),
                     weights=gen.get("weights"), seed=int(gen.get("seed", 0)))


def build_judge_pool(config_path: Optional[Path] = None):
    """Build a ModelPool of judge CLIENTS from the config's `judges:` list, so the
    Pass-4 validation is split evenly across judge models. Returns None if none
    configured. (run_dialogue wraps each client in a Judge per site.)"""
    from toduq.runners.pool import ModelPool
    path = config_path or _CONFIG
    if not path.exists():
        return None
    cfg = _load_yaml(path)
    specs = cfg.get("judges")
    if not specs:
        return None
    clients = [build_client(s) for s in specs]
    j = cfg.get("judging", {}) or cfg.get("generation", {}) or {}
    return ModelPool(clients, strategy=j.get("split", "round_robin"),
                     weights=j.get("weights"), seed=int(j.get("seed", 0)))
