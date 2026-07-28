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
    if adapter.endswith("gateway:GatewayClient"):
        from toduq.runners.gateway import GatewayClient
        return GatewayClient(model_id, base=spec.get("base"),
                             api_key_env=spec.get("api_key_env", "GATEWAY_KEY"),
                             send_temperature=spec.get("send_temperature", True))
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


# --- Config validation (dry-run) --------------------------------------------

def _ping_openai_endpoint(endpoint: str, model_id: str, timeout: float = 5.0) -> tuple[bool, str]:
    """GET {endpoint}/models on an OpenAI-compatible server (vLLM/Ollama/TGI).

    Reachable + serving model_id -> (True, ...); reachable but model absent ->
    (True, warning); no response -> (False, reason). Uses only the stdlib so a
    dry-run needs no extra deps."""
    import json
    import urllib.error
    import urllib.request
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (trusted config URL)
            served = [m.get("id") for m in json.loads(r.read().decode("utf-8")).get("data", [])]
    except Exception as e:  # pragma: no cover - network-dependent
        return False, f"UNREACHABLE at {url} ({e.__class__.__name__}: {e})"
    if model_id in served:
        return True, f"reachable, serving {model_id}"
    return True, f"reachable, but {model_id!r} not in served models {served}"


def _role_spec(cfg: dict[str, Any], role: str) -> Optional[dict[str, Any]]:
    try:
        group = cfg["roles"][role]
        section, _, key = group.partition(".")
        specs = cfg.get(section, {})
        return specs[key] if key else next(iter(specs.values()))
    except Exception:
        return None


def _ping_gateway(base: str, key_env: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Cheap gateway reachability probe — GET {base}/compat/models with the bearer
    key. Any HTTP response (even 404) means the host is up + the key is present;
    only a network error is a failure. Spends no completion tokens."""
    import os
    import urllib.error
    import urllib.request
    key = os.environ.get(key_env)
    if not key:
        return False, f"env {key_env} NOT set"
    url = base.rstrip("/") + "/compat/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):  # noqa: S310 (trusted config URL)
            return True, f"reachable ({key_env} set, {url} responded)"
    except urllib.error.HTTPError as e:  # up, but /models not exposed or auth quirk
        return True, f"reachable ({key_env} set, HTTP {e.code} at {url})"
    except Exception as e:  # pragma: no cover - network-dependent
        return False, f"UNREACHABLE at {url} ({e.__class__.__name__}: {e})"


def check_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate one client spec: ping open/gateway endpoints, check keys for closed APIs."""
    import os
    adapter = spec.get("adapter", "")
    model_id = spec.get("model_id", "")
    if "open_source" in adapter:
        endpoint = spec.get("endpoint", "")
        ok, detail = _ping_openai_endpoint(endpoint, model_id) if endpoint else (False, "no endpoint set")
        return {"model_id": model_id, "kind": "open", "target": endpoint, "ok": ok, "detail": detail}
    if adapter.endswith("gateway:GatewayClient"):
        key_env = spec.get("api_key_env", "GATEWAY_KEY")
        base = spec.get("base") or os.environ.get("GATEWAY_BASE") \
            or "https://gateway.engineering.jhu.edu/gateway"
        ok, detail = _ping_gateway(base, key_env)
        return {"model_id": model_id, "kind": "gateway", "target": base, "ok": ok, "detail": detail}
    env = spec.get("api_key_env", "")
    ok = bool(env and os.environ.get(env))
    detail = f"env {env} is set" if ok else f"env {env or '<none>'} NOT set"
    return {"model_id": model_id, "kind": "closed", "target": env, "ok": ok, "detail": detail}


def planned_split(units: int, key: str, split_section: str,
                  config_path: Optional[Path] = None) -> Optional[dict[str, int]]:
    """Simulate how `units` generation units would divide across the configured
    `key` (generators/judges) — WITHOUT instantiating real clients (no SDK/network
    needed). Returns {model_id: count} or None if that list isn't configured."""
    from toduq.runners.pool import ModelPool
    path = config_path or _CONFIG
    if not path.exists():
        return None
    cfg = _load_yaml(path)
    specs = cfg.get(key)
    if not specs:
        return None
    stubs = [type("_Stub", (), {"model_id": s.get("model_id", f"model-{i}")})()
             for i, s in enumerate(specs)]
    sec = cfg.get(split_section, {}) or cfg.get("generation", {}) or {}
    pool = ModelPool(stubs, strategy=sec.get("split", "round_robin"),
                     weights=sec.get("weights"), seed=int(sec.get("seed", 0)))
    for _ in range(units):
        pool.next()
    return pool.summary()


def check_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Validate every configured generator/judge without generating. Returns
    {"config": path, "generators": [...], "judges": [...]} of check_spec results."""
    path = config_path or _CONFIG
    if not path.exists():
        return {"config": str(path), "exists": False, "generators": [], "judges": []}
    cfg = _load_yaml(path)
    gens = cfg.get("generators") or ([s] if (s := _role_spec(cfg, "generator")) else [])
    juds = cfg.get("judges") or ([s] if (s := _role_spec(cfg, "judge")) else [])
    return {"config": str(path), "exists": True,
            "generators": [check_spec(s) for s in gens],
            "judges": [check_spec(s) for s in juds]}
