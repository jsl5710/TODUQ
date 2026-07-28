"""JHU Gateway adapter: OpenAI-compatible body, GATEWAY_KEY auth, pool split."""
import sys
import types

import pytest

from toduq.runners.factory import build_client, check_spec

_SPEC_GPT = {"adapter": "toduq.runners.gateway:GatewayClient",
             "model_id": "openai/gpt-4o-mini", "api_key_env": "GATEWAY_KEY"}
_SPEC_CLAUDE = {"adapter": "toduq.runners.gateway:GatewayClient",
                "model_id": "anthropic/claude-haiku-4.5", "api_key_env": "GATEWAY_KEY"}


def _fake_requests(capture):
    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "OK"}}]}
    def post(url, headers=None, json=None, timeout=None):
        capture.update(url=url, headers=headers, body=json, timeout=timeout)
        return _Resp()
    return types.SimpleNamespace(post=post)


def test_build_client_returns_gateway():
    c = build_client(_SPEC_GPT)
    assert c.model_id == "openai/gpt-4o-mini"
    assert c.url == "https://gateway.engineering.jhu.edu/gateway/compat/chat/completions"


def test_generate_posts_gateway_contract(monkeypatch):
    monkeypatch.setenv("GATEWAY_KEY", "fake-key")
    cap: dict = {}
    monkeypatch.setitem(sys.modules, "requests", _fake_requests(cap))
    out = build_client(_SPEC_CLAUDE).generate("hi", system="sys")
    assert out == "OK"
    assert cap["url"].endswith("/compat/chat/completions")
    assert cap["headers"]["Authorization"] == "Bearer fake-key"
    assert cap["body"]["model"] == "anthropic/claude-haiku-4.5"
    assert cap["body"]["max_completion_tokens"] == 1024 and "max_tokens" not in cap["body"]
    assert cap["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GATEWAY_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "requests", _fake_requests({}))
    with pytest.raises(RuntimeError):
        build_client(_SPEC_GPT).generate("hi")


def test_send_temperature_toggle(monkeypatch):
    monkeypatch.setenv("GATEWAY_KEY", "k")
    cap: dict = {}
    monkeypatch.setitem(sys.modules, "requests", _fake_requests(cap))
    spec = dict(_SPEC_GPT, send_temperature=False)
    build_client(spec).generate("hi")
    assert "temperature" not in cap["body"]        # reasoning-model safe


def test_check_spec_gateway_flags_missing_key(monkeypatch):
    monkeypatch.delenv("GATEWAY_KEY", raising=False)
    c = check_spec(_SPEC_GPT)
    assert c["kind"] == "gateway" and c["ok"] is False and "NOT set" in c["detail"]


def test_gateway_pool_even_split(monkeypatch, tmp_path):
    monkeypatch.setenv("GATEWAY_KEY", "k")
    monkeypatch.setitem(sys.modules, "requests", _fake_requests({}))
    from toduq.runners import ModelPool
    pool = ModelPool([build_client(_SPEC_GPT), build_client(_SPEC_CLAUDE)])
    for _ in range(10):
        pool.next()
    assert pool.summary() == {"openai/gpt-4o-mini": 5, "anthropic/claude-haiku-4.5": 5}
