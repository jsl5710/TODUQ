"""Dry-run: plan the run and validate the model config without generating."""
import textwrap

from toduq.generate import plan_run
from toduq.runners.factory import check_config, check_spec, planned_split

_CFG = textwrap.dedent("""
    generators:
      - {adapter: toduq.runners.open_source:VLLMClient, model_id: qwen, endpoint: http://127.0.0.1:1/v1}
      - {adapter: toduq.runners.open_source:VLLMClient, model_id: llama, endpoint: http://127.0.0.1:1/v1}
      - {adapter: toduq.runners.open_source:VLLMClient, model_id: mistral, endpoint: http://127.0.0.1:1/v1}
    generation: {split: round_robin, seed: 0}
    judges:
      - {adapter: toduq.runners.open_source:VLLMClient, model_id: judge-a, endpoint: http://127.0.0.1:1/v1}
      - {adapter: toduq.runners.openai:OpenAIClient, model_id: gpt-4o, api_key_env: OPENAI_API_KEY}
    judging: {split: round_robin, seed: 0}
""")


def _write_cfg(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(_CFG, encoding="utf-8")
    return p


def test_plan_run_counts_units_and_multiplier():
    plan = plan_run(control_multiplier="auto")
    assert plan["control_multiplier"] > 1
    assert plan["total_units"] == plan["violation_units"] + plan["control_units"]
    # control units scale with the multiplier
    p2 = plan_run(control_multiplier=1)
    assert plan["control_units"] == p2["control_units"] * plan["control_multiplier"]
    assert plan["violation_units"] == p2["violation_units"]


def test_planned_split_is_even(tmp_path):
    cfg = _write_cfg(tmp_path)
    split = planned_split(99, "generators", "generation", config_path=cfg)
    assert split == {"qwen": 33, "llama": 33, "mistral": 33}       # no clients built
    assert planned_split(10, "generators", "generation", config_path=cfg) == \
        {"qwen": 4, "llama": 3, "mistral": 3}                       # round-robin


def test_planned_split_none_when_absent(tmp_path):
    (tmp_path / "models.yaml").write_text("roles: {}\n", encoding="utf-8")
    assert planned_split(10, "generators", "generation",
                         config_path=tmp_path / "models.yaml") is None


def test_check_config_missing_file(tmp_path):
    rep = check_config(config_path=tmp_path / "nope.yaml")
    assert rep["exists"] is False and rep["generators"] == []


def test_check_config_flags_unreachable_and_missing_key(tmp_path):
    rep = check_config(config_path=_write_cfg(tmp_path))
    assert rep["exists"] is True
    assert [c["model_id"] for c in rep["generators"]] == ["qwen", "llama", "mistral"]
    assert all(not c["ok"] for c in rep["generators"])             # endpoints down
    judge_by_id = {c["model_id"]: c for c in rep["judges"]}
    assert judge_by_id["judge-a"]["kind"] == "open" and not judge_by_id["judge-a"]["ok"]
    # closed API without the env var set is flagged, not pinged
    assert judge_by_id["gpt-4o"]["kind"] == "closed" and not judge_by_id["gpt-4o"]["ok"]


def test_check_spec_closed_key_present(monkeypatch):
    monkeypatch.setenv("MY_KEY", "x")
    c = check_spec({"adapter": "toduq.runners.claude:ClaudeClient",
                    "model_id": "claude-sonnet-5", "api_key_env": "MY_KEY"})
    assert c["ok"] and c["kind"] == "closed"
