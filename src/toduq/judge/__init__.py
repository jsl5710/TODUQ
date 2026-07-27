"""LLM-as-judge: validates injections (Pass 4) and scores system responses (eval).

`Judge` is provider-agnostic — it wraps any `LLMClient`. The offline `NullJudge`
lets the pipeline run end-to-end without a model (everything -> needs_review),
which is the safe default so nothing auto-enters the seed set without a real judge.
"""
from __future__ import annotations

from typing import Any, Optional

from toduq.prompts import render_judge_prompt
from toduq.runners.base import LLMClient


class NullJudge:
    """Offline judge. Never approves; routes everything to human review."""

    judge_model = "null-judge"

    def validate_injection(self, change_from: str, change_to: str, intended: str,
                           gold_action: str) -> dict[str, Any]:
        return {"fidelity": "uncertain", "uncertainty_present": False,
                "naturalness": 0.0, "_offline": True}


class Judge:
    """Real judge backed by an LLMClient. Parsing is intentionally strict; a
    malformed judge reply is treated as `uncertain` (-> needs_review), never as
    a pass."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.judge_model = llm.model_id

    def validate_injection(self, change_from: str, change_to: str, intended: str,
                           gold_action: str) -> dict[str, Any]:
        prompt = render_judge_prompt(change_from, change_to, intended, gold_action)
        raw = self.llm.generate(prompt)
        return _parse_verdict(raw)


def _parse_verdict(raw: str) -> dict[str, Any]:
    import json

    try:
        data = json.loads(raw)
        return {
            "fidelity": data.get("fidelity", "uncertain"),
            "uncertainty_present": bool(data.get("uncertainty_present", False)),
            "naturalness": float(data.get("naturalness", 0.0)),
        }
    except (ValueError, TypeError):
        return {"fidelity": "uncertain", "uncertainty_present": False,
                "naturalness": 0.0, "_parse_error": True}
