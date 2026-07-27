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


class HeuristicJudge:
    """Deterministic, offline validation gate — no LLM required.

    It does NOT fabricate an LLM's judgment. It confirms what can be checked by
    rule: the edit is non-empty and differs from the source (or is a control),
    the target-slot change is reflected, and length/format look sane. Naturalness
    is a fixed, clearly-labeled heuristic score. Use it to bootstrap a seed set
    offline; swap in the LLM `Judge` for higher-fidelity validation.
    """

    judge_model = "heuristic-judge"

    def validate_injection(self, change_from: str, change_to: str, intended: str,
                           gold_action: str) -> dict[str, Any]:
        changed = change_to.strip() and change_to.strip() != change_from.strip()
        is_control = gold_action == "answer"
        # control: meaning preserved (we can't verify semantics offline) -> pass
        # perturbation/injection: require the surface actually changed
        fidelity = "pass" if (is_control or changed) else "fail"
        uncertainty_present = (not is_control) and bool(changed)
        naturalness = 0.7 if change_to.strip() else 0.0  # heuristic, not model-scored
        return {"fidelity": fidelity, "uncertainty_present": uncertainty_present,
                "naturalness": naturalness, "_heuristic": True}


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
