"""The chain-of-passes orchestrator: analyse -> document -> apply -> confirm.

Produces one `Record` per (turn, operator). The deterministic passes (analyse,
document) set the label; apply/confirm may use an LLM + judge. See
docs/pass_chain.md.
"""
from __future__ import annotations

from typing import Optional

from toduq.judge import Judge, NullJudge
from toduq.operators.base import Operator
from toduq.routing.gold_action import should_abstain
from toduq.runners.base import LLMClient
from toduq.schema import (
    ConfirmPass,
    Gold,
    Provenance,
    Record,
    Turn,
)

UNCERTAINTY_SOURCE = {
    "input": "aleatoric",
    "reasoning": "mixed",
    "parameter": "epistemic",
    "prediction": "mixed",
}


def run_chain(
    *,
    dialogue_id: str,
    turn_idx: int,
    turn: Turn,
    operator: Operator,
    llm: Optional[LLMClient] = None,
    judge: Optional[Judge | NullJudge] = None,
    seed: int = 0,
    sgd_version: str = "GEM/schema_guided_dialog",
) -> Optional[Record]:
    """Run all four passes. Returns None if the turn is not a viable site."""
    judge = judge or NullJudge()

    # Pass 1 — Analyse
    analysis = operator.analyse(turn)
    if not analysis.modifiable:
        return None

    # Pass 2 — Document (owns the gold label)
    document = operator.document(turn, analysis)

    # Pass 3 — Apply
    apply = operator.apply(turn, document, llm)

    # Pass 4 — Confirm: structural checks first, then judge gate.
    structural = _structural_checks(turn, document, apply)
    verdict = judge.validate_injection(
        document.change_from, apply.modified_utterance,
        document.intended_uncertainty, document.gold_action,
    )
    status = _decide_status(operator, structural, verdict)
    confirm = ConfirmPass(
        change_applied=all(structural.values()),
        status=status,
        structural_checks=structural,
        judge_verdict=verdict,
        notes="",
    )

    gold = Gold(
        action=document.gold_action,
        severity=document.expected_severity,
        should_abstain=should_abstain(document.gold_action),
        clarification_question=document.gold_clarification_question,
        query=document.gold_query,
    )

    return Record(
        record_id=f"{dialogue_id}:{turn_idx}:{operator.id}:{seed}",
        dialogue_id=dialogue_id,
        turn_idx=turn_idx,
        services=list(turn.belief_state.keys()) or ["unknown"],
        family=operator.family,
        operator=operator.id,
        uncertainty_type=document.intended_uncertainty,
        uncertainty_source=UNCERTAINTY_SOURCE[document.intended_uncertainty],
        source=turn,
        passes_analyse=analysis,
        passes_document=document,
        passes_apply=apply,
        passes_confirm=confirm,
        gold=gold,
        provenance=Provenance(
            sgd_version=sgd_version,
            seed=seed,
            generator_model=getattr(llm, "model_id", None),
            judge_model=getattr(judge, "judge_model", None),
        ),
    )


def _structural_checks(turn: Turn, document, apply) -> dict[str, bool]:
    """Deterministic verification that the documented edit actually landed."""
    checks: dict[str, bool] = {}
    # Every dropped slot must be gone from the new belief state.
    for slot, delta in document.slot_delta.items():
        if delta.get("after") is None:
            gone = all(slot not in fr.slot_values for fr in apply.new_belief_state.values())
            checks[f"slot_{slot}_removed"] = gone
    # Paraphrase controls must not touch slots.
    if document.operator == "paraphrase":
        checks["slots_unchanged"] = document.slot_delta == {}
    # The utterance must actually have changed unless it's an identity control.
    checks["utterance_changed_or_control"] = (
        apply.modified_utterance != turn.utterance or document.operator == "paraphrase"
    )
    return checks or {"noop": True}


def _decide_status(operator: Operator, structural: dict[str, bool], verdict: dict) -> str:
    if not all(structural.values()):
        return "rejected"
    if verdict.get("_offline") or verdict.get("_parse_error"):
        return "needs_review"
    # Controls: accept if the judge confirms meaning preserved (fidelity pass).
    if operator.family == "paraphrase":
        return "accepted" if verdict.get("fidelity") == "pass" else "needs_review"
    if verdict.get("fidelity") == "pass" and verdict.get("uncertainty_present"):
        return "accepted"
    if verdict.get("fidelity") == "fail":
        return "rejected"
    return "needs_review"
