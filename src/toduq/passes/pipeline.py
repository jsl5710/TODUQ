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
    EditPass,
    Gold,
    Position,
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
    position: Optional[Position] = None,
    llm: Optional[LLMClient] = None,
    judge: Optional[Judge | NullJudge] = None,
    seed: int = 0,
    sgd_version: str = "GEM/schema_guided_dialog",
) -> Optional[Record]:
    """Run all five passes on ONE site. Returns None if the turn is not viable.

    `position` locates this turn among the dialogue's user turns; when omitted
    (single-turn use) it defaults to a lone early turn. Use `run_dialogue` to
    spread injections across positions within a dialogue.
    """
    judge = judge or NullJudge()
    if position is None:
        position = Position(user_turn_ordinal=0, num_user_turns=1,
                            relative_position=0.0, band="early")

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

    # Pass 5 — Edit: finalize. Copy the confirmed output forward, or repair a
    # defect Pass 4 flagged, producing the single canonical final version.
    edit = _edit(turn, document, apply, confirm)

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
        position=position,
        source=turn,
        passes_analyse=analysis,
        passes_document=document,
        passes_apply=apply,
        passes_confirm=confirm,
        passes_edit=edit,
        gold=gold,
        provenance=Provenance(
            sgd_version=sgd_version,
            seed=seed,
            generator_model=getattr(llm, "model_id", None),
            judge_model=getattr(judge, "judge_model", None),
        ),
    )


def run_dialogue(
    *,
    dialogue_id: str,
    user_turns: list[Turn],
    operators: list[Operator],
    policy: str = "stratified_position",
    k: Optional[int] = None,
    turn_indices: Optional[list[int]] = None,
    llm: Optional[LLMClient] = None,
    pool: Optional["object"] = None,
    workers: int = 0,
    judge: Optional[Judge | NullJudge] = None,
    seed: int = 0,
    sgd_version: str = "GEM/schema_guided_dialog",
) -> list[Record]:
    """Generate multiple samples from ONE dialogue, each injecting uncertainty at
    a DIFFERENT user turn.

    Enumerates every applicable (turn, operator) site, selects a positionally
    balanced subset (see toduq.positioning), and runs the 5-pass chain on each.

    If `pool` (a ModelPool) is given, each site's paraphrase (Pass 3) is generated
    by the next model in the pool — splitting the work evenly across models and
    recording the assigned model in each record's provenance.
    """
    # Imported here to avoid a circular import (positioning imports schema only).
    from toduq.positioning import enumerate_sites, select_sites

    sites = enumerate_sites(user_turns, operators, turn_indices=turn_indices)
    chosen = select_sites(sites, policy=policy, k=k, seed=seed)

    # Assign a model per site FIRST (sequential -> deterministic, even split),
    # then optionally execute the sites concurrently across model endpoints.
    assignments = [(pool.next() if pool is not None else llm) for _ in chosen]

    def _run(site, client):
        return run_chain(
            dialogue_id=dialogue_id, turn_idx=site.turn_idx,
            turn=user_turns[site.ordinal], operator=site.operator,
            position=site.position, llm=client, judge=judge, seed=seed,
            sgd_version=sgd_version,
        )

    if workers and workers > 1 and len(chosen) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda sc: _run(*sc), zip(chosen, assignments)))
    else:
        results = [_run(s, c) for s, c in zip(chosen, assignments)]
    return [r for r in results if r is not None]


def _edit(turn: Turn, document, apply, confirm) -> EditPass:
    """Pass 5 — finalize / repair.

    - If Pass 4 confirmed the change landed structurally, `copy` the applied
      output forward as the canonical final version (the "copy the final
      version" case). The human-review gate in `confirm.status` is untouched.
    - If a structural check failed (something was missed), `repair`
      deterministically by re-enforcing the documented `slot_delta`, then
      re-verify. If it still can't be made consistent, mark `unresolved`.
    """
    import copy

    final_state = copy.deepcopy(apply.new_belief_state)
    final_utterance = apply.modified_utterance
    changes: list[str] = []

    if confirm.change_applied:
        mode = "copy"
    else:
        mode = "repair"
        # Re-enforce every documented slot edit that didn't take.
        for slot, delta in document.slot_delta.items():
            if delta.get("after") is None:
                for frame in final_state.values():
                    if slot in frame.slot_values:
                        frame.slot_values.pop(slot, None)
                        changes.append(f"repaired: removed lingering slot '{slot}'")
            else:
                for frame in final_state.values():
                    if frame.slot_values.get(slot) != delta["after"]:
                        frame.slot_values[slot] = delta["after"]
                        changes.append(f"repaired: set slot '{slot}' to documented value")

    # Re-verify the final version against the documented spec.
    repaired_ok = _slot_delta_satisfied(document, final_state)
    final_status = "finalized" if repaired_ok else "unresolved"
    notes = "" if repaired_ok else "structural spec unsatisfiable by deterministic repair; route to human."

    return EditPass(
        mode=mode,
        final_utterance=final_utterance,
        final_belief_state=final_state,
        final_status=final_status,
        changes=changes,
        notes=notes,
    )


def _slot_delta_satisfied(document, state) -> bool:
    for slot, delta in document.slot_delta.items():
        after = delta.get("after")
        if after is None:
            if any(slot in fr.slot_values for fr in state.values()):
                return False
        else:
            if not any(fr.slot_values.get(slot) == after for fr in state.values()):
                return False
    return True


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
