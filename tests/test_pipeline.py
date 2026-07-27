"""End-to-end pass-chain tests, runnable offline (no network, no API keys)."""
from toduq.ingest import RESTAURANT_TURN_CITY
from toduq.operators import get_operator
from toduq.passes import run_chain
from toduq.validate import check_invariants


def test_slot_drop_chain_offline():
    rec = run_chain(
        dialogue_id="1_00000", turn_idx=2, turn=RESTAURANT_TURN_CITY,
        operator=get_operator("slot_drop"), llm=None, seed=42,
    )
    assert rec is not None
    d = rec.to_dict()

    # pass keys are all retrievable
    assert set(d["passes"]) == {"analyse", "document", "apply", "confirm", "edit"}

    # pass 5 finalizes the canonical version
    assert d["passes"]["edit"]["mode"] == "copy"          # structural change landed
    assert d["passes"]["edit"]["final_status"] == "finalized"
    assert d["passes"]["edit"]["final_utterance"] == d["passes"]["apply"]["modified_utterance"]
    assert d["passes"]["edit"]["final_belief_state"]["Restaurants_1"]["slot_values"] == {}

    # label integrity
    assert d["gold"]["action"] == "clarify"
    assert d["gold"]["should_abstain"] is True
    assert d["passes"]["document"]["slot_delta"]["city"]["after"] is None

    # structural check: city slot really removed from the new belief state
    assert d["passes"]["apply"]["new_belief_state"]["Restaurants_1"]["slot_values"] == {}
    assert d["passes"]["confirm"]["change_applied"] is True

    # offline (NullJudge) never auto-accepts
    assert d["passes"]["confirm"]["status"] == "needs_review"

    assert check_invariants(d) == []


def test_paraphrase_is_answer_control():
    rec = run_chain(
        dialogue_id="1_00000", turn_idx=2, turn=RESTAURANT_TURN_CITY,
        operator=get_operator("paraphrase"), llm=None, seed=1,
    )
    d = rec.to_dict()
    assert d["gold"]["action"] == "answer"
    assert d["gold"]["should_abstain"] is False
    assert d["passes"]["document"]["slot_delta"] == {}
    assert check_invariants(d) == []


def test_edit_repairs_missed_slot_drop():
    # Simulate a defective apply: confirm says the change did NOT land (slot
    # still present). Pass 5 must repair it deterministically and finalize.
    from toduq.passes.pipeline import _edit
    from toduq.schema import ApplyPass, ConfirmPass, DocumentPass, Frame, Turn

    turn = Turn(utterance="in San Jose",
                belief_state={"Restaurants_1": Frame(slot_values={"city": "San Jose"})})
    document = DocumentPass(
        operator="slot_drop", change_from="in San Jose", change_to="somewhere",
        slot_delta={"city": {"before": "San Jose", "after": None}},
        intended_uncertainty="input", expected_severity="minor", gold_action="clarify",
    )
    # apply "forgot" to drop the slot from the belief state
    apply = ApplyPass(modified_utterance="somewhere", method="template",
                      new_belief_state={"Restaurants_1": Frame(slot_values={"city": "San Jose"})})
    confirm = ConfirmPass(change_applied=False, status="rejected",
                          structural_checks={"slot_city_removed": False})

    edit = _edit(turn, document, apply, confirm)
    assert edit.mode == "repair"
    assert edit.final_status == "finalized"
    assert "city" not in edit.final_belief_state["Restaurants_1"].slot_values
    assert edit.changes  # records what it fixed


def test_unknown_operator_raises():
    import pytest

    with pytest.raises(KeyError):
        get_operator("does_not_exist")
