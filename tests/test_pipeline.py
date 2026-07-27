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
    assert set(d["passes"]) == {"analyse", "document", "apply", "confirm"}

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


def test_unknown_operator_raises():
    import pytest

    with pytest.raises(KeyError):
        get_operator("does_not_exist")
