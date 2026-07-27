"""v1 operator suite: gold-action mapping, applicability, and invariants."""
import pytest

from toduq.ingest import MUSIC_EVENTS_SWITCH_TURN, SGD_1_00000_RAW, parse_dialogue
from toduq.operators import REGISTRY, all_operators, get_operator
from toduq.passes import run_chain, run_dialogue
from toduq.positioning import position_of
from toduq.routing.gold_action import OPERATOR_ACTION
from toduq.validate import check_invariants

DIALOGUE = parse_dialogue(SGD_1_00000_RAW)


def test_all_documented_operators_registered():
    # every operator with a gold-action mapping is registered and vice-versa
    assert set(REGISTRY) == set(OPERATOR_ACTION)


@pytest.mark.parametrize("op_id,expected", list(OPERATOR_ACTION.items()))
def test_operator_gold_actions_via_registry(op_id, expected):
    assert get_operator(op_id).id == op_id
    assert OPERATOR_ACTION[op_id] == expected


def test_every_sample_satisfies_invariants():
    recs = run_dialogue(
        dialogue_id="1_00000", user_turns=DIALOGUE.user_turns, operators=all_operators(),
        policy="all", turn_indices=DIALOGUE.user_turn_indices, seed=1,
    )
    assert len(recs) > 30  # many operators x many turns
    actions = {r.gold.action for r in recs}
    # all routing targets are exercised by the single-domain dialogue
    assert {"clarify", "rag_structured", "rag_unstructured", "handoff_llm", "hitl", "answer"} <= actions
    for r in recs:
        assert check_invariants(r.to_dict()) == []


def test_long_tail_only_on_mapped_verbalized_slot():
    op = get_operator("long_tail_entity")
    # applies on u1 (city) and u2 (cuisine), not on request turns
    assert op.is_applicable(DIALOGUE.user_turns[1]) is True
    assert op.is_applicable(DIALOGUE.user_turns[3]) is False


def test_cross_turn_contra_needs_carried_slot():
    op = get_operator("cross_turn_contra")
    # u1 introduces city (not carried) -> no carried slot yet
    assert op.is_applicable(DIALOGUE.user_turns[1]) is False
    # u3 carries city+cuisine from earlier turns -> applicable
    assert op.is_applicable(DIALOGUE.user_turns[3]) is True


def test_cross_service_dep_needs_multidomain():
    op = get_operator("cross_service_dep")
    assert op.is_applicable(DIALOGUE.user_turns[1]) is False       # single service
    assert op.is_applicable(MUSIC_EVENTS_SWITCH_TURN) is True       # two frames
    rec = run_chain(dialogue_id="50_00000", turn_idx=6, turn=MUSIC_EVENTS_SWITCH_TURN,
                    operator=op, position=position_of(3, 8), seed=1)
    assert rec.gold.action == "handoff_llm"
    assert check_invariants(rec.to_dict()) == []


def test_rag_structured_carries_gold_query():
    op = get_operator("out_of_kb_entity")
    rec = run_chain(dialogue_id="1_00000", turn_idx=6, turn=DIALOGUE.user_turns[3],
                    operator=op, seed=1)
    d = rec.to_dict()
    assert d["gold"]["action"] == "rag_structured"
    assert d["passes"]["document"]["gold_query"] is not None
    assert d["passes"]["document"]["gold_query"]["retrieval_kind"] == "structured"
