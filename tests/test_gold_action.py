"""The gold-action derivation table is the single source of truth for labels."""
import pytest

from toduq.routing.gold_action import derive_action, derive_severity, should_abstain


@pytest.mark.parametrize("op,expected", [
    ("slot_drop", "clarify"),
    ("referential_ambig", "clarify"),
    ("out_of_kb_entity", "rag_structured"),
    ("out_of_schema_req", "rag_unstructured"),
    ("cross_turn_contra", "handoff_llm"),
    ("paraphrase", "answer"),
])
def test_minor_severity_actions(op, expected):
    sev = derive_severity(op)
    assert derive_action(op, "input", sev) == expected


def test_major_severity_escalates_to_hitl():
    # unknowable_fact is fixed-major -> hitl
    assert derive_action("unknowable_fact", "parameter", derive_severity("unknowable_fact")) == "hitl"
    # any abstaining operator escalates to hitl when severity is major
    assert derive_action("slot_drop", "input", "major") == "hitl"


def test_answer_never_escalates():
    assert derive_action("paraphrase", "input", "major") == "answer"


def test_should_abstain():
    assert should_abstain("clarify") is True
    assert should_abstain("answer") is False


def test_unknown_operator_raises():
    with pytest.raises(KeyError):
        derive_action("nope", "input", "minor")
