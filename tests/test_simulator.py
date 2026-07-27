"""TODUQ Simulator: replay a sample and test UQ-metric localization (offline)."""
from toduq.ingest import RESTAURANT_DIALOGUE_USER_TURNS, SGD_1_00000_RAW, parse_dialogue
from toduq.operators import all_operators, get_operator
from toduq.passes import run_chain, run_dialogue
from toduq.positioning import position_of
from toduq.simulator import Chatbot, LexicalUncertaintyMetric, perturbed_user_turns, simulate_record

D = parse_dialogue(SGD_1_00000_RAW)
TURNS = RESTAURANT_DIALOGUE_USER_TURNS


def _record(op_id):
    recs = run_dialogue(dialogue_id=D.dialogue_id, user_turns=TURNS,
                        operators=all_operators(), turn_indices=D.user_turn_indices,
                        policy="all", seed=1)
    return next(r for r in recs if r.operator == op_id)


def test_perturbed_dialogue_replaces_only_target_turn():
    rec = _record("slot_drop")
    turns = perturbed_user_turns(TURNS, rec)
    inj = rec.position.user_turn_ordinal
    assert turns[inj].utterance == rec.passes_edit.final_utterance
    assert all(turns[i].utterance == TURNS[i].utterance for i in range(len(TURNS)) if i != inj)


def test_lexical_metric_localizes_slot_drop():
    rec = _record("slot_drop")
    res = simulate_record(rec, TURNS, Chatbot(), LexicalUncertaintyMetric())
    assert res.predicted_ordinal == res.injected_ordinal
    assert res.rank_of_injected == 1
    assert res.identified is True


def test_control_is_identified_when_no_turn_spikes():
    rec = _record("paraphrase")
    res = simulate_record(rec, TURNS, Chatbot(), LexicalUncertaintyMetric())
    assert res.should_abstain is False
    assert all(ts.score == 0 for ts in res.turn_scores)
    assert res.identified is True   # control correctly localized == no spike


def test_metric_never_sees_the_label():
    # the metric scores every turn identically regardless of which is injected:
    # scoring the same utterance in two different records yields the same score.
    metric = LexicalUncertaintyMetric()
    s1 = metric.score(Chatbot(), "I would like to find somewhere to eat.", [])
    s2 = metric.score(Chatbot(), "I would like to find somewhere to eat.", ["User: hi"])
    assert s1 == s2 > 0


def test_scores_are_per_turn_and_bounded():
    rec = _record("underspecify")
    res = simulate_record(rec, TURNS, Chatbot(), LexicalUncertaintyMetric())
    assert len(res.turn_scores) == len(TURNS)
    assert all(0.0 <= ts.score <= 1.0 for ts in res.turn_scores)


def test_mode_recorded_and_intrinsic_metric_is_mode_invariant():
    rec = _record("slot_drop")
    hist = simulate_record(rec, TURNS, Chatbot(), LexicalUncertaintyMetric(), mode="history")
    imm = simulate_record(rec, TURNS, Chatbot(), LexicalUncertaintyMetric(), mode="immediate")
    assert hist.mode == "history" and imm.mode == "immediate"
    # lexical is intrinsic (ignores context) -> same scores in both modes
    assert [t.score for t in hist.turn_scores] == [t.score for t in imm.turn_scores]


def test_mode_changes_context_for_context_aware_metric():
    # a context-aware metric: high uncertainty only when history is present
    from toduq.simulator.metrics import _MetricAdapter

    class _CtxMethod:
        name = "ctx"
        def score(self, text, *, context=(), client=None):
            from toduq.uq import UQScore
            return UQScore(1.0 if context else 0.0, self.name)

    metric = _MetricAdapter(_CtxMethod())
    rec = _record("slot_drop")  # injected at ordinal 1, so later turns have history
    hist = simulate_record(rec, TURNS, Chatbot(), metric, mode="history")
    imm = simulate_record(rec, TURNS, Chatbot(), metric, mode="immediate")
    assert imm.turn_scores[-1].score == 0.0          # no context -> 0
    assert hist.turn_scores[-1].score == 1.0          # has history -> 1
    assert hist.turn_scores[0].score == 0.0           # first turn: no prior history either way
