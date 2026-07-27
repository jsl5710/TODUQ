"""Injection is spread across dialogue positions, not fixed to one turn."""
from toduq.ingest import RESTAURANT_DIALOGUE_USER_TURNS
from toduq.operators import get_operator
from toduq.passes import run_dialogue
from toduq.positioning import enumerate_sites, position_of, select_sites
from toduq.validate import check_invariants


def test_position_bands():
    assert position_of(0, 6).band == "early"
    assert position_of(3, 6).band == "middle"
    assert position_of(5, 6).band == "late"
    assert position_of(0, 1).relative_position == 0.0


def test_enumerate_only_verbalized_slot_drop_sites():
    # slot_drop fires only where the turn SAYS a slot value: turn 1 ("San Jose")
    # and turn 2 ("American"). Turn 0 has no slots; turns 3-5 only request info.
    ops = [get_operator("slot_drop")]
    sites = enumerate_sites(RESTAURANT_DIALOGUE_USER_TURNS, ops)
    assert sorted(s.ordinal for s in sites) == [1, 2]


def test_stratified_selection_spreads_across_bands():
    # paraphrase is applicable to every turn, so all six positions are candidates.
    ops = [get_operator("paraphrase")]
    sites = enumerate_sites(RESTAURANT_DIALOGUE_USER_TURNS, ops)
    chosen = select_sites(sites, policy="stratified_position", k=3, seed=7)
    bands = {s.position.band for s in chosen}
    assert len(chosen) == 3
    assert bands == {"early", "middle", "late"}  # one from each band


def test_run_dialogue_injects_at_different_turns():
    ops = [get_operator("slot_drop"), get_operator("paraphrase")]
    records = run_dialogue(
        dialogue_id="1_00000",
        user_turns=RESTAURANT_DIALOGUE_USER_TURNS,
        operators=ops,
        policy="one_per_turn",
        seed=3,
    )
    # one sample per user turn, each at a DISTINCT position spanning the dialogue
    ordinals = sorted(r.position.user_turn_ordinal for r in records)
    assert ordinals == [0, 1, 2, 3, 4, 5]
    assert {r.position.band for r in records} == {"early", "middle", "late"}
    for r in records:
        d = r.to_dict()
        assert d["position"]["num_user_turns"] == 6
        assert d["passes"]["edit"]["final_status"] == "finalized"
        assert check_invariants(d) == []
