"""SGD belief-state reconstruction + provenance from the canonical JSON."""
from toduq.ingest import SGD_1_00000_RAW, parse_dialogue


def test_parse_user_turns_and_indices():
    d = parse_dialogue(SGD_1_00000_RAW)
    assert d.dialogue_id == "1_00000"
    assert d.services == ["Restaurants_1"]
    assert len(d.user_turns) == 6
    # user turns are the even absolute indices
    assert d.user_turn_indices == [0, 2, 4, 6, 8, 10]


def test_slot_values_accumulate():
    d = parse_dialogue(SGD_1_00000_RAW)
    sv = [t.belief_state["Restaurants_1"].slot_values for t in d.user_turns]
    assert sv[0] == {}
    assert sv[1] == {"city": "San Jose"}
    assert sv[2] == {"city": "San Jose", "cuisine": "American"}
    assert sv[5] == {"city": "San Jose", "cuisine": "American"}  # persists


def test_verbalized_vs_carried_provenance():
    d = parse_dialogue(SGD_1_00000_RAW)
    # city is spoken on u1, cuisine on u2; request turns verbalize nothing
    assert d.user_turns[1].verbalized_slots == {"Restaurants_1": ["city"]}
    assert d.user_turns[2].verbalized_slots == {"Restaurants_1": ["cuisine"]}
    assert d.user_turns[3].verbalized_slots == {}
    # requested slots surface on the request turns
    assert d.user_turns[3].belief_state["Restaurants_1"].requested_slots == ["street_address"]
    assert d.user_turns[4].belief_state["Restaurants_1"].requested_slots == ["phone_number"]
