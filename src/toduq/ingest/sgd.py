"""Ingest SGD and normalize into TODUQ's `Turn` + belief-state + provenance form.

Parses the canonical Schema-Guided Dialogue JSON structure: a dialogue has
`turns`, each with a `speaker` (USER/SYSTEM), an `utterance`, and per-service
`frames`. USER-turn frames carry `state = {active_intent, requested_slots,
slot_values}` (cumulative) and `slots` (spans of values spoken IN this turn).

From that we reconstruct, for every user turn:
  - belief_state         : the cumulative frame state (slot_values flattened to
                           first value per slot)
  - verbalized_slots     : slots whose value is spoken in THIS utterance
                           (from the frame `slots` spans) — ground truth for
                           operator applicability
  - introduced_slots     : slots new/changed vs. the previous user turn

`load_sgd` pulls the dataset from the HF hub (needs `datasets`); `parse_dialogue`
is pure and is what the rest of the pipeline and tests use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from toduq.schema import Frame, Turn


@dataclass
class Dialogue:
    dialogue_id: str
    services: list[str]
    user_turns: list[Turn]          # user turns, in order
    user_turn_indices: list[int]    # absolute turn index of each (for `turn_idx`)


def _flatten_slot_values(raw: dict[str, Any]) -> dict[str, Any]:
    """SGD stores slot_values as {slot: [v1, v2, ...]}; keep the first value."""
    out: dict[str, Any] = {}
    for slot, values in raw.items():
        out[slot] = values[0] if isinstance(values, list) and values else values
    return out


def parse_dialogue(raw: dict[str, Any]) -> Dialogue:
    """Reconstruct user-turn belief states + provenance from an SGD dialogue."""
    dialogue_id = raw["dialogue_id"]
    services = list(raw.get("services", []))
    prev_values: dict[str, dict[str, Any]] = {}   # service -> last slot_values

    user_turns: list[Turn] = []
    user_turn_indices: list[int] = []

    for abs_idx, turn in enumerate(raw["turns"]):
        if turn.get("speaker") != "USER":
            continue

        belief: dict[str, Frame] = {}
        verbalized: dict[str, list[str]] = {}
        introduced: dict[str, list[str]] = {}

        for frame in turn.get("frames", []):
            service = frame["service"]
            state = frame.get("state", {}) or {}
            slot_values = _flatten_slot_values(state.get("slot_values", {}) or {})
            belief[service] = Frame(
                active_intent=state.get("active_intent"),
                requested_slots=list(state.get("requested_slots", []) or []),
                slot_values=slot_values,
            )
            # verbalized this turn = slots with a span in `slots`
            verbalized[service] = sorted({s["slot"] for s in frame.get("slots", []) or []})
            # introduced = slots new/changed vs. previous user turn for this service
            prev = prev_values.get(service, {})
            introduced[service] = sorted(
                k for k, v in slot_values.items() if prev.get(k) != v
            )
            prev_values[service] = slot_values

        user_turns.append(Turn(
            utterance=turn.get("utterance", ""),
            belief_state=belief,
            verbalized_slots={k: v for k, v in verbalized.items() if v},
            introduced_slots={k: v for k, v in introduced.items() if v},
        ))
        user_turn_indices.append(abs_idx)

    return Dialogue(dialogue_id, services, user_turns, user_turn_indices)


def load_sgd(split: str = "validation") -> Iterator[dict[str, Any]]:
    """Yield raw SGD dialogues from the HF hub. Requires `datasets`.

    The GEM/schema_guided_dialog flattening targets response generation and does
    not expose full per-turn frame state; for TODUQ we need the original SGD
    frames. Prefer the canonical `schema-guided-dialogue` dialogue JSON (this
    parser's input); adapt here to whichever HF export carries `turns[].frames`.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError("Install `datasets` (see pyproject extras) to load SGD.") from e
    ds = load_dataset("schema-guided-dialogue", split=split)  # noqa: F841
    raise NotImplementedError(
        "Map the chosen HF export's rows to the canonical {dialogue_id, services, "
        "turns[{speaker, utterance, frames[{service, state, slots}]}]} shape, then "
        "call parse_dialogue(). Wired against fixtures in tests/."
    )


def iter_user_turns(dialogue: Dialogue) -> Iterator[tuple[int, Turn]]:
    """Yield (absolute_turn_idx, Turn) for each user turn."""
    yield from zip(dialogue.user_turn_indices, dialogue.user_turns)


# --- Fixtures -----------------------------------------------------------------
# Verbatim SGD dialogue 1_00000 (Restaurants_1) in canonical JSON form, used to
# test reconstruction and to drive the offline demos.

SGD_1_00000_RAW: dict[str, Any] = {
    "dialogue_id": "1_00000",
    "services": ["Restaurants_1"],
    "turns": [
        {"speaker": "USER", "utterance": "I am feeling hungry so I would like to find a place to eat.",
         "frames": [{"service": "Restaurants_1", "slots": [],
                     "state": {"active_intent": "FindRestaurants", "requested_slots": [], "slot_values": {}}}]},
        {"speaker": "SYSTEM", "utterance": "Do you have a specific which you want the eating place to be located at?",
         "frames": [{"service": "Restaurants_1", "slots": [], "actions": []}]},
        {"speaker": "USER", "utterance": "I would like for it to be in San Jose.",
         "frames": [{"service": "Restaurants_1", "slots": [{"slot": "city", "start": 24, "exclusive_end": 32}],
                     "state": {"active_intent": "FindRestaurants", "requested_slots": [], "slot_values": {"city": ["San Jose"]}}}]},
        {"speaker": "SYSTEM", "utterance": "Is there a specific cuisine type you enjoy, such as Mexican, Italian or something else?",
         "frames": [{"service": "Restaurants_1", "slots": [], "actions": []}]},
        {"speaker": "USER", "utterance": "I usually like eating the American type of food.",
         "frames": [{"service": "Restaurants_1", "slots": [{"slot": "cuisine", "start": 25, "exclusive_end": 33}],
                     "state": {"active_intent": "FindRestaurants", "requested_slots": [],
                               "slot_values": {"city": ["San Jose"], "cuisine": ["American"]}}}]},
        {"speaker": "SYSTEM", "utterance": "I see that at 71 Saint Peter there is a good restaurant which is in San Jose.",
         "frames": [{"service": "Restaurants_1", "slots": [], "actions": []}]},
        {"speaker": "USER", "utterance": "Can you give me the address of this restaurant.",
         "frames": [{"service": "Restaurants_1", "slots": [],
                     "state": {"active_intent": "FindRestaurants", "requested_slots": ["street_address"],
                               "slot_values": {"city": ["San Jose"], "cuisine": ["American"]}}}]},
        {"speaker": "SYSTEM", "utterance": "If you want to go to this restaurant you can find it at 71 North San Pedro Street.",
         "frames": [{"service": "Restaurants_1", "slots": [], "actions": []}]},
        {"speaker": "USER", "utterance": "Can you give me the phone number that I can contact them with?",
         "frames": [{"service": "Restaurants_1", "slots": [],
                     "state": {"active_intent": "FindRestaurants", "requested_slots": ["phone_number"],
                               "slot_values": {"city": ["San Jose"], "cuisine": ["American"]}}}]},
        {"speaker": "SYSTEM", "utterance": "If you want to phone them you can at 408-971-8523.",
         "frames": [{"service": "Restaurants_1", "slots": [], "actions": []}]},
        {"speaker": "USER", "utterance": "Is there some other restaurant which you can suggest?",
         "frames": [{"service": "Restaurants_1", "slots": [],
                     "state": {"active_intent": "FindRestaurants", "requested_slots": [],
                               "slot_values": {"city": ["San Jose"], "cuisine": ["American"]}}}]},
        {"speaker": "SYSTEM", "utterance": "How would you like Bazille restaurant which is situated in San Jose.",
         "frames": [{"service": "Restaurants_1", "slots": [], "actions": []}]},
    ],
}

# Parsed convenience fixtures (kept for backwards compatibility with earlier code).
_DIALOGUE_1_00000 = parse_dialogue(SGD_1_00000_RAW)
RESTAURANT_DIALOGUE_USER_TURNS = _DIALOGUE_1_00000.user_turns
RESTAURANT_TURN_CITY = RESTAURANT_DIALOGUE_USER_TURNS[1]   # "...in San Jose."

MUSIC_EVENTS_SWITCH_TURN = Turn(
    utterance="Thanks! I really like music events. I enjoy rock and want to see something on March 7th.",
    belief_state={
        "Music_1": Frame(active_intent="PlaySong", requested_slots=[],
                         slot_values={"song_name": "Lost Stars", "playback_device": "TV"}),
        "Events_1": Frame(active_intent="FindEvents", requested_slots=[],
                          slot_values={"category": "Music", "subcategory": "rock", "date": "March 7th"}),
    },
    verbalized_slots={"Events_1": ["subcategory", "date"]},
    introduced_slots={"Events_1": ["category", "subcategory", "date"]},
)
