"""Ingest SGD and normalize turns into TODUQ's `Turn` + belief-state form.

Loads GEM/schema_guided_dialog from the HF hub and reconstructs the per-service
frame state (active_intent, requested_slots, slot_values) after each user turn.

Network + `datasets` are only needed here; the rest of the pipeline runs offline.
The two doc examples (1_00000, 50_00000) are provided as fixtures for tests.
"""
from __future__ import annotations

from typing import Iterator

from toduq.schema import Frame, Turn


def load_sgd(split: str = "validation"):
    """Yield raw SGD dialogues from the HF hub. Requires `datasets`."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError("Install extras: pip install 'toduq[.]' or `datasets`") from e
    return load_dataset("GEM/schema_guided_dialog", split=split)


def iter_user_turns(dialogue: dict) -> Iterator[tuple[int, Turn]]:
    """Reconstruct belief state after each USER turn.

    NOTE: field names below follow the GEM/schema_guided_dialog schema and are
    pinned in tests against fixtures; adjust here if the HF schema shifts.
    """
    raise NotImplementedError(
        "Belief-state reconstruction is implemented against fixtures in "
        "tests/fixtures/; wire the HF field mapping here in milestone 2."
    )


# --- Fixtures: the two verbatim examples from SGD_dialogue_flows.docx ----------

RESTAURANT_TURN_CITY = Turn(
    utterance="I would like for it to be in San Jose.",
    belief_state={
        "Restaurants_1": Frame(
            active_intent="FindRestaurants",
            requested_slots=[],
            slot_values={"city": "San Jose"},
        )
    },
)

MUSIC_EVENTS_SWITCH_TURN = Turn(
    utterance="Thanks! I really like music events. I enjoy rock and want to see something on March 7th.",
    belief_state={
        "Music_1": Frame(active_intent="PlaySong", requested_slots=[],
                         slot_values={"song_name": "Lost Stars", "playback_device": "TV"}),
        "Events_1": Frame(active_intent="FindEvents", requested_slots=[],
                          slot_values={"category": "Music", "subcategory": "rock", "date": "March 7th"}),
    },
)
