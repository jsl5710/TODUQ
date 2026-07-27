"""The worked example in examples/ must satisfy the design invariants and (if
jsonschema is installed) the JSON Schema."""
import json
from pathlib import Path

import pytest

from toduq.validate import check_invariants

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "record_slot_drop.json"


def _load():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_example_passes_invariants():
    assert check_invariants(_load()) == []


def test_example_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from toduq.validate import load_schema

    jsonschema.validate(_load(), load_schema())
