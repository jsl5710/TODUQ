"""Validate records against data/schema/annotation.schema.json + design invariants.

`jsonschema` covers structure; `check_invariants` covers the cross-field rules
from docs/annotation_schema.md that JSON Schema can't easily express.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schema" / "annotation.schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_schema(record: dict[str, Any]) -> None:
    """Raise if the record violates the JSON Schema. Requires `jsonschema`."""
    import jsonschema  # optional dep

    jsonschema.validate(record, load_schema())


def check_invariants(record: dict[str, Any]) -> list[str]:
    """Return a list of invariant violations (empty == valid). Pure stdlib."""
    errs: list[str] = []
    gold = record["gold"]
    doc = record["passes"]["document"]
    confirm = record["passes"]["confirm"]

    # 1. should_abstain == (action != answer)
    if gold["should_abstain"] != (gold["action"] != "answer"):
        errs.append("should_abstain must equal (action != 'answer')")

    # 2. paraphrase controls -> answer + empty slot_delta
    if record["family"] == "paraphrase":
        if gold["action"] != "answer":
            errs.append("paraphrase family must have gold.action == 'answer'")
        if doc["slot_delta"]:
            errs.append("paraphrase family must not edit slots")

    # 3. parameter uncertainty -> rag_*/hitl
    if record["uncertainty_type"] == "parameter":
        if gold["action"] not in ("rag_structured", "rag_unstructured", "hitl"):
            errs.append("parameter uncertainty must route to rag_*/hitl")

    # 4. rag_structured -> gold_query present
    if gold["action"] == "rag_structured" and not doc.get("gold_query"):
        errs.append("rag_structured requires a gold_query payload")

    # 5. accepted -> change_applied
    if confirm["status"] == "accepted" and not confirm["change_applied"]:
        errs.append("accepted status requires change_applied == true")

    return errs
