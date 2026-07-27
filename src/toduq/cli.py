"""Minimal CLI. Real subcommands (ingest, generate, eval) land per milestone.

    toduq demo      # run the chain on a built-in fixture, print the JSON record
    toduq dialogue  # spread injections across a fixture dialogue's user turns
    toduq schema    # print the annotation JSON Schema path
"""
from __future__ import annotations

import argparse
import json
import sys


def _demo() -> int:
    from toduq.ingest import RESTAURANT_TURN_CITY
    from toduq.operators import get_operator
    from toduq.passes import run_chain

    record = run_chain(
        dialogue_id="1_00000",
        turn_idx=2,
        turn=RESTAURANT_TURN_CITY,
        operator=get_operator("slot_drop"),
        llm=None,          # offline: template-only, judge -> needs_review
        seed=42,
    )
    if record is None:
        print("Turn was not a viable injection site.", file=sys.stderr)
        return 1
    print(json.dumps(record.to_dict(), indent=2))
    return 0


def _dialogue() -> int:
    from toduq.ingest import SGD_1_00000_RAW, parse_dialogue
    from toduq.operators import all_operators
    from toduq.passes import run_dialogue

    d = parse_dialogue(SGD_1_00000_RAW)
    records = run_dialogue(
        dialogue_id=d.dialogue_id,
        user_turns=d.user_turns,
        operators=all_operators(),
        turn_indices=d.user_turn_indices,
        policy="all",
        seed=1,
    )
    print(f"{len(records)} samples from one dialogue, injected at different turns:")
    for r in sorted(records, key=lambda x: (x.position.user_turn_ordinal, x.operator)):
        p = r.position
        print(f"  t{r.turn_idx} [{p.band:6}] {r.operator:19} "
              f"action={r.gold.action:16} | {r.passes_edit.final_utterance}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toduq")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the pass-chain on a fixture and print the record")
    sub.add_parser("dialogue", help="spread injections across a fixture dialogue")
    sub.add_parser("schema", help="print the path to the annotation JSON Schema")

    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _demo()
    if args.cmd == "dialogue":
        return _dialogue()
    if args.cmd == "schema":
        from toduq.validate import _SCHEMA_PATH
        print(_SCHEMA_PATH)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
