"""Minimal CLI. Real subcommands (ingest, generate, eval) land per milestone.

    toduq demo     # run the chain on a built-in fixture, print the JSON record
    toduq schema   # print the annotation JSON Schema path
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toduq")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the pass-chain on a fixture and print the record")
    sub.add_parser("schema", help="print the path to the annotation JSON Schema")

    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _demo()
    if args.cmd == "schema":
        from toduq.validate import _SCHEMA_PATH
        print(_SCHEMA_PATH)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
