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


def _generate(use_live: bool) -> int:
    from toduq.generate import generate_seed
    llm = judge = None
    if use_live:
        from toduq.judge import Judge
        from toduq.runners.factory import client_for_role
        gen = client_for_role("generator")
        llm = gen
        judge = Judge(client_for_role("judge"))
    stats = generate_seed(llm=llm, judge=judge)
    print("Seed set written to data/seed_v1/")
    print(json.dumps(stats.__dict__, indent=2))
    return 0


def _simulate() -> int:
    from toduq.ingest import RESTAURANT_DIALOGUE_USER_TURNS, SGD_1_00000_RAW, parse_dialogue
    from toduq.operators import all_operators
    from toduq.passes import run_dialogue
    from toduq.simulator import Chatbot, LexicalUncertaintyMetric, simulate_record

    d = parse_dialogue(SGD_1_00000_RAW)
    bot = Chatbot()                       # offline EchoClient
    metric = LexicalUncertaintyMetric()   # input-based, runs offline
    # one record per operator (first occurrence) with its real dialogue position
    records = run_dialogue(dialogue_id=d.dialogue_id, user_turns=RESTAURANT_DIALOGUE_USER_TURNS,
                           operators=all_operators(), turn_indices=d.user_turn_indices,
                           policy="all", seed=1)
    first: dict[str, object] = {}
    for r in records:
        first.setdefault(r.operator, r)

    print(f"TODUQ Simulator — chatbot replay, UQ metric = {metric.name}\n")
    hits = total = 0
    for op_id in ("slot_drop", "underspecify", "multi_value", "referential_ambig",
                  "out_of_kb_entity", "cross_turn_contra", "paraphrase"):
        rec = first.get(op_id)
        if rec is None:
            continue
        res = simulate_record(rec, RESTAURANT_DIALOGUE_USER_TURNS, bot, metric)
        total += 1
        hits += res.identified
        peak = ", ".join(f"t{ts.ordinal}={ts.score}" for ts in res.turn_scores if ts.score > 0) or "(no spike)"
        print(f"  {op_id:18} injected@t{res.injected_ordinal:<2} "
              f"peak@t{res.predicted_ordinal} rank={res.rank_of_injected} "
              f"identified={str(res.identified):5}  scores>0: {peak}")
    print(f"\nlocalization: {hits}/{total}. The lexical metric catches lexicalized "
          f"input uncertainty and the control; referential (coref) and parameter/"
          f"reasoning types need response-based UQ (semantic entropy / verbalized "
          f"confidence) with a live model — run with configs/models.yaml.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toduq")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run the pass-chain on a fixture and print the record")
    sub.add_parser("dialogue", help="spread injections across a fixture dialogue")
    sub.add_parser("simulate", help="replay a sample turn-by-turn; test if a UQ metric localizes the injection")
    gen = sub.add_parser("generate", help="build the curated seed set into data/seed_v1/")
    gen.add_argument("--live", action="store_true",
                     help="use live generator+judge from configs/models.yaml (needs keys)")
    sub.add_parser("schema", help="print the path to the annotation JSON Schema")

    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _demo()
    if args.cmd == "dialogue":
        return _dialogue()
    if args.cmd == "simulate":
        return _simulate()
    if args.cmd == "generate":
        return _generate(args.live)
    if args.cmd == "schema":
        from toduq.validate import _SCHEMA_PATH
        print(_SCHEMA_PATH)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
