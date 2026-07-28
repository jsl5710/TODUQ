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


def _generate(use_live: bool, workers: int = 0, balance: bool = True,
              balance_ratio: float = 1.0, control_multiplier: str = "auto") -> int:
    from toduq.generate import generate_seed
    llm = pool = judge = judge_pool = None
    if use_live:
        from toduq.judge import Judge
        from toduq.runners.factory import (
            build_generator_pool, build_judge_pool, client_for_role,
        )
        pool = build_generator_pool()          # multi-model even split, if configured
        if pool is not None:
            print(f"Generators: {len(pool.clients)}-model pool "
                  f"(split={pool.strategy}): {', '.join(pool.labels)}")
        else:
            llm = client_for_role("generator")  # single generator
        judge_pool = build_judge_pool()        # split the judge across models too
        if judge_pool is not None:
            print(f"Judges: {len(judge_pool.clients)}-model pool: {', '.join(judge_pool.labels)}")
        else:
            judge = Judge(client_for_role("judge"))
    cm: object = "auto" if control_multiplier == "auto" else int(control_multiplier)
    stats = generate_seed(llm=llm, pool=pool, workers=workers,
                          judge=judge, judge_pool=judge_pool, balance=balance,
                          balance_ratio=balance_ratio, control_multiplier=cm)
    print("Seed set written to data/seed_v1/ (records.jsonl = balanced, "
          "records_all.jsonl = full)")
    print(json.dumps(stats.__dict__, indent=2))
    print(f"Class balance: {stats.positive} positive (abstain/route) vs "
          f"{stats.negative} negative (answer) -> {stats.balanced_total} balanced")
    if pool is not None:
        print("Generator split:", json.dumps(pool.summary()))
    if judge_pool is not None:
        print("Judge split:", json.dumps(judge_pool.summary()))
    return 0


def _dry_run(control_multiplier: str = "auto") -> int:
    """Validate the model config and print the planned split — no generation."""
    from toduq.generate import plan_run
    from toduq.runners.factory import check_config, planned_split

    report = check_config()
    print(f"Config: {report['config']}")
    if not report.get("exists"):
        print("  (no models.yaml — a live run would fall back to the offline EchoClient)")
        return 0

    all_ok = True
    for role in ("generators", "judges"):
        checks = report[role]
        print(f"\n{role} ({len(checks)}):")
        if not checks:
            print("  (none configured)")
        for c in checks:
            mark = "OK " if c["ok"] else "XX "
            all_ok = all_ok and c["ok"]
            print(f"  [{mark}] {c['model_id']:<40} {c['kind']:<6} {c['detail']}")

    cm: object = "auto" if control_multiplier == "auto" else int(control_multiplier)
    plan = plan_run(control_multiplier=cm)
    units = plan["total_units"]
    gsplit = planned_split(units, "generators", "generation")
    jsplit = planned_split(units, "judges", "judging")
    print(f"\nPlanned run: {units} generation units "
          f"({plan['violation_units']} violation + {plan['control_units']} control, "
          f"control_multiplier={plan['control_multiplier']}) over "
          f"{plan['num_dialogues']} dialogue(s).")
    if gsplit is not None:
        print("  generator split:", json.dumps(gsplit))
    if jsplit is not None:
        print("  judge split:    ", json.dumps(jsplit))
    print(f"\n{'All endpoints/keys OK — ready to run.' if all_ok else 'Some checks FAILED — fix the marked entries before --live.'}")
    return 0 if all_ok else 1


def _simulate(metric_name: str = "lexical", mode: str = "history") -> int:
    from toduq.ingest import RESTAURANT_DIALOGUE_USER_TURNS, SGD_1_00000_RAW, parse_dialogue
    from toduq.operators import all_operators
    from toduq.passes import run_dialogue
    from toduq.simulator import Chatbot, load_metric, simulate_record

    d = parse_dialogue(SGD_1_00000_RAW)
    bot = Chatbot()                       # offline EchoClient
    metric = load_metric(metric_name)     # any method from the shared UQ registry
    # one record per operator (first occurrence) with its real dialogue position
    records = run_dialogue(dialogue_id=d.dialogue_id, user_turns=RESTAURANT_DIALOGUE_USER_TURNS,
                           operators=all_operators(), turn_indices=d.user_turn_indices,
                           policy="all", seed=1)
    first: dict[str, object] = {}
    for r in records:
        first.setdefault(r.operator, r)

    print(f"TODUQ Simulator — chatbot replay, UQ metric = {metric.name}, mode = {mode}\n")
    hits = total = 0
    for op_id in ("slot_drop", "underspecify", "multi_value", "referential_ambig",
                  "out_of_kb_entity", "cross_turn_contra", "paraphrase"):
        rec = first.get(op_id)
        if rec is None:
            continue
        res = simulate_record(rec, RESTAURANT_DIALOGUE_USER_TURNS, bot, metric, mode=mode)
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
    sim = sub.add_parser("simulate", help="replay a sample turn-by-turn; test if a UQ metric localizes the injection")
    sim.add_argument("--metric", default="lexical",
                     help="UQ method from the shared registry (lexical | semantic_entropy | self_consistency | verbalized_confidence)")
    sim.add_argument("--mode", default="history", choices=["history", "immediate"],
                     help="history = turn + prior context; immediate = current turn alone")
    gen = sub.add_parser("generate", help="build the curated seed set into data/seed_v1/")
    gen.add_argument("--live", action="store_true",
                     help="use live generator+judge from configs/models.yaml (needs keys)")
    gen.add_argument("--dry-run", action="store_true",
                     help="validate configured endpoints/keys and print the planned "
                          "split — no generation")
    gen.add_argument("--workers", type=int, default=0,
                     help="parallel generation across sites/models (0/1 = sequential)")
    gen.add_argument("--no-balance", dest="balance", action="store_false",
                     help="skip class balancing (keep the raw positive-skewed set)")
    gen.add_argument("--balance-ratio", type=float, default=1.0,
                     help="majority:minority cap after balancing (1.0 = exact 1:1)")
    gen.add_argument("--control-multiplier", default="auto",
                     help="control-paraphrase variants per turn to grow the negative "
                          "class ('auto' sizes it from the operator mix)")
    sub.add_parser("schema", help="print the path to the annotation JSON Schema")

    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _demo()
    if args.cmd == "dialogue":
        return _dialogue()
    if args.cmd == "simulate":
        return _simulate(args.metric, args.mode)
    if args.cmd == "generate":
        if args.dry_run:
            return _dry_run(args.control_multiplier)
        return _generate(args.live, args.workers, args.balance,
                         args.balance_ratio, args.control_multiplier)
    if args.cmd == "schema":
        from toduq.validate import _SCHEMA_PATH
        print(_SCHEMA_PATH)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
