"""Seed-set generation (Milestone 4).

Runs the chain-of-passes over selected dialogues with a generator LLM (Pass-3
paraphrase) and a judge (Pass-4 validation), then partitions the output:
  - accepted   -> seed records (data/seed_v1/records.jsonl)
  - needs_review / rejected -> human queue (data/seed_v1/review_queue.jsonl)
plus a manifest with counts and provenance.

Runs fully offline with EchoClient + HeuristicJudge; pass live clients (via the
model factory) to regenerate with LLM paraphrase + LLM judge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from toduq.ingest import SGD_1_00000_RAW, parse_dialogue
from toduq.judge import HeuristicJudge, Judge, NullJudge
from toduq.operators import all_operators
from toduq.passes import run_dialogue
from toduq.runners.base import LLMClient
from toduq.validate import check_invariants

_OUT = Path(__file__).resolve().parents[2] / "data" / "seed_v1"


@dataclass
class SeedStats:
    total: int
    accepted: int
    needs_review: int
    rejected: int
    invariant_failures: int
    by_action: dict[str, int]
    positive: int = 0          # should_abstain == True (abstain/route)
    negative: int = 0          # should_abstain == False (answer / control)
    balanced_total: int = 0    # size of the balanced records.jsonl


def _is_positive(payload: dict[str, Any]) -> bool:
    """Positive class = the model should abstain / route (not just answer)."""
    return bool(payload["gold"]["should_abstain"])


def generate_seed(
    raw_dialogues: Optional[list[dict[str, Any]]] = None,
    *,
    llm: Optional[LLMClient] = None,
    pool: Optional[Any] = None,
    workers: int = 0,
    judge: Optional[Judge | NullJudge | HeuristicJudge] = None,
    judge_pool: Optional[Any] = None,
    policy: str = "all",
    seed: int = 42,
    balance: bool = True,
    balance_ratio: float = 1.0,
    control_multiplier: Any = "auto",
    balance_seed: int = 0,
    out_dir: Path = _OUT,
) -> SeedStats:
    """Build the seed set. Pass `pool` (a ModelPool) to split generation evenly
    across several models — one shared pool across all dialogues keeps the split
    even over the whole run, and each record records its generating model.

    Class balance: `control_multiplier` (int or "auto") grows the negative
    (answer) class by emitting that many control-paraphrase variants per turn;
    "auto" sizes it from the operator mix. When `balance` is set, the *accepted*
    records are then trimmed to `balance_ratio` (1.0 = exact 1:1) and written to
    records.jsonl, while the full accepted set is kept in records_all.jsonl."""
    from toduq import balancing

    raw_dialogues = raw_dialogues or [SGD_1_00000_RAW]
    judge = judge or HeuristicJudge()
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = all_operators()
    mult = (balancing.auto_control_multiplier(ops, lambda o: o.family == "paraphrase")
            if control_multiplier == "auto" else int(control_multiplier))

    accepted, review = [], []
    by_action: dict[str, int] = {}
    by_generator: dict[str, int] = {}
    by_judge: dict[str, int] = {}
    inv_fail = 0

    for raw in raw_dialogues:
        d = parse_dialogue(raw)
        records = run_dialogue(
            dialogue_id=d.dialogue_id, user_turns=d.user_turns,
            operators=all_operators(), turn_indices=d.user_turn_indices,
            policy=policy, seed=seed, llm=llm, pool=pool, workers=workers,
            judge=judge, judge_pool=judge_pool, control_multiplier=mult,
        )
        for rec in records:
            payload = rec.to_dict()
            errs = check_invariants(payload)
            if errs:
                inv_fail += 1
                payload["_invariant_errors"] = errs
            by_action[rec.gold.action] = by_action.get(rec.gold.action, 0) + 1
            gen = rec.provenance.generator_model or "echo-stub"
            by_generator[gen] = by_generator.get(gen, 0) + 1
            jm = rec.provenance.judge_model or "null-judge"
            by_judge[jm] = by_judge.get(jm, 0) + 1
            (accepted if rec.passes_confirm.status == "accepted" and not errs
             else review).append(payload)

    # Class-balance the accepted set (positive = should_abstain).
    if balance:
        balanced, _dropped, report = balancing.balance(
            accepted, is_positive=_is_positive, ratio=balance_ratio, seed=balance_seed)
    else:
        balanced = accepted
        report = {"positive": sum(_is_positive(r) for r in accepted),
                  "negative": sum(not _is_positive(r) for r in accepted),
                  "target_ratio": None, "majority_class": None, "dropped": 0,
                  "balanced_total": len(accepted)}

    _write_jsonl(out_dir / "records.jsonl", balanced)          # balanced, shipped
    _write_jsonl(out_dir / "records_all.jsonl", accepted)      # full, unbalanced
    _write_jsonl(out_dir / "review_queue.jsonl", review)

    stats = SeedStats(
        total=len(accepted) + len(review),
        accepted=len(accepted),
        needs_review=sum(1 for r in review if r["passes"]["confirm"]["status"] == "needs_review"),
        rejected=sum(1 for r in review if r["passes"]["confirm"]["status"] == "rejected"),
        invariant_failures=inv_fail,
        by_action=dict(sorted(by_action.items())),
        positive=report["positive"], negative=report["negative"],
        balanced_total=report["balanced_total"],
    )
    manifest = {
        "sgd_version": "GEM/schema_guided_dialog",
        "num_dialogues": len(raw_dialogues),
        "policy": policy,
        "seed": seed,
        "generator_model": getattr(pool, "model_id", None) or getattr(llm, "model_id", "echo-stub"),
        "generators": pool.summary() if pool is not None else None,   # per-model split
        "by_generator": dict(sorted(by_generator.items())),           # from record provenance
        "judge_model": getattr(judge_pool, "model_id", None) or getattr(judge, "judge_model", "heuristic-judge"),
        "judges": judge_pool.summary() if judge_pool is not None else None,  # per-judge split
        "by_judge": dict(sorted(by_judge.items())),                   # from record provenance
        "control_multiplier": mult,
        "class_balance": report,                                      # positive vs negative
        "stats": stats.__dict__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
