"""Class balancing: control multiplier grows negatives, balance() trims to 1:1."""
from toduq import balancing
from toduq.generate import generate_seed
from toduq.ingest import SGD_1_00000_RAW, parse_dialogue
from toduq.operators import all_operators
from toduq.passes import run_dialogue


def _is_pos(r):
    return r["cls"]


def test_balance_exact_one_to_one():
    recs = [{"cls": True}] * 20 + [{"cls": False}] * 5     # 20 pos, 5 neg
    kept, dropped, report = balancing.balance(recs, is_positive=_is_pos, ratio=1.0, seed=0)
    assert report["positive"] == 20 and report["negative"] == 5
    assert report["majority_class"] == "positive"
    assert report["kept_positive"] == 5 and report["kept_negative"] == 5   # 1:1
    assert len(kept) == 10 and len(dropped) == 15


def test_balance_ratio_two_to_one():
    recs = [{"cls": True}] * 20 + [{"cls": False}] * 5
    kept, _dropped, report = balancing.balance(recs, is_positive=_is_pos, ratio=2.0, seed=0)
    assert report["kept_positive"] == 10 and report["kept_negative"] == 5  # 2:1
    assert len(kept) == 15


def test_balance_noop_when_a_class_is_empty():
    recs = [{"cls": True}] * 7
    kept, dropped, report = balancing.balance(recs, is_positive=_is_pos)
    assert len(kept) == 7 and dropped == [] and report["majority_class"] == "none"


def test_balance_is_deterministic():
    recs = [{"cls": True, "i": i} for i in range(30)] + [{"cls": False, "i": i} for i in range(6)]
    a, _, _ = balancing.balance(recs, is_positive=_is_pos, seed=3)
    b, _, _ = balancing.balance(recs, is_positive=_is_pos, seed=3)
    assert [r["i"] for r in a if r["cls"]] == [r["i"] for r in b if r["cls"]]


def test_auto_control_multiplier_from_operator_mix():
    ops = all_operators()
    m = balancing.auto_control_multiplier(ops, lambda o: o.family == "paraphrase")
    n_ctrl = sum(1 for o in ops if o.family == "paraphrase")
    n_other = len(ops) - n_ctrl
    assert m == max(1, round(n_other / max(1, n_ctrl)))
    assert m > 1                                            # one control vs many violations


def test_control_multiplier_grows_negative_class():
    d = parse_dialogue(SGD_1_00000_RAW)
    kw = dict(dialogue_id=d.dialogue_id, user_turns=d.user_turns,
              operators=all_operators(), turn_indices=d.user_turn_indices,
              policy="all", seed=1)
    base = run_dialogue(control_multiplier=1, **kw)
    grown = run_dialogue(control_multiplier=5, **kw)
    neg_base = sum(1 for r in base if not r.gold.should_abstain)
    neg_grown = sum(1 for r in grown if not r.gold.should_abstain)
    pos_base = sum(1 for r in base if r.gold.should_abstain)
    pos_grown = sum(1 for r in grown if r.gold.should_abstain)
    assert neg_grown == neg_base * 5        # 5 variants per control site
    assert pos_grown == pos_base            # positives untouched
    # variants are distinct records (distinct seeds -> distinct record_ids)
    assert len({r.record_id for r in grown}) == len(grown)


def test_generate_seed_writes_balanced_and_full(tmp_path):
    stats = generate_seed(out_dir=tmp_path, balance=True, balance_ratio=1.0)
    balanced = (tmp_path / "records.jsonl").read_text().strip().splitlines()
    full = (tmp_path / "records_all.jsonl").read_text().strip().splitlines()
    assert len(balanced) == stats.balanced_total
    assert len(full) == stats.accepted
    # the shipped set is balanced: positive count == negative count (1:1)
    import json
    pos = sum(1 for ln in balanced if json.loads(ln)["gold"]["should_abstain"])
    neg = len(balanced) - pos
    assert pos == neg
