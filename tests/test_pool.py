"""ModelPool: even split across models + per-record provenance."""
import pytest

from toduq.ingest import RESTAURANT_DIALOGUE_USER_TURNS, SGD_1_00000_RAW, parse_dialogue
from toduq.operators import all_operators
from toduq.passes import run_dialogue
from toduq.runners import ModelPool


class _Fake:
    def __init__(self, model_id):
        self.model_id = model_id

    def generate(self, prompt, *, system="", cfg=None):
        return f"[{self.model_id}] " + (prompt.strip().splitlines()[-1] if prompt.strip() else "")

    def sample(self, prompt, n, *, system="", cfg=None):
        return [self.generate(prompt) for _ in range(n)]


def _pool(strategy="round_robin", **kw):
    return ModelPool([_Fake("qwen"), _Fake("llama"), _Fake("mistral")], strategy=strategy, **kw)


def test_round_robin_cycles_in_order():
    p = _pool()
    got = [p.next().model_id for _ in range(7)]
    assert got == ["qwen", "llama", "mistral", "qwen", "llama", "mistral", "qwen"]
    assert p.summary() == {"qwen": 3, "llama": 2, "mistral": 2}


def test_round_robin_even_split_over_many():
    p = _pool()
    for _ in range(99):
        p.next()
    assert p.summary() == {"qwen": 33, "llama": 33, "mistral": 33}   # exactly even


def test_weighted_needs_matching_weights():
    with pytest.raises(ValueError):
        _pool("weighted", weights=[1, 1])           # 2 weights, 3 clients


def test_weighted_split_respects_weights():
    p = _pool("weighted", weights=[8, 1, 1], seed=0)
    for _ in range(400):
        p.next()
    # qwen (weight 8) should dominate
    assert p.summary()["qwen"] > p.summary()["llama"] + p.summary()["mistral"]


def test_run_dialogue_splits_and_records_generator():
    d = parse_dialogue(SGD_1_00000_RAW)
    pool = _pool()
    recs = run_dialogue(dialogue_id=d.dialogue_id, user_turns=RESTAURANT_DIALOGUE_USER_TURNS,
                        operators=all_operators(), turn_indices=d.user_turn_indices,
                        policy="all", seed=1, pool=pool)
    models = {r.provenance.generator_model for r in recs}
    assert models == {"qwen", "llama", "mistral"}          # all three used
    # even-ish split: counts differ by at most 1
    counts = sorted(pool.summary().values())
    assert counts[-1] - counts[0] <= 1


def test_parallel_matches_sequential():
    d = parse_dialogue(SGD_1_00000_RAW)
    kw = dict(dialogue_id=d.dialogue_id, user_turns=RESTAURANT_DIALOGUE_USER_TURNS,
              operators=all_operators(), turn_indices=d.user_turn_indices, policy="all", seed=1)
    seq = run_dialogue(pool=_pool(), workers=0, **kw)
    par = run_dialogue(pool=_pool(), workers=4, **kw)
    # same records in the same order, same per-site model assignment
    assert [r.record_id for r in seq] == [r.record_id for r in par]
    assert [r.provenance.generator_model for r in seq] == [r.provenance.generator_model for r in par]
