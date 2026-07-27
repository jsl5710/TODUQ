"""Shared UQ registry: load any method of choice; each method's behavior."""
import pytest

from toduq.uq import available, load_uq


class _Vary:
    """Fake client: sample() returns the given responses (cycled)."""
    model_id = "vary"

    def __init__(self, responses):
        self.responses = responses

    def generate(self, prompt):
        return self.responses[0]

    def sample(self, prompt, n):
        return (self.responses * n)[:n]


def test_registry_lists_all_methods():
    assert set(available()) == {"lexical", "semantic_entropy",
                                "self_consistency", "verbalized_confidence"}


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        load_uq("nope")


def test_lexical_input_and_parameter():
    r = load_uq("lexical").score("find somewhere to eat")
    assert r.method == "lexical" and r.uncertainty_type == "input" and r.score > 0
    assert load_uq("lexical").score("book a table for two at seven").score == 0.0
    p = load_uq("lexical").score("will it be busy next friday")
    assert p.uncertainty_type == "parameter" and p.score >= 0.5


def test_semantic_entropy_discriminates():
    se = load_uq("semantic_entropy", n=3)
    agree = se.score("q", client=_Vary(["same"])).score          # 1 cluster -> 0
    disagree = se.score("q", client=_Vary(["a", "b", "c"])).score  # 3 clusters -> high
    assert agree == 0.0 and disagree > agree
    assert se.score("q").score == 0.0                            # no client -> neutral


def test_self_consistency():
    sc = load_uq("self_consistency", n=4)
    assert sc.score("q", client=_Vary(["x"])).score == 0.0                 # full agreement
    assert sc.score("q", client=_Vary(["x", "y", "z", "w"])).score > 0.0   # all disagree


def test_verbalized_confidence():
    class _Conf:
        def generate(self, p): return "0.2"
        def sample(self, p, n): return []
    r = load_uq("verbalized_confidence").score("q", client=_Conf())
    assert abs(r.score - 0.8) < 1e-6
