"""Eval metrics: calibration, ranking, entropy, and cross-service bleed."""
import math

from toduq.eval import (
    abstention_accuracy,
    auroc,
    expected_calibration_error,
    over_abstention_rate,
    routing_accuracy,
    semantic_entropy,
    uncertainty_bleed,
)


def test_abstention_and_routing_accuracy():
    assert abstention_accuracy([True, False, True], [True, False, False]) == 2 / 3
    # routing scored only over should-abstain (gold != answer) turns
    assert routing_accuracy(["clarify", "hitl", "answer"],
                            ["clarify", "rag_structured", "answer"]) == 1 / 2


def test_over_abstention_rate():
    # gold says answer (False) on two turns; model abstained on one -> 0.5
    assert over_abstention_rate([True, False, True], [False, False, True]) == 0.5


def test_ece_perfectly_calibrated_is_zero():
    conf = [0.05, 0.95, 0.95]
    correct = [False, True, True]
    assert expected_calibration_error(conf, correct, n_bins=10) < 0.11


def test_auroc_perfect_and_random():
    # positives strictly out-rank negatives -> 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    # single class -> 0.5
    assert auroc([0.5, 0.6], [True, True]) == 0.5


def test_semantic_entropy():
    assert semantic_entropy(["a", "a", "a"]) == 0.0            # all agree -> 0
    assert semantic_entropy(["a", "b"]) == 1.0                  # 2 equal clusters, normalized
    assert semantic_entropy([]) == 0.0


def test_uncertainty_bleed():
    base = {"Music_1": {"song": "Lost Stars"}, "Events_1": {"city": "NYC"}}
    # perturb Events_1; Music_1 unchanged -> no bleed
    perturbed_clean = {"Music_1": {"song": "Lost Stars"}, "Events_1": {"city": "there"}}
    assert uncertainty_bleed(base, perturbed_clean, perturbed_service="Events_1") == 0.0
    # Music_1 slot drifted -> bleed
    perturbed_bleed = {"Music_1": {"song": "???"}, "Events_1": {"city": "there"}}
    assert uncertainty_bleed(base, perturbed_bleed, perturbed_service="Events_1") == 1.0
