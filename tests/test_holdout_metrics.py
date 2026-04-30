import math

from music_discovery.evaluate.holdout_eval import (
    _hit_rate_at_k,
    _ndcg_at_k,
    _recall_at_k,
)


def test_recall_adjusted_matches_raw_with_full_coverage():
    recommended = ["s1", "s2", "s3"]
    relevant = {"s1", "s4"}

    raw, adj = _recall_at_k(recommended, relevant, k=3, covered_relevant=relevant)

    assert raw == 0.5
    assert adj == raw


def test_recall_adjusted_uses_only_covered_relevant_denominator():
    recommended = ["s3", "s1", "s2"]
    relevant = {"s1", "s2", "s4", "s5"}
    covered_relevant = {"s1", "s2"}

    raw, adj = _recall_at_k(recommended, relevant, k=3, covered_relevant=covered_relevant)

    assert raw == 0.5
    assert adj == 1.0
    assert adj >= raw


def test_recall_returns_zeros_for_empty_relevant():
    assert _recall_at_k(["s1"], set(), k=5) == (0.0, 0.0)


def test_recall_adjusted_handles_empty_covered_set():
    raw, adj = _recall_at_k(["s1"], {"s1", "s2"}, k=1, covered_relevant=set())

    assert raw == 0.5
    assert adj == 0.0


def test_ndcg_adjusted_matches_raw_with_full_coverage():
    recommended = ["s1", "s2", "s3"]
    relevant = {"s1", "s3"}

    raw, adj = _ndcg_at_k(recommended, relevant, k=3, covered_relevant=relevant)

    assert math.isclose(raw, adj)


def test_ndcg_adjusted_uses_covered_relevant_for_idcg():
    recommended = ["s4", "s2", "s1"]
    relevant = {"s1", "s2", "s5", "s6"}
    covered_relevant = {"s1", "s2"}

    raw, adj = _ndcg_at_k(recommended, relevant, k=3, covered_relevant=covered_relevant)

    assert math.isclose(raw, 0.5307212739772434)
    assert math.isclose(adj, 0.6934264036172708)
    assert adj > raw


def test_ndcg_returns_zeros_for_empty_relevant():
    assert _ndcg_at_k(["s1"], set(), k=5) == (0.0, 0.0)


def test_ndcg_adjusted_handles_empty_covered_set():
    raw, adj = _ndcg_at_k(["s1"], {"s1"}, k=1, covered_relevant=set())

    assert raw == 1.0
    assert adj == 0.0


def test_hit_rate_at_k_sanity():
    assert _hit_rate_at_k(["s1", "s2"], {"s2"}, k=2) == 1.0
    assert _hit_rate_at_k(["s1", "s2"], {"s3"}, k=2) == 0.0
