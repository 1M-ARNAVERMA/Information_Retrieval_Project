"""Unit tests for Information Retrieval metrics: Precision, Recall, MAP, and nDCG."""

import pytest

from src.evaluation.map import average_precision_at_k, mean_average_precision
from src.evaluation.ndcg import dcg_at_k, mean_ndcg_at_k, ndcg_at_k
from src.evaluation.precision_recall import precision_at_k, recall_at_k


def test_perfect_ranking():
    """Validates metrics for a perfect ranking where all relevant items are retrieved first."""
    relevant = {"doc1", "doc2"}
    retrieved = ["doc1", "doc2", "doc3", "doc4"]

    # Precision@2 = 2/2 = 1.0
    assert precision_at_k(retrieved, relevant, k=2) == 1.0
    # Recall@2 = 2/2 = 1.0
    assert recall_at_k(retrieved, relevant, k=2) == 1.0
    # AP@2 = (1/1 + 2/2) / 2 = 1.0
    assert average_precision_at_k(retrieved, relevant, k=2) == 1.0
    # nDCG@2 = 1.0
    assert ndcg_at_k(retrieved, relevant, k=2) == 1.0


def test_empty_results_and_zero_relevance():
    """Validates metrics when no results are retrieved or no relevant documents exist."""
    relevant = {"doc1"}

    assert precision_at_k([], relevant, k=5) == 0.0
    assert recall_at_k([], relevant, k=5) == 0.0
    assert average_precision_at_k([], relevant, k=5) == 0.0
    assert ndcg_at_k([], relevant, k=5) == 0.0

    # No ground truth relevant
    assert precision_at_k(["doc1"], set(), k=5) == 0.0
    assert recall_at_k(["doc1"], set(), k=5) == 0.0
    assert average_precision_at_k(["doc1"], set(), k=5) == 0.0
    assert ndcg_at_k(["doc1"], set(), k=5) == 0.0


def test_partially_correct_ranking():
    """Validates metrics when relevant items appear with false positives."""
    relevant = {"doc1", "doc3"}
    # Rank 1: doc1 (hit)
    # Rank 2: doc2 (miss)
    # Rank 3: doc3 (hit)
    retrieved = ["doc1", "doc2", "doc3", "doc4"]

    p_at_3 = precision_at_k(retrieved, relevant, k=3)
    assert p_at_3 == round(2 / 3, 4) or abs(p_at_3 - 2 / 3) < 1e-4

    r_at_3 = recall_at_k(retrieved, relevant, k=3)
    assert r_at_3 == 1.0

    # AP@3:
    # rank 1 hit: P@1 = 1/1 = 1.0
    # rank 2 miss
    # rank 3 hit: P@3 = 2/3 = 0.6667
    # sum = 1.6667 / 2 = 0.8333
    ap = average_precision_at_k(retrieved, relevant, k=3)
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-4

    # nDCG@3: DCG@3 < IDCG@3 because doc3 is at rank 3 instead of rank 2
    ndcg = ndcg_at_k(retrieved, relevant, k=3)
    assert 0.0 < ndcg < 1.0


def test_mean_metrics():
    """Validates multi-query aggregation for MAP and mean nDCG."""
    all_ret = {
        "q1": ["doc1", "doc2"],
        "q2": ["doc3", "doc4"],
    }
    all_rel = {
        "q1": {"doc1"},
        "q2": {"doc4"},
    }

    map_val = mean_average_precision(all_ret, all_rel, k=2)
    assert 0.0 < map_val <= 1.0

    ndcg_val = mean_ndcg_at_k(all_ret, all_rel, k=2)
    assert 0.0 < ndcg_val <= 1.0
