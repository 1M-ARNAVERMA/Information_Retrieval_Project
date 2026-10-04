"""Normalized Discounted Cumulative Gain (nDCG@K) evaluation metrics."""

import math
from typing import Dict, List, Set, Union


def dcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10,
) -> float:
    """Calculates Discounted Cumulative Gain (DCG@K) with binary relevance."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            # log2(i + 1) discount
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
    k: int = 10,
) -> float:
    """Calculates Normalized Discounted Cumulative Gain (nDCG@K).

    Formula:
        nDCG@K = DCG@K / IDCG@K

    Args:
        retrieved: Ranked list of retrieved document identifiers.
        relevant: Set of relevant document identifiers.
        k: Rank cutoff.

    Returns:
        nDCG score in [0.0, 1.0].
    """
    rel_set = set(relevant)
    if not rel_set or not retrieved or k <= 0:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, rel_set, k=k)

    # Ideal DCG: all top min(k, len(rel_set)) positions are relevant
    ideal_hits = min(k, len(rel_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0

    return float(actual_dcg / idcg)


def mean_ndcg_at_k(
    all_retrieved: Dict[str, List[str]],
    all_relevant: Dict[str, Union[Set[str], List[str]]],
    k: int = 10,
) -> float:
    """Calculates Mean nDCG@K across an evaluation query set."""
    if not all_relevant:
        return 0.0

    scores = []
    for qid, rel in all_relevant.items():
        ret = all_retrieved.get(qid, [])
        score = ndcg_at_k(ret, rel, k=k)
        scores.append(score)

    return float(sum(scores) / len(scores)) if scores else 0.0
