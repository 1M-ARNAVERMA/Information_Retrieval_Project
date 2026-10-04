"""Average Precision (AP) and Mean Average Precision (MAP) evaluation metrics."""

from typing import Dict, List, Optional, Set, Union


def average_precision_at_k(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
    k: int = 10,
) -> float:
    """Calculates Average Precision at rank K for a single query.

    Formula:
        AP@K = sum_{i=1}^K [ P@i * I(d_i in Relevant) ] / min(K, |Relevant|)

    Args:
        retrieved: Ranked list of retrieved identifiers.
        relevant: Set of relevant identifiers.
        k: Rank cutoff.

    Returns:
        AP score in [0.0, 1.0].
    """
    rel_set = set(relevant)
    if not rel_set or not retrieved or k <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in rel_set:
            hits += 1
            precision_sum += hits / i

    if hits == 0:
        return 0.0

    normalizer = min(k, len(rel_set))
    return float(precision_sum / normalizer)


def mean_average_precision(
    all_retrieved: Dict[str, List[str]],
    all_relevant: Dict[str, Union[Set[str], List[str]]],
    k: int = 10,
) -> float:
    """Calculates Mean Average Precision (MAP@K) across an evaluation query set.

    Formula:
        MAP@K = (1 / |Q|) * sum_{q in Q} AP@K(q)

    Args:
        all_retrieved: Mapping of query_id -> ranked retrieved document IDs.
        all_relevant: Mapping of query_id -> relevant ground truth document IDs.
        k: Rank cutoff.

    Returns:
        MAP score in [0.0, 1.0].
    """
    if not all_relevant:
        return 0.0

    ap_scores: List[float] = []

    for qid, rel in all_relevant.items():
        ret = all_retrieved.get(qid, [])
        ap = average_precision_at_k(ret, rel, k=k)
        ap_scores.append(ap)

    return float(sum(ap_scores) / len(ap_scores)) if ap_scores else 0.0
