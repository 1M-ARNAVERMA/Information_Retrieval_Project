"""Precision@K and Recall metrics for Information Retrieval evaluation."""

from typing import Iterable, List, Optional, Set, Union


def precision_at_k(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
    k: int = 10,
) -> float:
    """Calculates Precision@K.

    Formula:
        Precision@K = |Retrieved@K intersect Relevant| / K

    Args:
        retrieved: Ordered list of retrieved document/sentence identifiers.
        relevant: Set of ground-truth relevant document/sentence identifiers.
        k: Cutoff rank position.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k <= 0 or not retrieved:
        return 0.0

    rel_set = set(relevant)
    if not rel_set:
        return 0.0

    top_k_retrieved = retrieved[:k]
    hits = sum(1 for item in top_k_retrieved if item in rel_set)
    return float(hits / k)


def recall_at_k(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
    k: Optional[int] = None,
) -> float:
    """Calculates Recall@K.

    Formula:
        Recall@K = |Retrieved@K intersect Relevant| / |Relevant|

    Args:
        retrieved: Ordered list of retrieved identifiers.
        relevant: Set of ground-truth relevant identifiers.
        k: Optional cutoff position. If None, considers all retrieved items.

    Returns:
        Recall score in [0.0, 1.0].
    """
    rel_set = set(relevant)
    if not rel_set:
        return 0.0

    if not retrieved:
        return 0.0

    items_to_check = retrieved[:k] if k is not None and k > 0 else retrieved
    hits = sum(1 for item in items_to_check if item in rel_set)
    return float(hits / len(rel_set))
