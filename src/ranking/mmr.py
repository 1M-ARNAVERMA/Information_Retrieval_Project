"""Maximal Marginal Relevance (MMR) for evidence diversification."""

from typing import List, Set
import numpy as np

from src.common.types import RerankedResult


def _token_jaccard(text_a: str, text_b: str) -> float:
    """Computes Jaccard lexical similarity between two sentences."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)
    return len(intersection) / len(union)


def maximal_marginal_relevance(
    candidates: List[RerankedResult],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[RerankedResult]:
    r"""Applies MMR diversification to reduce evidence redundancy.

    Formula:
        MMR = argmax_{d_i in R \ S} [ lambda * Sim(d_i, Q) - (1 - lambda) * max_{d_j in S} Sim(d_i, d_j) ]

    Args:
        candidates: List of RerankedResult items sorted by relevance.
        top_k: Number of diverse evidence items to return.
        lambda_param: Tradeoff between relevance (1.0) and diversity (0.0).

    Returns:
        Diversified list of RerankedResult models.
    """
    if not candidates or top_k <= 0:
        return []

    if len(candidates) <= top_k:
        return candidates

    selected: List[RerankedResult] = [candidates[0]]
    remaining: List[RerankedResult] = candidates[1:]

    while len(selected) < top_k and remaining:
        best_score = -float("inf")
        best_idx = 0

        for idx, cand in enumerate(remaining):
            relevance = cand.rerank_score
            # Max similarity to any already-selected evidence sentence
            max_sim_to_selected = max(
                _token_jaccard(cand.text, sel.text) for sel in selected
            )

            mmr_score = (lambda_param * relevance) - ((1.0 - lambda_param) * max_sim_to_selected)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    return selected
