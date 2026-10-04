"""Reciprocal Rank Fusion (RRF) for combining multi-retriever candidate rankings."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from src.common.types import RetrievalResult
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("rrf_fusion")


def reciprocal_rank_fusion(
    rankings: Dict[str, List[RetrievalResult]],
    weights: Optional[Dict[str, float]] = None,
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """Combines multiple ranked lists using Reciprocal Rank Fusion.

    Formula:
        RRF_score(d) = sum_{m in M} ( w_m / (k + rank_m(d)) )

    Args:
        rankings: Dictionary mapping retriever name -> list of RetrievalResult.
        weights: Optional dictionary mapping retriever name -> multiplier weight.
        k: Smoothing constant to control impact of top ranks (standard: 60).
        top_k: Optional maximum number of fused results to return.

    Returns:
        List of RetrievalResult instances sorted by RRF score descending.
    """
    if not rankings:
        return []

    scores: Dict[Tuple[str, int], float] = defaultdict(float)
    doc_info: Dict[Tuple[str, int], Dict[str, str]] = {}
    rank_history: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(dict)
    contributing_retrievers: Dict[Tuple[str, int], List[str]] = defaultdict(list)

    active_weights = weights or {}

    for retriever_name, result_list in rankings.items():
        if not result_list:
            continue

        weight = active_weights.get(retriever_name, 1.0)

        # Result list is assumed sorted by score descending (rank 1 = best)
        for rank_idx, item in enumerate(result_list, start=1):
            key = (item.doc_id, item.sentence_id)

            # Store document text if not already stored
            if key not in doc_info:
                doc_info[key] = {
                    "doc_id": item.doc_id,
                    "sentence_id": item.sentence_id,
                    "text": item.text,
                }

            # Accumulate weighted RRF score
            rrf_contribution = weight / (k + rank_idx)
            scores[key] += rrf_contribution
            rank_history[key][retriever_name] = rank_idx
            contributing_retrievers[key].append(retriever_name)

    if not scores:
        return []

    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        sorted_candidates = sorted_candidates[:top_k]

    fused_results: List[RetrievalResult] = []
    for (doc_id, sentence_id), score in sorted_candidates:
        info = doc_info[(doc_id, sentence_id)]
        fused_results.append(
            RetrievalResult(
                doc_id=info["doc_id"],
                sentence_id=info["sentence_id"],
                text=info["text"],
                score=round(score, 6),
                retriever="rrf",
                metadata={
                    "rrf_ranks": rank_history[(doc_id, sentence_id)],
                    "sources": contributing_retrievers[(doc_id, sentence_id)],
                },
            )
        )

    return fused_results


class ReciprocalRankFusion:
    """Configurable RRF engine reading defaults from system configuration."""

    def __init__(
        self,
        k: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        """Initializes RRF engine.

        Args:
            k: Smoothing constant (defaults to config.yaml 'retrieval.rrf_k' or 60).
            weights: Optional weight per retriever (defaults to config.yaml).
            default_top_k: Default candidate limit (defaults to config.yaml).
        """
        try:
            cfg = load_config()
            ret_cfg = cfg.get("retrieval", {})
            self.k = k if k is not None else ret_cfg.get("rrf_k", 60)
            self.weights = weights if weights is not None else ret_cfg.get("weights", {})
            self.default_top_k = (
                default_top_k
                if default_top_k is not None
                else ret_cfg.get("top_k_candidates", 50)
            )
        except Exception:
            self.k = k or 60
            self.weights = weights or {}
            self.default_top_k = default_top_k or 50

    def fuse(
        self,
        rankings: Dict[str, List[RetrievalResult]],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Fuses multiple ranked lists using configured parameters.

        Args:
            rankings: Dict of retriever_name -> List[RetrievalResult].
            top_k: Max candidates to return.

        Returns:
            List of fused RetrievalResult models.
        """
        limit = top_k or self.default_top_k
        return reciprocal_rank_fusion(
            rankings=rankings,
            weights=self.weights,
            k=self.k,
            top_k=limit,
        )
