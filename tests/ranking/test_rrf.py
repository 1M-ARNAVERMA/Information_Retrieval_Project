"""Unit tests for Phase 6: Reciprocal Rank Fusion (RRF)."""

import pytest

from src.common.types import RetrievalResult
from src.ranking.rrf import ReciprocalRankFusion, reciprocal_rank_fusion


def _make_res(doc_id: str, sentence_id: int, score: float, retriever: str) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        sentence_id=sentence_id,
        text=f"Text for {doc_id}:{sentence_id}",
        score=score,
        retriever=retriever,
    )


def test_rrf_duplicate_and_consensus():
    """Validates that items appearing at top ranks in multiple retrievers are boosted."""
    # DocA is rank 1 in both BM25 and TFIDF
    # DocB is rank 2 in BM25 only
    # DocC is rank 2 in TFIDF only
    bm25_results = [
        _make_res("DocA", 0, 10.0, "bm25"),
        _make_res("DocB", 0, 8.0, "bm25"),
    ]
    tfidf_results = [
        _make_res("DocA", 0, 0.9, "tfidf"),
        _make_res("DocC", 0, 0.7, "tfidf"),
    ]

    fused = reciprocal_rank_fusion(
        rankings={"bm25": bm25_results, "tfidf": tfidf_results},
        k=60,
    )

    assert len(fused) == 3
    # DocA appears in both, so score = 1/(60+1) + 1/(60+1) = 2/61 ~= 0.032787
    assert fused[0].doc_id == "DocA"
    assert fused[0].retriever == "rrf"
    assert "bm25" in fused[0].metadata["sources"]
    assert "tfidf" in fused[0].metadata["sources"]
    assert fused[0].metadata["rrf_ranks"]["bm25"] == 1
    assert fused[0].metadata["rrf_ranks"]["tfidf"] == 1


def test_rrf_different_rankings_and_weights():
    """Validates custom weights impact RRF outcome."""
    retriever_1 = [
        _make_res("DocX", 0, 1.0, "r1"),  # rank 1
        _make_res("DocY", 0, 0.5, "r1"),  # rank 2
    ]
    retriever_2 = [
        _make_res("DocY", 0, 1.0, "r2"),  # rank 1
        _make_res("DocX", 0, 0.5, "r2"),  # rank 2
    ]

    # Equal weights -> both DocX and DocY have identical score: 1/(60+1) + 1/(60+2)
    equal_fused = reciprocal_rank_fusion(
        rankings={"r1": retriever_1, "r2": retriever_2},
        k=60,
    )
    assert len(equal_fused) == 2
    assert abs(equal_fused[0].score - equal_fused[1].score) < 1e-5

    # Giving r1 high weight (2.0 vs 0.5) should push DocX above DocY
    weighted_fused = reciprocal_rank_fusion(
        rankings={"r1": retriever_1, "r2": retriever_2},
        weights={"r1": 2.0, "r2": 0.5},
        k=60,
    )
    assert weighted_fused[0].doc_id == "DocX"
    assert weighted_fused[1].doc_id == "DocY"


def test_rrf_missing_candidates():
    """Validates handling when retrievers return disjoint sets of candidates."""
    r1 = [_make_res("Doc1", 0, 1.0, "r1")]
    r2 = [_make_res("Doc2", 0, 1.0, "r2")]
    r3 = [_make_res("Doc3", 0, 1.0, "r3")]

    fused = reciprocal_rank_fusion(rankings={"r1": r1, "r2": r2, "r3": r3}, k=60)
    assert len(fused) == 3
    doc_ids = {r.doc_id for r in fused}
    assert doc_ids == {"Doc1", "Doc2", "Doc3"}


def test_rrf_empty_inputs():
    """Validates behavior when one or all candidate lists are empty."""
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"r1": [], "r2": []}) == []

    # One non-empty, one empty
    non_empty = [_make_res("Doc1", 0, 1.0, "r1")]
    fused = reciprocal_rank_fusion({"r1": non_empty, "r2": []})
    assert len(fused) == 1
    assert fused[0].doc_id == "Doc1"


def test_rrf_top_k():
    """Validates top_k truncation."""
    r = [_make_res(f"Doc{i}", 0, float(10 - i), "r") for i in range(10)]
    fused = reciprocal_rank_fusion({"r": r}, top_k=4)
    assert len(fused) == 4


def test_rrf_class_wrapper():
    """Validates ReciprocalRankFusion class with config defaults."""
    engine = ReciprocalRankFusion(k=60)
    r1 = [_make_res("DocA", 0, 1.0, "tfidf")]
    r2 = [_make_res("DocA", 0, 1.0, "bm25")]

    fused = engine.fuse({"tfidf": r1, "bm25": r2}, top_k=5)
    assert len(fused) == 1
    assert fused[0].doc_id == "DocA"
