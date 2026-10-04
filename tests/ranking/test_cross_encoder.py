"""Unit tests for Cross-Encoder neural reranker and MMR diversification."""

import pytest

from src.common.types import RerankedResult, RetrievalResult
from src.ranking.cross_encoder import BaseCrossEncoder, CrossEncoderReranker
from src.ranking.mmr import maximal_marginal_relevance


class MockCrossEncoder(BaseCrossEncoder):
    """Custom replaceable cross-encoder implementation for test verification."""

    def rerank(self, claim: str, candidates, top_k=None):
        out = []
        for c in candidates:
            # Deterministic score based on word overlap
            score = 0.99 if "relativity" in c.text.lower() else 0.10
            out.append(
                RerankedResult(
                    doc_id=c.doc_id,
                    sentence_id=c.sentence_id,
                    text=c.text,
                    retrieval_score=c.score,
                    rerank_score=score,
                    retriever_origin=c.retriever,
                )
            )
        out.sort(key=lambda x: x.rerank_score, reverse=True)
        return out[:top_k] if top_k else out


def test_cross_encoder_reranking_fallback():
    """Validates CrossEncoderReranker scoring and sorting with fallback model."""
    reranker = CrossEncoderReranker(model_name="fallback", default_top_k=2)

    cands = [
        RetrievalResult(
            doc_id="DocA",
            sentence_id=0,
            text="General science is an academic discipline.",
            score=0.90,  # High initial retrieval score
            retriever="bm25",
        ),
        RetrievalResult(
            doc_id="DocB",
            sentence_id=0,
            text="Albert Einstein developed the general theory of relativity in 1915.",
            score=0.75,  # Lower retrieval score
            retriever="bm25",
        ),
    ]

    reranked = reranker.rerank(
        claim="Albert Einstein developed relativity.",
        candidates=cands,
        top_k=2,
    )

    assert len(reranked) == 2
    assert all(isinstance(r, RerankedResult) for r in reranked)

    # Cross-encoder should recognize DocB is much more relevant than DocA and promote it to rank #1
    assert reranked[0].doc_id == "DocB"
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_cross_encoder_interface_replaceability():
    """Validates that any custom BaseCrossEncoder can drop into the architecture."""
    custom_encoder = MockCrossEncoder()
    cands = [
        RetrievalResult(doc_id="X", sentence_id=0, text="irrelevant", score=0.5, retriever="r"),
        RetrievalResult(doc_id="Y", sentence_id=0, text="theory of relativity", score=0.5, retriever="r"),
    ]
    results = custom_encoder.rerank("relativity", cands)
    assert results[0].doc_id == "Y"
    assert results[0].rerank_score == 0.99


def test_cross_encoder_empty():
    """Validates behavior on empty candidate list."""
    reranker = CrossEncoderReranker(model_name="fallback")
    assert reranker.rerank("Any claim", candidates=[]) == []


def test_mmr_diversification():
    """Validates MMR penalizes redundant candidates and promotes diverse evidence."""
    # Doc1 and Doc2 are almost identical paraphrases
    # Doc3 is a distinct, informative fact
    cand1 = RerankedResult(
        doc_id="DocA",
        sentence_id=0,
        text="Einstein developed relativity theory in Germany.",
        retrieval_score=0.9,
        rerank_score=0.95,
        retriever_origin="r",
    )
    cand2 = RerankedResult(
        doc_id="DocA",
        sentence_id=1,
        text="Einstein developed the theory of relativity in Germany.",
        retrieval_score=0.9,
        rerank_score=0.94,  # High relevance, but almost identical to cand1!
        retriever_origin="r",
    )
    cand3 = RerankedResult(
        doc_id="DocB",
        sentence_id=0,
        text="Albert Einstein received the Nobel Prize in Physics.",
        retrieval_score=0.8,
        rerank_score=0.88,  # Slightly lower relevance, but novel information!
        retriever_origin="r",
    )

    diverse = maximal_marginal_relevance(
        candidates=[cand1, cand2, cand3],
        top_k=2,
        lambda_param=0.6,
    )

    assert len(diverse) == 2
    # cand1 is chosen first (highest relevance)
    assert diverse[0].doc_id == "DocA" and diverse[0].sentence_id == 0

    # cand3 should be chosen over the redundant cand2
    assert diverse[1].doc_id == "DocB"
