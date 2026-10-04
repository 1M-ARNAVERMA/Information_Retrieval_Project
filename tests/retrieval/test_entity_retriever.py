"""Unit tests for Entity-Aware Retrieval and Boosting."""

import pytest

from src.common.types import Entity, RetrievalResult
from src.retrieval.entity_retriever import EntityAwareRetriever


def test_entity_overlap_and_boosting():
    """Validates entity matching and score multiplier logic."""
    booster = EntityAwareRetriever(boost_factor=1.0, title_boost_bonus=0.25)

    entities = [
        Entity(text="Albert Einstein", label="PERSON"),
        Entity(text="Germany", label="GPE"),
    ]

    # Candidate A contains both entities (and doc_id matches)
    cand_a = RetrievalResult(
        doc_id="Albert Einstein",
        sentence_id=1,
        text="Albert Einstein was born in the German Empire in Germany.",
        score=0.80,
        retriever="bm25",
    )

    # Candidate B contains no entities
    cand_b = RetrievalResult(
        doc_id="General physics",
        sentence_id=0,
        text="Theoretical physics studies natural phenomena.",
        score=0.85,  # Higher initial score than cand_a!
        retriever="bm25",
    )

    # Candidate C contains one entity ("Einstein")
    cand_c = RetrievalResult(
        doc_id="Quantum mechanics",
        sentence_id=3,
        text="Einstein also contributed to photon quantum theory.",
        score=0.75,
        retriever="bm25",
    )

    results = booster.rerank(
        claim="Albert Einstein was born in Germany.",
        candidates=[cand_b, cand_a, cand_c],
        entities=entities,
    )

    # Candidate A should be boosted to the top because of 100% entity overlap + title bonus!
    assert results[0].doc_id == "Albert Einstein"
    assert results[0].score > cand_a.score
    assert len(results[0].metadata["entity_matches"]) >= 1

    # Candidate B should NOT be removed; it remains in the list to preserve recall!
    doc_ids = [r.doc_id for r in results]
    assert "General physics" in doc_ids
    assert len(results) == 3


def test_empty_candidates_or_entities():
    """Validates behavior when no candidates or entities exist."""
    booster = EntityAwareRetriever()
    assert booster.rerank("Any claim", candidates=[]) == []

    cand = RetrievalResult(
        doc_id="Sample",
        sentence_id=0,
        text="Text",
        score=0.5,
        retriever="tfidf",
    )
    # Claim with no entities
    unchanged = booster.rerank("Just words here.", candidates=[cand], entities=[])
    assert len(unchanged) == 1
    assert unchanged[0].score == 0.5
