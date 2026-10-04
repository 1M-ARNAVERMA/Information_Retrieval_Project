"""Unit tests for Phase 9: ColBERT Late-Interaction Token-Level Retrieval."""

from pathlib import Path
import numpy as np
import pytest

from src.common.types import Document, RetrievalResult
from src.retrieval.colbert_retriever import ColbertRetriever, compute_maxsim


def test_compute_maxsim_exact_match():
    """Validates MaxSim operator with known vector matrices."""
    # 2 query tokens, 3 doc tokens, dim = 2
    # Q: [[1, 0], [0, 1]]
    # D: [[1, 0], [0, 1], [0.7071, 0.7071]]
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)

    # For Q[0]=[1,0], max with D is 1.0 (with D[0])
    # For Q[1]=[0,1], max with D is 1.0 (with D[1])
    # Total sum = 2.0
    score = compute_maxsim(q, d)
    assert abs(score - 2.0) < 1e-4

    # Empty inputs
    assert compute_maxsim(np.empty((0, 2)), d) == 0.0
    assert compute_maxsim(q, np.empty((0, 2))) == 0.0


def test_colbert_retriever_workflow(tmp_path):
    """Validates ColBERT indexing, MaxSim retrieval, and disk persistence."""
    retriever = ColbertRetriever(checkpoint="fallback", embedding_dim=16)

    docs = [
        Document(doc_id="Albert Einstein", sentence_id=0, text="Albert Einstein developed relativity."),
        Document(doc_id="Marie Curie", sentence_id=0, text="Marie Curie won the Nobel Prize."),
        Document(doc_id="SpaceX", sentence_id=0, text="SpaceX launches reusable rockets."),
    ]

    # Test indexing
    retriever.index(docs)
    assert len(retriever.doc_embeddings) == 3

    # Test retrieval
    results = retriever.retrieve("Albert Einstein theory of relativity", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.retriever == "colbert" for r in results)
    assert results[0].doc_id == "Albert Einstein"
    assert results[0].score > 0.0

    # Test empty query
    assert retriever.retrieve("", top_k=5) == []

    # Test save and reload
    retriever.save(tmp_path)
    loaded_retriever = ColbertRetriever(embedding_dim=16)
    loaded_retriever.load(tmp_path)

    assert len(loaded_retriever.doc_embeddings) == 3
    loaded_results = loaded_retriever.retrieve("Albert Einstein", top_k=2)
    assert loaded_results[0].doc_id == "Albert Einstein"


def test_colbert_availability():
    """Validates that availability check executes without crashing."""
    avail = ColbertRetriever.is_available()
    assert isinstance(avail, bool)


def test_unindexed_retriever():
    """Validates querying unindexed retriever returns empty list safely."""
    empty_retriever = ColbertRetriever()
    assert empty_retriever.retrieve("query") == []
