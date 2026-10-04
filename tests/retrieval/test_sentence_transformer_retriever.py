"""Unit tests for Phase 5: Sentence Transformer and FAISS dense retrieval."""

from pathlib import Path
import numpy as np
import pytest

from src.common.types import Document, RetrievalResult
from src.retrieval.faiss_index import FaissVectorIndex
from src.retrieval.sentence_transformer_retriever import SentenceTransformerRetriever
from src.utils.model_loader import clear_model_cache, get_sentence_transformer


def test_faiss_vector_index_basic(tmp_path):
    """Validates vector indexing, cosine similarity search, and persistence."""
    dim = 4
    idx = FaissVectorIndex(dimension=dim, metric="ip")

    # Create 3 normalized vectors
    # v0: [1, 0, 0, 0]
    # v1: [0, 1, 0, 0]
    # v2: [0.7071, 0.7071, 0, 0]
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7071, 0.7071, 0.0, 0.0],
    ], dtype=np.float32)

    metadata = [
        {"doc_id": "DocA", "sentence_id": 0, "text": "Text A"},
        {"doc_id": "DocB", "sentence_id": 0, "text": "Text B"},
        {"doc_id": "DocC", "sentence_id": 0, "text": "Text C"},
    ]

    idx.add(vectors, metadata)
    assert idx.ntotal == 3

    # Query with vector close to DocA
    query_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    matches = idx.search(query_vec, top_k=2)

    assert len(matches) == 2
    # Top match should be DocA
    assert matches[0][0]["doc_id"] == "DocA"
    assert matches[0][1] > matches[1][1]

    # Save and reload
    idx.save(tmp_path)
    loaded_idx = FaissVectorIndex(dimension=dim, metric="ip")
    loaded_idx.load(tmp_path)
    assert loaded_idx.ntotal == 3

    reloaded_matches = loaded_idx.search(query_vec, top_k=2)
    assert reloaded_matches[0][0]["doc_id"] == "DocA"
    assert abs(reloaded_matches[0][1] - matches[0][1]) < 1e-4


def test_faiss_empty_search():
    """Validates behavior when searching an empty index."""
    idx = FaissVectorIndex(dimension=4)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert idx.search(q) == []


class DummySentenceTransformer:
    """Fast deterministic mock for testing SentenceTransformerRetriever without heavy GPU/network."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        # Generate deterministic vectors based on string hash / length
        embs = []
        for t in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            # Encode character frequencies or length
            vec[0] = len(t) % 10 + 1.0
            if "einstein" in t.lower() or "relativity" in t.lower():
                vec[1] = 10.0
            if "curie" in t.lower():
                vec[2] = 10.0
            if "spacex" in t.lower():
                vec[3] = 10.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embs.append(vec)
        res = np.array(embs, dtype=np.float32)
        return res[0] if len(texts) == 1 and not isinstance(texts, list) else res


def test_sentence_transformer_retriever_mock(tmp_path):
    """Validates SentenceTransformerRetriever workflow with FAISS."""
    retriever = SentenceTransformerRetriever()
    # Inject dummy model for fast, reproducible unit test
    retriever._model = DummySentenceTransformer(dimension=16)

    docs = [
        Document(doc_id="Albert Einstein", sentence_id=0, text="Albert Einstein developed relativity."),
        Document(doc_id="Marie Curie", sentence_id=0, text="Marie Curie won the Nobel Prize."),
        Document(doc_id="SpaceX", sentence_id=0, text="SpaceX develops rockets."),
    ]

    retriever.index(docs)
    assert retriever.index_wrapper.ntotal == 3

    # Query for Einstein
    results = retriever.retrieve("Albert Einstein and general relativity", top_k=2)
    assert len(results) == 2
    assert results[0].doc_id == "Albert Einstein"
    assert results[0].retriever == "sentence_transformer"
    assert isinstance(results[0].score, float)

    # Empty query test
    assert retriever.retrieve("", top_k=5) == []

    # Save and reload
    retriever.save(tmp_path)
    reloaded_retriever = SentenceTransformerRetriever()
    reloaded_retriever._model = DummySentenceTransformer(dimension=16)
    reloaded_retriever.load(tmp_path)

    reloaded_results = reloaded_retriever.retrieve("Albert Einstein relativity", top_k=2)
    assert reloaded_results[0].doc_id == "Albert Einstein"
