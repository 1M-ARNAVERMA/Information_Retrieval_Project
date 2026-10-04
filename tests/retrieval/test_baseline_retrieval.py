"""Unit tests for Phase 4: Baseline Retrieval (TF-IDF and BM25)."""

from pathlib import Path
import pytest

from src.common.exceptions import IndexNotFoundError
from src.common.types import Document, RetrievalResult
from src.data_ingestion.wiki_loader import load_documents
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever

SAMPLE_WIKI = Path("data/sample/wiki_sample.jsonl")


@pytest.fixture
def sample_documents():
    """Loads sample Wikipedia documents for retrieval testing."""
    return load_documents(SAMPLE_WIKI)


def test_tfidf_indexing_and_retrieval(sample_documents):
    """Validates TF-IDF index construction and scoring."""
    retriever = TfidfRetriever()
    retriever.index(sample_documents)

    # 1. Standard query retrieval
    query = "Albert Einstein developed the theory of relativity"
    results = retriever.retrieve(query, top_k=3)

    assert len(results) > 0
    assert len(results) <= 3
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.retriever == "tfidf" for r in results)

    # Top result should be from Albert Einstein article
    top_result = results[0]
    assert top_result.doc_id == "Albert Einstein"
    assert "relativity" in top_result.text.lower()
    assert top_result.score > 0.0

    # 2. Empty query handling
    assert retriever.retrieve("", top_k=5) == []
    assert retriever.retrieve("   ", top_k=5) == []

    # 3. Completely unrelated query
    unrelated = retriever.retrieve("xylophone quantum avocado saxophone", top_k=5)
    # Either empty or very low score
    assert len(unrelated) == 0 or all(r.score >= 0.0 for r in unrelated)


def test_tfidf_save_and_load(sample_documents, tmp_path):
    """Validates TF-IDF persistence to and from disk."""
    retriever = TfidfRetriever()
    retriever.index(sample_documents)

    query = "Marie Curie won Nobel Prize in Physics"
    original_results = retriever.retrieve(query, top_k=3)

    # Save to tmp_path
    retriever.save(tmp_path)

    # Reload into fresh instance
    new_retriever = TfidfRetriever()
    new_retriever.load(tmp_path)
    loaded_results = new_retriever.retrieve(query, top_k=3)

    assert len(loaded_results) == len(original_results)
    assert loaded_results[0].doc_id == original_results[0].doc_id
    assert loaded_results[0].sentence_id == original_results[0].sentence_id
    assert abs(loaded_results[0].score - original_results[0].score) < 1e-5


def test_bm25_indexing_and_retrieval(sample_documents):
    """Validates BM25 index construction and scoring."""
    retriever = BM25Retriever()
    retriever.index(sample_documents)

    # 1. Standard query retrieval
    query = "Elon Musk founded SpaceX to colonize Mars"
    results = retriever.retrieve(query, top_k=3)

    assert len(results) > 0
    assert len(results) <= 3
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.retriever == "bm25" for r in results)

    # Top result should be SpaceX
    top_result = results[0]
    assert top_result.doc_id == "SpaceX"
    assert "SpaceX" in top_result.text
    assert top_result.score > 0.0

    # 2. Empty query handling
    assert retriever.retrieve("", top_k=5) == []
    assert retriever.retrieve("   ", top_k=5) == []


def test_bm25_save_and_load(sample_documents, tmp_path):
    """Validates BM25 persistence to and from disk."""
    retriever = BM25Retriever()
    retriever.index(sample_documents)

    query = "Marie Curie radioactivity pioneering research"
    original_results = retriever.retrieve(query, top_k=3)

    # Save to tmp_path
    retriever.save(tmp_path)

    # Reload into fresh instance
    new_retriever = BM25Retriever()
    new_retriever.load(tmp_path)
    loaded_results = new_retriever.retrieve(query, top_k=3)

    assert len(loaded_results) == len(original_results)
    assert loaded_results[0].doc_id == original_results[0].doc_id
    assert loaded_results[0].sentence_id == original_results[0].sentence_id
    assert abs(loaded_results[0].score - original_results[0].score) < 1e-5


def test_missing_index_error(tmp_path):
    """Validates IndexNotFoundError when loading from empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(IndexNotFoundError):
        TfidfRetriever().load(empty_dir)

    with pytest.raises(IndexNotFoundError):
        BM25Retriever().load(empty_dir)
