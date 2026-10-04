"""Unit tests for Phase 2: FEVER Data Ingestion module."""

from pathlib import Path
import pytest

from src.common.exceptions import DataIngestionError
from src.common.types import VeracityLabel
from src.data_ingestion.fever_loader import load_claims, parse_fever_record, stream_claims
from src.data_ingestion.normalizer import normalize_fever_text, normalize_wiki_title
from src.data_ingestion.wiki_loader import load_documents, parse_wiki_page, stream_documents

SAMPLE_FEVER = Path("data/sample/fever_sample.jsonl")
SAMPLE_WIKI = Path("data/sample/wiki_sample.jsonl")


def test_normalizer_text():
    """Validates FEVER bracket token normalization."""
    raw = "Albert Einstein -LRB- 1879 - 1955 -RRB- was a physicist -COLON- yes ."
    expected = "Albert Einstein ( 1879 - 1955 ) was a physicist : yes ."
    assert normalize_fever_text(raw) == expected
    assert normalize_fever_text("") == ""


def test_normalizer_wiki_title():
    """Validates Wikipedia article title normalization."""
    assert normalize_wiki_title("Albert_Einstein") == "Albert Einstein"
    assert normalize_wiki_title("Marie_Curie_-LRB-scientist-RRB-") == "Marie Curie (scientist)"
    assert normalize_wiki_title("New%20York%20City") == "New York City"
    assert normalize_wiki_title("") == ""


def test_parse_fever_record():
    """Validates parsing a raw FEVER JSON entry into a typed Claim."""
    raw_support = {
        "id": 1001,
        "verifiable": "VERIFIABLE",
        "label": "SUPPORTS",
        "claim": "Albert Einstein developed the theory of relativity .",
        "evidence": [[[1, 1, "Albert_Einstein", 0]]],
    }
    claim = parse_fever_record(raw_support)
    assert claim.id == 1001
    assert claim.claim == "Albert Einstein developed the theory of relativity ."
    assert claim.label == VeracityLabel.SUPPORTS
    assert len(claim.evidence) == 1
    assert claim.evidence[0].items[0].doc_id == "Albert Einstein"
    assert claim.evidence[0].items[0].sentence_id == 0

    # Test NOT ENOUGH INFO claim with null evidence
    raw_nei = {
        "id": 1003,
        "verifiable": "NOT VERIFIABLE",
        "label": "NOT ENOUGH INFO",
        "claim": "Albert Einstein loved painting watercolors .",
        "evidence": [[[3, None, None, None]]],
    }
    nei_claim = parse_fever_record(raw_nei)
    assert nei_claim.label == VeracityLabel.NOT_ENOUGH_INFO
    assert len(nei_claim.evidence) == 0


def test_load_and_stream_claims():
    """Validates loading claims from the sample JSONL file."""
    assert SAMPLE_FEVER.exists()

    claims = load_claims(SAMPLE_FEVER)
    assert len(claims) == 6
    assert claims[0].id == 1001
    assert claims[0].label == VeracityLabel.SUPPORTS

    # Test filtering by label
    supports_only = load_claims(SAMPLE_FEVER, labels=[VeracityLabel.SUPPORTS])
    assert all(c.label == VeracityLabel.SUPPORTS for c in supports_only)
    assert len(supports_only) == 3

    # Test limit
    limited = load_claims(SAMPLE_FEVER, limit=2)
    assert len(limited) == 2


def test_parse_wiki_page():
    """Validates parsing a Wikipedia page with tab-delimited lines into Documents."""
    raw_page = {
        "id": "Albert_Einstein",
        "text": "Albert Einstein developed relativity .\nHe was born in Germany .",
        "lines": "0\tAlbert Einstein developed relativity .\tAlbert_Einstein\t0\n1\tHe was born in Germany .\tAlbert_Einstein\t1",
    }
    docs = parse_wiki_page(raw_page)
    assert len(docs) == 2
    assert docs[0].doc_id == "Albert Einstein"
    assert docs[0].sentence_id == 0
    assert docs[0].text == "Albert Einstein developed relativity ."
    assert docs[1].sentence_id == 1
    assert docs[1].text == "He was born in Germany ."


def test_load_and_stream_documents():
    """Validates streaming and loading Wikipedia sentences from sample file."""
    assert SAMPLE_WIKI.exists()

    docs = load_documents(SAMPLE_WIKI)
    assert len(docs) == 9  # 3 articles * 3 sentences each
    assert any(d.doc_id == "Albert Einstein" for d in docs)
    assert any(d.doc_id == "Marie Curie" for d in docs)
    assert any(d.doc_id == "SpaceX" for d in docs)

    # Test sentence limit
    limited_docs = load_documents(SAMPLE_WIKI, limit_sentences=4)
    assert len(limited_docs) == 4


def test_nonexistent_files():
    """Validates proper domain exception on missing dataset files."""
    with pytest.raises(DataIngestionError):
        list(stream_claims("nonexistent_claims.jsonl"))

    with pytest.raises(DataIngestionError):
        list(stream_documents("nonexistent_wiki.jsonl"))
