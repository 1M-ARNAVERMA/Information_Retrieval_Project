"""Unit tests verifying Phase 1 project skeleton, types, config, and utilities."""

import os
from pathlib import Path
import pytest

from src.common.types import (
    VeracityLabel,
    PreprocessMode,
    Entity,
    SubClaim,
    Document,
    Claim,
    EvidenceItem,
    EvidenceSet,
    RetrievalResult,
    RerankedResult,
    NLIPrediction,
    EvidenceScoreBreakdown,
    GraphNode,
    GraphEdge,
    EvidenceGraphData,
    VerificationResult,
)
from src.common.exceptions import (
    FeverXException,
    ConfigurationError,
    DataIngestionError,
)
from src.utils.logger import get_logger
from src.utils.config_loader import load_config, load_model_config
from src.utils.file_utils import (
    ensure_dir,
    read_jsonl,
    write_jsonl,
    save_pickle,
    load_pickle,
)


def test_veracity_labels():
    """Validates that standard FEVER veracity labels exist."""
    assert VeracityLabel.SUPPORTS == "SUPPORTS"
    assert VeracityLabel.REFUTES == "REFUTES"
    assert VeracityLabel.NOT_ENOUGH_INFO == "NOT ENOUGH INFO"


def test_shared_data_models():
    """Validates instantiation and serialization of core shared models."""
    entity = Entity(text="Albert Einstein", label="PERSON", start_char=0, end_char=15)
    assert entity.text == "Albert Einstein"

    doc = Document(doc_id="Albert_Einstein", sentence_id=0, text="Albert Einstein was a physicist.")
    assert doc.doc_id == "Albert_Einstein"
    assert doc.sentence_id == 0

    claim = Claim(
        id=101,
        claim="Albert Einstein developed relativity.",
        label=VeracityLabel.SUPPORTS,
        evidence=[
            EvidenceSet(items=[EvidenceItem(doc_id="Albert_Einstein", sentence_id=0, text="He was a physicist.")])
        ],
    )
    assert claim.claim == "Albert Einstein developed relativity."
    assert claim.label == VeracityLabel.SUPPORTS

    retrieval = RetrievalResult(
        doc_id="Albert_Einstein",
        sentence_id=0,
        text="He was a physicist.",
        score=0.95,
        retriever="bm25",
    )
    assert retrieval.retriever == "bm25"
    assert retrieval.score == 0.95

    verification = VerificationResult(
        claim=claim.claim,
        verdict=VeracityLabel.SUPPORTS,
        confidence=0.92,
        top_evidence=[
            RerankedResult(
                doc_id=retrieval.doc_id,
                sentence_id=retrieval.sentence_id,
                text=retrieval.text,
                retrieval_score=retrieval.score,
                rerank_score=0.98,
                retriever_origin=retrieval.retriever,
            )
        ],
        explanation={"summary": "Confirmed via Wikipedia."},
    )
    assert verification.verdict == VeracityLabel.SUPPORTS
    assert verification.confidence == 0.92
    assert len(verification.top_evidence) == 1


def test_config_loader():
    """Validates loading configuration from config/."""
    cfg = load_config()
    assert "app" in cfg
    assert cfg["app"]["name"] == "FEVER-X"
    assert "retrieval" in cfg
    assert cfg["retrieval"]["top_k_candidates"] == 50

    model_cfg = load_model_config()
    assert "models" in model_cfg
    assert "sentence_transformer" in model_cfg["models"]
    assert "runtime" in model_cfg


def test_logger():
    """Validates logger creation and non-crashing execution."""
    logger = get_logger("test_module")
    assert logger is not None
    logger.info("Test log line from skeleton test.")


def test_file_utils(tmp_path):
    """Validates JSONL and Pickle reading and writing."""
    sample_records = [
        {"id": 1, "title": "Test 1"},
        {"id": 2, "title": "Test 2"},
    ]
    jsonl_path = tmp_path / "test.jsonl"
    count = write_jsonl(jsonl_path, sample_records)
    assert count == 2

    loaded_records = list(read_jsonl(jsonl_path))
    assert loaded_records == sample_records

    pickle_path = tmp_path / "test.pkl"
    save_pickle(sample_records, pickle_path)
    loaded_pickle = load_pickle(pickle_path)
    assert loaded_pickle == sample_records


def test_exceptions_hierarchy():
    """Validates custom domain exception hierarchy."""
    assert issubclass(ConfigurationError, FeverXException)
    assert issubclass(DataIngestionError, FeverXException)
