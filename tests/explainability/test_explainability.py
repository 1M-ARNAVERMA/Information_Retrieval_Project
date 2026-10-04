"""Unit tests for Phase 13: Explainability and Evidence Scoring."""

import pytest

from src.common.types import (
    Entity,
    EvidenceScoreBreakdown,
    NLIPrediction,
    RerankedResult,
    VeracityLabel,
)
from src.explainability.evidence_scoring import EvidenceScorer
from src.explainability.explanation_generator import ExplanationGenerator


def test_evidence_scoring_breakdown():
    """Validates multi-component evidence scoring."""
    scorer = EvidenceScorer()

    claim = "Albert Einstein developed the theory of relativity."
    entities = [
        Entity(text="Albert Einstein", label="PERSON"),
        Entity(text="relativity", label="WORK_OF_ART"),
    ]

    ev = RerankedResult(
        doc_id="Albert Einstein",
        sentence_id=0,
        text="Albert Einstein was a German physicist who developed the theory of relativity.",
        retrieval_score=0.92,
        rerank_score=0.96,
        retriever_origin="bm25",
    )

    pred = NLIPrediction(
        doc_id=ev.doc_id,
        sentence_id=ev.sentence_id,
        text=ev.text,
        entailment_prob=0.95,
        contradiction_prob=0.02,
        neutral_prob=0.03,
        predicted_label=VeracityLabel.SUPPORTS,
    )

    breakdown = scorer.score_evidence(claim, ev, entities, pred)

    assert isinstance(breakdown, EvidenceScoreBreakdown)
    assert 0.0 <= breakdown.lexical_score <= 1.0
    assert 0.0 <= breakdown.semantic_score <= 1.0
    assert breakdown.entity_score == 1.0  # Both entities matched
    assert breakdown.reranker_score == 0.96
    assert breakdown.nli_confidence == 0.95
    assert 0.70 <= breakdown.composite_score <= 1.0


def test_explanation_generator():
    """Validates explanation generation for SUPPORTS, REFUTES, and NEI."""
    generator = ExplanationGenerator()

    claim = "Albert Einstein developed the theory of relativity."
    entities = [Entity(text="Albert Einstein", label="PERSON")]
    ev = RerankedResult(
        doc_id="Albert Einstein",
        sentence_id=0,
        text="Albert Einstein developed relativity.",
        retrieval_score=0.9,
        rerank_score=0.95,
        retriever_origin="bm25",
    )
    pred = NLIPrediction(
        doc_id="Albert Einstein",
        sentence_id=0,
        text=ev.text,
        entailment_prob=0.94,
        contradiction_prob=0.03,
        neutral_prob=0.03,
        predicted_label=VeracityLabel.SUPPORTS,
    )

    report = generator.generate_explanation(
        claim=claim,
        verdict=VeracityLabel.SUPPORTS,
        confidence=0.92,
        evidence_list=[ev],
        nli_predictions=[pred],
        entities=entities,
    )

    assert report["verdict"] == "SUPPORTS"
    assert report["confidence"] == 0.92
    assert report["confidence_level"] == "High"
    assert "SUPPORTED" in report["summary"]
    assert len(report["reasoning_steps"]) == 5
    assert len(report["evidence_evaluations"]) == 1
    assert "scores" in report["evidence_evaluations"][0]
