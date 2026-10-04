"""Unit tests for Phase 11: NLI Verification, Contradiction Detection, and Verdict Mapping."""

import pytest

from src.common.types import NLIPrediction, RerankedResult, VeracityLabel
from src.verification.confidence import ConfidenceEstimator
from src.verification.contradiction_detector import ContradictionDetector
from src.verification.nli_verifier import NLIVerifier
from src.verification.verdict_mapper import VerdictMapper


def test_nli_verifier_heuristic_modes():
    """Validates Entailment, Contradiction, and Neutral predictions."""
    verifier = NLIVerifier(model_name="fallback")

    # 1. Clear Entailment (SUPPORTS)
    e_p, c_p, n_p, label_sup = verifier.predict_pair(
        claim="Albert Einstein developed the theory of relativity.",
        evidence_text="Albert Einstein was a German physicist who developed the theory of relativity.",
    )
    assert label_sup == VeracityLabel.SUPPORTS
    assert e_p > c_p and e_p > n_p

    # 2. Clear Contradiction (REFUTES) via country mismatch
    e_p, c_p, n_p, label_ref = verifier.predict_pair(
        claim="Albert Einstein was born in France.",
        evidence_text="Albert Einstein was born in Germany in 1879.",
    )
    assert label_ref == VeracityLabel.REFUTES
    assert c_p > e_p

    # 3. Neutral (NOT ENOUGH INFO)
    e_p, c_p, n_p, label_nei = verifier.predict_pair(
        claim="Albert Einstein loved playing the saxophone on weekends.",
        evidence_text="Albert Einstein published papers on Brownian motion.",
    )
    assert label_nei == VeracityLabel.NOT_ENOUGH_INFO
    assert n_p > e_p


def test_verdict_mapper_weighted_prob():
    """Validates weighted probability aggregation across evidence items."""
    mapper = VerdictMapper(strategy="weighted_prob")

    p1 = NLIPrediction(
        doc_id="DocA",
        sentence_id=0,
        text="High confidence support",
        entailment_prob=0.90,
        contradiction_prob=0.05,
        neutral_prob=0.05,
        predicted_label=VeracityLabel.SUPPORTS,
    )
    p2 = NLIPrediction(
        doc_id="DocB",
        sentence_id=1,
        text="Low confidence neutral",
        entailment_prob=0.10,
        contradiction_prob=0.10,
        neutral_prob=0.80,
        predicted_label=VeracityLabel.NOT_ENOUGH_INFO,
    )

    # Weights: give p1 3x more weight than p2
    verdict, prob_dist = mapper.map_verdict([p1, p2], weights=[0.75, 0.25])
    assert verdict == VeracityLabel.SUPPORTS
    assert prob_dist["SUPPORTS"] > prob_dist["REFUTES"]


def test_contradiction_priority_rule():
    """Validates that a single definitive contradiction triggers refutation."""
    mapper = VerdictMapper(contradiction_priority_threshold=0.75)

    p_support = NLIPrediction(
        doc_id="DocA",
        sentence_id=0,
        text="Albert Einstein was famous.",
        entailment_prob=0.70,
        contradiction_prob=0.10,
        neutral_prob=0.20,
        predicted_label=VeracityLabel.SUPPORTS,
    )
    p_refute = NLIPrediction(
        doc_id="DocB",
        sentence_id=1,
        text="He was born in Germany, not France.",
        entailment_prob=0.05,
        contradiction_prob=0.92,  # Strong contradiction
        neutral_prob=0.03,
        predicted_label=VeracityLabel.REFUTES,
    )

    verdict, _ = mapper.map_verdict([p_support, p_refute], weights=[0.5, 0.5])
    assert verdict == VeracityLabel.REFUTES


def test_verdict_mapper_strategies():
    """Validates majority vote and max confidence strategies."""
    preds = [
        NLIPrediction(doc_id="A", sentence_id=0, text="T1", entailment_prob=0.85, contradiction_prob=0.1, neutral_prob=0.05, predicted_label=VeracityLabel.SUPPORTS),
        NLIPrediction(doc_id="B", sentence_id=0, text="T2", entailment_prob=0.80, contradiction_prob=0.1, neutral_prob=0.10, predicted_label=VeracityLabel.SUPPORTS),
        NLIPrediction(doc_id="C", sentence_id=0, text="T3", entailment_prob=0.10, contradiction_prob=0.70, neutral_prob=0.20, predicted_label=VeracityLabel.REFUTES),
    ]

    maj_mapper = VerdictMapper(strategy="majority_vote")
    maj_verdict, _ = maj_mapper.map_verdict(preds)
    assert maj_verdict == VeracityLabel.SUPPORTS

    max_mapper = VerdictMapper(strategy="max_confidence")
    max_verdict, _ = max_mapper.map_verdict(preds)
    assert max_verdict == VeracityLabel.SUPPORTS


def test_contradiction_detector():
    """Validates contradiction alert extraction."""
    detector = ContradictionDetector(contradiction_threshold=0.80)
    preds = [
        NLIPrediction(doc_id="DocA", sentence_id=0, text="Text A", entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05, predicted_label=VeracityLabel.SUPPORTS),
        NLIPrediction(doc_id="DocB", sentence_id=1, text="Text B", entailment_prob=0.05, contradiction_prob=0.88, neutral_prob=0.07, predicted_label=VeracityLabel.REFUTES),
    ]

    alerts = detector.find_contradictions(preds)
    assert len(alerts) == 1
    assert alerts[0]["doc_id"] == "DocB"
    assert detector.has_conflict(preds) is True


def test_confidence_estimator():
    """Validates confidence calculation bounded between 0.35 and 0.99."""
    estimator = ConfidenceEstimator()

    # High margin & quality
    high_conf = estimator.estimate_confidence(
        verdict=VeracityLabel.SUPPORTS,
        prob_dist={"SUPPORTS": 0.92, "REFUTES": 0.04, "NOT ENOUGH INFO": 0.04},
        evidence_scores=[0.95, 0.90],
    )
    assert 0.80 <= high_conf <= 0.99

    # Neutral verdict
    nei_conf = estimator.estimate_confidence(
        verdict=VeracityLabel.NOT_ENOUGH_INFO,
        prob_dist={"SUPPORTS": 0.20, "REFUTES": 0.10, "NOT ENOUGH INFO": 0.70},
    )
    assert nei_conf == 0.70
