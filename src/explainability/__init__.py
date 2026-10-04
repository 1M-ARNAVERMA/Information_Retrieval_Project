"""Explainability package for FEVER-X: multi-factor scoring and audit trails."""

from src.explainability.evidence_scoring import EvidenceScorer
from src.explainability.explanation_generator import ExplanationGenerator

__all__ = [
    "EvidenceScorer",
    "ExplanationGenerator",
]
