"""NLI Verification and Verdict Aggregation package for FEVER-X."""

from src.verification.confidence import ConfidenceEstimator
from src.verification.contradiction_detector import ContradictionDetector
from src.verification.nli_verifier import NLIVerifier
from src.verification.verdict_mapper import VerdictMapper

__all__ = [
    "NLIVerifier",
    "ContradictionDetector",
    "VerdictMapper",
    "ConfidenceEstimator",
]
