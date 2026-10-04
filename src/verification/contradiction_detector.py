"""Contradiction detector resolving semantic conflicts and polarity inversions."""

import re
from typing import Dict, List, Optional, Tuple

from src.common.types import NLIPrediction, VeracityLabel


class ContradictionDetector:
    """Detects explicit factual contradictions and polarity oppositions."""

    def __init__(self, contradiction_threshold: float = 0.70) -> None:
        self.contradiction_threshold = contradiction_threshold

    def find_contradictions(
        self,
        predictions: List[NLIPrediction],
    ) -> List[Dict[str, str]]:
        """Identifies any evidence sentences that refute the claim.

        Args:
            predictions: List of NLIPrediction models.

        Returns:
            List of contradiction alerts with doc_id, text, and contradiction confidence.
        """
        contradictions = []
        for p in predictions:
            if p.contradiction_prob >= self.contradiction_threshold:
                contradictions.append({
                    "doc_id": p.doc_id,
                    "sentence_id": str(p.sentence_id),
                    "evidence_text": p.text,
                    "contradiction_prob": str(p.contradiction_prob),
                    "reason": f"Evidence directly refutes claim with {p.contradiction_prob:.1%} confidence.",
                })
        return contradictions

    def has_conflict(self, predictions: List[NLIPrediction]) -> bool:
        """Returns True if there is both strong support and strong refutation in the evidence set."""
        has_support = any(p.entailment_prob >= 0.70 for p in predictions)
        has_refute = any(p.contradiction_prob >= self.contradiction_threshold for p in predictions)
        return has_support and has_refute
