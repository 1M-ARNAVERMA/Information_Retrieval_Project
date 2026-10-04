"""Confidence score calculation and evidence agreement estimation.

NOTE: As per project guidelines, this metric is an uncalibrated composite score
derived from model margins and retrieval quality, NOT a formally calibrated Bayesian probability.
"""

from typing import Dict, List, Optional
import numpy as np

from src.common.types import NLIPrediction, VeracityLabel


class ConfidenceEstimator:
    """Estimates decision confidence based on NLI margins and evidence relevance."""

    def __init__(self, quality_weight: float = 0.3, margin_weight: float = 0.7) -> None:
        self.quality_weight = quality_weight
        self.margin_weight = margin_weight

    def estimate_confidence(
        self,
        verdict: VeracityLabel,
        prob_dist: Dict[str, float],
        evidence_scores: Optional[List[float]] = None,
    ) -> float:
        """Calculates a normalized composite confidence score in [0.0, 1.0].

        Components:
            1. Margin: Difference between winner probability and runner-up.
            2. Evidence Quality: Mean reranking/retrieval score of selected evidence.

        Args:
            verdict: The decided VeracityLabel.
            prob_dist: Dict with 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO' probabilities.
            evidence_scores: Reranker/retrieval scores of the evidence sentences.

        Returns:
            Float confidence score rounded to 4 decimals.
        """
        if verdict == VeracityLabel.NOT_ENOUGH_INFO:
            # If not enough info, confidence reflects certainty that information is missing
            return float(round(prob_dist.get("NOT ENOUGH INFO", 0.5), 4))

        sorted_probs = sorted(prob_dist.values(), reverse=True)
        winner_prob = sorted_probs[0] if sorted_probs else 0.5
        runner_up_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        prob_margin = max(0.0, winner_prob - runner_up_prob)

        # Evidence quality factor
        if evidence_scores and len(evidence_scores) > 0:
            avg_quality = float(np.mean(evidence_scores))
            # Clip quality to [0.0, 1.0]
            avg_quality = min(max(avg_quality, 0.0), 1.0)
        else:
            avg_quality = winner_prob

        composite = (self.margin_weight * prob_margin) + (self.quality_weight * avg_quality)
        # Normalize and scale to a realistic confidence range [0.35, 0.99]
        bounded = float(np.clip(composite, 0.35, 0.99))
        return round(bounded, 4)
