"""Aggregation strategies for mapping multiple evidence NLI predictions to a final verdict."""

from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.common.types import NLIPrediction, RerankedResult, VeracityLabel
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("verdict_mapper")


class VerdictMapper:
    """Aggregates multiple sentence-level NLI inferences into a unified claim veracity verdict.

    Supported strategies:
        - 'weighted_prob': Evidence-weighted expectation over Entailment/Contradiction/Neutral.
        - 'max_confidence': Highest-scoring decisive evidence sentence.
        - 'majority_vote': Consensus across all top evidence items.
    """

    def __init__(
        self,
        strategy: str = "weighted_prob",
        neutral_threshold: float = 0.50,
        contradiction_priority_threshold: float = 0.75,
    ) -> None:
        try:
            cfg = load_config()
            verif_cfg = cfg.get("verification", {})
            self.strategy = strategy or verif_cfg.get("aggregation_strategy", "weighted_prob")
            self.neutral_threshold = neutral_threshold or verif_cfg.get("neutral_threshold", 0.50)
        except Exception:
            self.strategy = strategy or "weighted_prob"
            self.neutral_threshold = neutral_threshold or 0.50

        self.contradiction_priority_threshold = contradiction_priority_threshold

    def map_verdict(
        self,
        predictions: List[NLIPrediction],
        weights: Optional[List[float]] = None,
    ) -> Tuple[VeracityLabel, Dict[str, float]]:
        """Determines final verdict and aggregate probability distribution.

        Args:
            predictions: List of sentence-level NLIPrediction items.
            weights: Optional weights (e.g. reranker relevance scores) per prediction.

        Returns:
            Tuple of (Final VeracityLabel, Aggregate Probability Dict).
        """
        if not predictions:
            return VeracityLabel.NOT_ENOUGH_INFO, {"SUPPORTS": 0.0, "REFUTES": 0.0, "NOT ENOUGH INFO": 1.0}

        # 1. Contradiction Priority Rule: In fact verification, a single definitive refuting
        # fact (e.g. 'Born in Germany' when claim is 'Born in France') refutes the claim.
        top_contradiction = max(p.contradiction_prob for p in predictions)
        if top_contradiction >= self.contradiction_priority_threshold:
            logger.info("High contradiction probability (%.2f) triggered refutation priority.", top_contradiction)
            return VeracityLabel.REFUTES, {
                "SUPPORTS": 0.05,
                "REFUTES": top_contradiction,
                "NOT ENOUGH INFO": 1.0 - top_contradiction,
            }

        if self.strategy == "majority_vote":
            return self._majority_vote(predictions)
        elif self.strategy == "max_confidence":
            return self._max_confidence(predictions)
        else:
            return self._weighted_prob(predictions, weights)

    def _weighted_prob(
        self,
        predictions: List[NLIPrediction],
        weights: Optional[List[float]],
    ) -> Tuple[VeracityLabel, Dict[str, float]]:
        """Weights each NLI prediction by its evidence relevance score."""
        n = len(predictions)
        if weights is None or len(weights) != n or sum(weights) == 0:
            w = [1.0 / n] * n
        else:
            total_w = sum(weights)
            w = [x / total_w for x in weights]

        p_support = sum(w[i] * predictions[i].entailment_prob for i in range(n))
        p_refute = sum(w[i] * predictions[i].contradiction_prob for i in range(n))
        p_neutral = sum(w[i] * predictions[i].neutral_prob for i in range(n))

        prob_dist = {
            "SUPPORTS": round(float(p_support), 4),
            "REFUTES": round(float(p_refute), 4),
            "NOT ENOUGH INFO": round(float(p_neutral), 4),
        }

        # If neutral probability is dominant or evidence is insufficient
        if p_neutral >= self.neutral_threshold and p_neutral > p_support and p_neutral > p_refute:
            return VeracityLabel.NOT_ENOUGH_INFO, prob_dist

        if p_support > p_refute:
            return VeracityLabel.SUPPORTS, prob_dist
        elif p_refute > p_support:
            return VeracityLabel.REFUTES, prob_dist
        else:
            return VeracityLabel.NOT_ENOUGH_INFO, prob_dist

    def _max_confidence(
        self,
        predictions: List[NLIPrediction],
    ) -> Tuple[VeracityLabel, Dict[str, float]]:
        """Picks the single most decisive evidence sentence."""
        best_p = max(predictions, key=lambda p: max(p.entailment_prob, p.contradiction_prob))
        prob_dist = {
            "SUPPORTS": best_p.entailment_prob,
            "REFUTES": best_p.contradiction_prob,
            "NOT ENOUGH INFO": best_p.neutral_prob,
        }
        return best_p.predicted_label, prob_dist

    def _majority_vote(
        self,
        predictions: List[NLIPrediction],
    ) -> Tuple[VeracityLabel, Dict[str, float]]:
        """Determines verdict by label mode across evidence items."""
        labels = [p.predicted_label for p in predictions]
        counts = Counter(labels)
        most_common_label = counts.most_common(1)[0][0]

        total = len(predictions)
        prob_dist = {
            "SUPPORTS": round(counts[VeracityLabel.SUPPORTS] / total, 4),
            "REFUTES": round(counts[VeracityLabel.REFUTES] / total, 4),
            "NOT ENOUGH INFO": round(counts[VeracityLabel.NOT_ENOUGH_INFO] / total, 4),
        }
        return most_common_label, prob_dist
