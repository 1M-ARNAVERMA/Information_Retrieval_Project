"""Generates human-readable natural language audit explanations for verification decisions."""

from typing import Any, Dict, List, Optional

from src.common.types import (
    Entity,
    EvidenceScoreBreakdown,
    NLIPrediction,
    RerankedResult,
    VeracityLabel,
)
from src.explainability.evidence_scoring import EvidenceScorer


class ExplanationGenerator:
    """Produces explainable audit trails and natural language justification for verdicts."""

    def __init__(self, scorer: Optional[EvidenceScorer] = None) -> None:
        self.scorer = scorer or EvidenceScorer()

    def generate_explanation(
        self,
        claim: str,
        verdict: VeracityLabel,
        confidence: float,
        evidence_list: List[RerankedResult],
        nli_predictions: List[NLIPrediction],
        entities: List[Entity],
        score_breakdowns: Optional[List[EvidenceScoreBreakdown]] = None,
    ) -> Dict[str, Any]:
        """Synthesizes an explainability report for the verification outcome.

        Args:
            claim: User claim string.
            verdict: Final VeracityLabel (SUPPORTS, REFUTES, NOT ENOUGH INFO).
            confidence: Estimated composite confidence score.
            evidence_list: Selected top evidence candidates.
            nli_predictions: Sentence-level NLI prediction models.
            entities: Extracted claim entities.
            score_breakdowns: Optional precomputed EvidenceScoreBreakdown items.

        Returns:
            Dictionary containing natural language summary, audit steps, and score breakdowns.
        """
        conf_pct = round(confidence * 100, 1)
        conf_level = "High" if confidence >= 0.80 else ("Moderate" if confidence >= 0.60 else "Low")

        # 1. Summary statement
        if verdict == VeracityLabel.SUPPORTS:
            summary = (
                f"The claim is SUPPORTED with {conf_level} confidence ({conf_pct}%). "
                f"Retrieved Wikipedia evidence verifies the key facts."
            )
        elif verdict == VeracityLabel.REFUTES:
            summary = (
                f"The claim is REFUTED with {conf_level} confidence ({conf_pct}%). "
                f"Retrieved Wikipedia evidence contradicts the asserted facts."
            )
        else:
            summary = (
                f"The claim cannot be verified (NOT ENOUGH INFO, {conf_pct}% confidence). "
                f"Available Wikipedia sentences do not contain sufficient evidence to confirm or deny the claim."
            )

        # 2. Reasoning audit steps
        steps = [
            f"1. Entity Recognition: Extracted {len(entities)} key entities ({', '.join(e.text for e in entities) or 'None'}).",
            f"2. Multi-Stage Retrieval: Retrieved and fused {len(evidence_list)} top candidate sentences from Wikipedia corpus.",
            f"3. Neural Reranking: Applied joint cross-encoder interaction scoring to align query with evidence.",
            f"4. NLI Inference: Evaluated premise-hypothesis entailment and contradiction probabilities.",
            f"5. Verdict Decision: Final verdict concluded as '{verdict.value}' with {conf_level} confidence.",
        ]

        # 3. Evidence item audit cards
        evidence_explanations = []
        pred_map = {(p.doc_id, p.sentence_id): p for p in nli_predictions}

        for idx, ev in enumerate(evidence_list):
            pred = pred_map.get((ev.doc_id, ev.sentence_id))
            breakdown = (
                score_breakdowns[idx]
                if score_breakdowns and idx < len(score_breakdowns)
                else self.scorer.score_evidence(claim, ev, entities, pred)
            )

            rel_str = pred.predicted_label.value if pred else "NEUTRAL"
            prob = (
                pred.entailment_prob
                if rel_str == "SUPPORTS"
                else (pred.contradiction_prob if rel_str == "REFUTES" else pred.neutral_prob)
            ) if pred else 0.5

            evidence_explanations.append({
                "source": f"{ev.doc_id} (Line {ev.sentence_id})",
                "text": ev.text,
                "nli_verdict": rel_str,
                "nli_prob": round(prob, 4),
                "scores": {
                    "lexical_score": breakdown.lexical_score,
                    "semantic_score": breakdown.semantic_score,
                    "entity_score": breakdown.entity_score,
                    "reranker_score": breakdown.reranker_score,
                    "nli_confidence": breakdown.nli_confidence,
                    "composite_score": breakdown.composite_score,
                },
            })

        return {
            "verdict": verdict.value,
            "confidence": confidence,
            "confidence_level": conf_level,
            "summary": summary,
            "reasoning_steps": steps,
            "entities_found": [e.text for e in entities],
            "evidence_count": len(evidence_list),
            "evidence_evaluations": evidence_explanations,
        }
