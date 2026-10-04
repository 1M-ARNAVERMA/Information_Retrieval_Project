"""Multi-factor explainable evidence scoring and relevance breakdown."""

import re
from typing import Dict, List, Optional
import numpy as np

from src.common.types import (
    Entity,
    EvidenceScoreBreakdown,
    NLIPrediction,
    RerankedResult,
    RetrievalResult,
)


class EvidenceScorer:
    """Computes a multi-component explainability breakdown for evidence relevance.

    NOTE: The composite score is a normalized explainability heuristic, NOT a formally
    calibrated Bayesian probability.
    """

    def __init__(
        self,
        w_lexical: float = 0.15,
        w_semantic: float = 0.20,
        w_entity: float = 0.25,
        w_reranker: float = 0.25,
        w_nli: float = 0.15,
    ) -> None:
        self.w_lex = w_lexical
        self.w_sem = w_semantic
        self.w_ent = w_entity
        self.w_rerank = w_reranker
        self.w_nli = w_nli

        # Normalize weights to sum to 1.0
        total = self.w_lex + self.w_sem + self.w_ent + self.w_rerank + self.w_nli
        self.w_lex /= total
        self.w_sem /= total
        self.w_ent /= total
        self.w_rerank /= total
        self.w_nli /= total

    def compute_lexical_overlap(self, claim: str, evidence_text: str) -> float:
        """Calculates token Jaccard overlap between claim and evidence."""
        c_tokens = set(re.findall(r"\b\w+\b", claim.lower()))
        e_tokens = set(re.findall(r"\b\w+\b", evidence_text.lower()))
        if not c_tokens or not e_tokens:
            return 0.0
        intersection = c_tokens.intersection(e_tokens)
        return float(len(intersection) / len(c_tokens.union(e_tokens)))

    def compute_entity_score(self, claim_entities: List[Entity], evidence_text: str) -> float:
        """Calculates fraction of claim entities mentioned in evidence text."""
        if not claim_entities:
            return 1.0  # If claim has no entities, no penalty

        ev_lower = evidence_text.lower()
        matched = 0
        for ent in claim_entities:
            ent_str = ent.text.strip().lower()
            if ent_str and ent_str in ev_lower:
                matched += 1
            else:
                # Check major token match (e.g. 'Einstein' in 'Albert Einstein')
                tokens = [t for t in ent_str.split() if len(t) > 2]
                if tokens and any(t in ev_lower for t in tokens):
                    matched += 1

        return float(matched / len(claim_entities))

    def score_evidence(
        self,
        claim: str,
        evidence: RerankedResult,
        claim_entities: List[Entity],
        nli_prediction: Optional[NLIPrediction] = None,
    ) -> EvidenceScoreBreakdown:
        """Produces a granular score breakdown for a single evidence item.

        Args:
            claim: User claim string.
            evidence: RerankedResult model.
            claim_entities: List of extracted Entity models.
            nli_prediction: Optional NLIPrediction for this evidence.

        Returns:
            EvidenceScoreBreakdown model.
        """
        # 1. Lexical score
        lex_score = self.compute_lexical_overlap(claim, evidence.text)

        # 2. Semantic score (from initial retrieval or reranker)
        sem_score = float(np.clip(evidence.retrieval_score, 0.0, 1.0))

        # 3. Entity score
        ent_score = self.compute_entity_score(claim_entities, evidence.text)

        # 4. Cross-encoder reranker score
        rerank_score = float(np.clip(evidence.rerank_score, 0.0, 1.0))

        # 5. NLI confidence (strength of non-neutral conviction)
        if nli_prediction is not None:
            nli_conf = float(max(nli_prediction.entailment_prob, nli_prediction.contradiction_prob))
        else:
            nli_conf = rerank_score

        # Composite score
        composite = (
            (self.w_lex * lex_score)
            + (self.w_sem * sem_score)
            + (self.w_ent * ent_score)
            + (self.w_rerank * rerank_score)
            + (self.w_nli * nli_conf)
        )

        return EvidenceScoreBreakdown(
            lexical_score=round(lex_score, 4),
            semantic_score=round(sem_score, 4),
            entity_score=round(ent_score, 4),
            reranker_score=round(rerank_score, 4),
            nli_confidence=round(nli_conf, 4),
            composite_score=round(float(composite), 4),
        )
