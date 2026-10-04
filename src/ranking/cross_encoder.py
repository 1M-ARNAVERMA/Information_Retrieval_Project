"""Cross-Encoder neural reranking module for joint claim-evidence scoring."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from src.common.types import PreprocessMode, RerankedResult, RetrievalResult
from src.preprocessing.pipeline import TextPreprocessor, get_preprocessor
from src.utils.config_loader import load_config, load_model_config
from src.utils.logger import get_logger
from src.utils.model_loader import get_cross_encoder, resolve_device

logger = get_logger("cross_encoder_reranker")


class BaseCrossEncoder(ABC):
    """Abstract interface allowing cross-encoders to be independently replaced."""

    @abstractmethod
    def rerank(
        self,
        claim: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Scores (claim, evidence) candidate pairs and returns reranked results."""
        pass


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Applies numerically stable sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


class CrossEncoderReranker(BaseCrossEncoder):
    """Neural cross-encoder joint reranker using sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        default_top_k: int = 5,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.default_top_k = default_top_k
        self.preprocessor = preprocessor or get_preprocessor()

        self._model = None

    @property
    def model(self) -> Any:
        """Lazy-loaded CrossEncoder model via singleton cache."""
        if self._model is None:
            if self.model_name == "fallback":
                self._model = None
            else:
                try:
                    self._model = get_cross_encoder(
                        model_name=self.model_name,
                        device=self.device,
                    )
                except Exception as e:
                    logger.warning("Could not load neural CrossEncoder (%s); using semantic overlap fallback.", e)
                    self._model = None
        return self._model

    def _fallback_score(self, claim: str, evidence: str) -> float:
        """Heuristic lexical-semantic overlap score for fallback environments."""
        claim_words = set(claim.lower().split())
        ev_words = set(evidence.lower().split())
        if not claim_words or not ev_words:
            return 0.0
        intersection = claim_words.intersection(ev_words)
        # Jaccard + containment score
        jaccard = len(intersection) / len(claim_words.union(ev_words))
        containment = len(intersection) / len(claim_words)
        return float(round(0.4 * jaccard + 0.6 * containment, 4))

    def rerank(
        self,
        claim: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Computes joint cross-encoder interaction scores and reranks candidates.

        Args:
            claim: User claim string.
            candidates: Candidate evidence list from retrieval / RRF.
            top_k: Number of highest scoring evidence items to retain.

        Returns:
            List of RerankedResult models sorted by rerank_score descending.
        """
        if not candidates:
            return []

        k = top_k or self.default_top_k
        clean_claim = self.preprocessor.preprocess(claim, mode=PreprocessMode.SEMANTIC)

        # Prepare pairs: [[claim, evidence_1], [claim, evidence_2], ...]
        pairs = [
            [clean_claim, self.preprocessor.preprocess(cand.text, mode=PreprocessMode.SEMANTIC)]
            for cand in candidates
        ]

        if self.model is not None:
            try:
                raw_scores = self.model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
                raw_arr = np.array(raw_scores, dtype=np.float32)

                # If model outputs unbounded logits, map via sigmoid
                if np.any(raw_arr < 0.0) or np.any(raw_arr > 1.0):
                    scores = _sigmoid(raw_arr).tolist()
                else:
                    scores = raw_arr.tolist()
            except Exception as e:
                logger.warning("Neural cross-encoder prediction failed: %s; using fallback.", e)
                scores = [self._fallback_score(clean_claim, cand.text) for cand in candidates]
        else:
            scores = [self._fallback_score(clean_claim, cand.text) for cand in candidates]

        # Combine candidates with rerank scores
        reranked_items: List[Tuple[float, RerankedResult]] = []
        for cand, r_score in zip(candidates, scores):
            normalized_score = float(round(r_score, 6))
            reranked_item = RerankedResult(
                doc_id=cand.doc_id,
                sentence_id=cand.sentence_id,
                text=cand.text,
                retrieval_score=cand.score,
                rerank_score=normalized_score,
                retriever_origin=cand.retriever,
            )
            reranked_items.append((normalized_score, reranked_item))

        # Sort by rerank_score descending
        reranked_items.sort(key=lambda x: x[0], reverse=True)
        top_results = [item[1] for item in reranked_items[:k]]

        return top_results
