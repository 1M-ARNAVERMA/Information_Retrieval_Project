"""Pretrained NLI inference module for claim verification against evidence premises."""

import re
from typing import Any, List, Optional, Tuple
import numpy as np

from src.common.types import NLIPrediction, PreprocessMode, RerankedResult, VeracityLabel
from src.preprocessing.pipeline import TextPreprocessor, get_preprocessor
from src.utils.config_loader import load_model_config
from src.utils.logger import get_logger
from src.utils.model_loader import resolve_device

logger = get_logger("nli_verifier")

# Standard 3-class NLI mapping: 0 -> Contradiction, 1 -> Entailment, 2 -> Neutral (or model-specific)
# In FEVER: Entailment = SUPPORTS, Contradiction = REFUTES, Neutral = NOT ENOUGH INFO


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Computes row-wise softmax probabilities."""
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class NLIVerifier:
    """Cross-attention NLI verifier (Premise=Evidence, Hypothesis=Claim)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.preprocessor = preprocessor or get_preprocessor()

        self._model = None
        self._tokenizer = None

    def _init_model(self) -> None:
        """Initializes NLI transformer sequence classifier or fallback."""
        if self._model is not None:
            return

        if not self.model_name:
            try:
                cfg = load_model_config()
                self.model_name = cfg.get("models", {}).get(
                    "nli_model", "cross-encoder/nli-deberta-v3-small"
                )
            except Exception:
                self.model_name = "fallback"

        if self.model_name == "fallback":
            self._model = None
            return

        logger.info("Initializing NLI model '%s' on %s...", self.model_name, self.device)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, local_files_only=True
                ).to(self.device)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                ).to(self.device)

            self._model.eval()
            logger.info("NLI model successfully loaded.")
        except Exception as e:
            logger.warning("Could not initialize neural NLI model (%s); using heuristic fallback.", e)
            self._model = None
            self._tokenizer = None

    def _heuristic_nli(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """Heuristic rule-based NLI inference for offline environments and fast unit testing.

        Returns:
            (entailment_prob, contradiction_prob, neutral_prob)
        """
        p_clean = premise.lower()
        h_clean = hypothesis.lower()

        p_words = set(re.findall(r"\b\w+\b", p_clean))
        h_words = set(re.findall(r"\b\w+\b", h_clean))

        # Country / Demonym mapping
        country_map = {
            "france": "france", "french": "france",
            "germany": "germany", "german": "germany",
            "poland": "poland", "polish": "poland",
            "usa": "usa", "american": "usa", "america": "usa",
            "uk": "uk", "british": "uk", "britain": "uk",
        }
        p_countries = {country_map[w] for w in p_words if w in country_map}
        h_countries = {country_map[w] for w in h_words if w in country_map}
        country_mismatch = bool(p_countries and h_countries and p_countries.isdisjoint(h_countries))

        # Negation conflict detection
        negations = {"not", "never", "no", "neither", "cannot"}
        h_has_neg = any(n in h_words for n in negations)
        # Check if premise has explicit negation of the hypothesis action/state
        p_neg_match = bool(re.search(r"\b(never|not|did not|was not|is not)\b", p_clean))
        
        shared = p_words.intersection(h_words)
        
        # Claim negates premise or premise negates claim on shared topic
        negation_conflict = False
        if h_has_neg and not p_neg_match and len(shared) >= 2:
            negation_conflict = True
        elif p_neg_match and not h_has_neg:
            # Check if negation is directly near shared content (not just 'not to be confused with')
            if "not to be confused" not in p_clean and any(re.search(rf"\b(not|never)\s+\w*\s*{w}\b", p_clean) for w in shared if len(w) > 3):
                negation_conflict = True

        if country_mismatch or negation_conflict:
            return 0.05, 0.90, 0.05

        # Stopwords filter for content overlap
        stopwords = {
            "the", "a", "an", "was", "is", "in", "at", "for", "of", "and", "or",
            "by", "to", "it", "his", "her", "he", "she", "with", "as", "on", "from",
        }
        h_content = {w for w in h_words if w not in stopwords}
        shared_content = {w for w in shared if w not in stopwords}

        # Check entity/subject preservation: If hypothesis mentions proper entities (e.g. Einstein),
        # the premise must mention at least one key entity word to entail.
        raw_h_tokens = re.findall(r"\b[A-Z][a-z]+\b", hypothesis)
        h_entities = {t.lower() for t in raw_h_tokens if t.lower() not in stopwords}
        if h_entities and not h_entities.intersection(p_words):
            # Evidence doesn't even mention the target entity/subject of the claim
            return 0.05, 0.05, 0.90

        content_overlap = len(shared_content) / max(len(h_content), 1)

        if content_overlap >= 0.50:
            return 0.88, 0.04, 0.08
        elif content_overlap >= 0.25:
            return 0.20, 0.10, 0.70
        else:
            return 0.05, 0.05, 0.90

    def predict_pair(self, claim: str, evidence_text: str) -> Tuple[float, float, float, VeracityLabel]:
        """Infers NLI probabilities for a single (claim, evidence) pair.

        Premise = evidence_text, Hypothesis = claim.
        """
        self._init_model()

        if self._model is not None and self._tokenizer is not None:
            try:
                import torch

                inputs = self._tokenizer(
                    evidence_text,
                    claim,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    logits = outputs.logits.cpu().numpy()[0]
                    probs = _softmax(logits)

                # Standard DeBERTa-v3 NLI mapping: 0=contradiction, 1=entailment, 2=neutral
                c_prob = float(probs[0])
                e_prob = float(probs[1])
                n_prob = float(probs[2])
            except Exception as e:
                logger.warning("Neural NLI prediction failed: %s; using heuristic fallback.", e)
                e_prob, c_prob, n_prob = self._heuristic_nli(evidence_text, claim)
        else:
            e_prob, c_prob, n_prob = self._heuristic_nli(evidence_text, claim)

        # Map highest probability to VeracityLabel
        max_idx = int(np.argmax([e_prob, c_prob, n_prob]))
        if max_idx == 0:
            label = VeracityLabel.SUPPORTS
        elif max_idx == 1:
            label = VeracityLabel.REFUTES
        else:
            label = VeracityLabel.NOT_ENOUGH_INFO

        return round(e_prob, 4), round(c_prob, 4), round(n_prob, 4), label

    def predict(
        self,
        claim: str,
        evidence_list: List[RerankedResult],
    ) -> List[NLIPrediction]:
        """Runs NLI inference across all reranked evidence sentences.

        Args:
            claim: Target claim string.
            evidence_list: List of RerankedResult models.

        Returns:
            List of NLIPrediction models.
        """
        predictions: List[NLIPrediction] = []
        for ev in evidence_list:
            e_prob, c_prob, n_prob, label = self.predict_pair(claim, ev.text)
            predictions.append(
                NLIPrediction(
                    doc_id=ev.doc_id,
                    sentence_id=ev.sentence_id,
                    text=ev.text,
                    entailment_prob=e_prob,
                    contradiction_prob=c_prob,
                    neutral_prob=n_prob,
                    predicted_label=label,
                )
            )
        return predictions
