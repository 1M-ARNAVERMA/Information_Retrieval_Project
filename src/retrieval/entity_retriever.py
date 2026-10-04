"""Entity-aware retrieval and boosting module."""

import re
from typing import Dict, List, Optional, Set, Tuple

from src.claim_analysis.ner import extract_entities
from src.common.types import Entity, RetrievalResult
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("entity_retriever")


class EntityAwareRetriever:
    """Ranks and boosts retrieval candidates according to named entity overlap."""

    def __init__(
        self,
        boost_factor: Optional[float] = None,
        title_boost_bonus: float = 0.25,
    ) -> None:
        """Initializes entity booster.

        Args:
            boost_factor: Multiplier for entity overlap ratio (defaults to config.yaml).
            title_boost_bonus: Extra boost when an entity matches the Wikipedia article doc_id.
        """
        if boost_factor is None:
            try:
                cfg = load_config()
                self.boost_factor = cfg.get("retrieval", {}).get("entity_boost_factor", 1.25)
            except Exception:
                self.boost_factor = 1.25
        else:
            self.boost_factor = boost_factor

        self.title_boost_bonus = title_boost_bonus

    def compute_overlap(
        self,
        claim_entities: List[Entity],
        candidate: RetrievalResult,
    ) -> Tuple[float, List[str], float]:
        """Calculates entity overlap ratio and overall multiplier boost for a candidate.

        Args:
            claim_entities: List of entities from the claim.
            candidate: Candidate RetrievalResult.

        Returns:
            Tuple of (overlap_ratio, matched_entity_names, total_multiplier).
        """
        if not claim_entities:
            return 0.0, [], 1.0

        cand_text_lower = candidate.text.lower()
        cand_doc_lower = candidate.doc_id.lower()

        matched: List[str] = []
        has_title_match = False

        for ent in claim_entities:
            ent_clean = ent.text.strip().lower()
            if not ent_clean:
                continue

            # Check direct substring match in text or doc_id
            in_text = ent_clean in cand_text_lower
            in_doc = ent_clean in cand_doc_lower

            if in_doc:
                has_title_match = True

            if in_text or in_doc:
                matched.append(ent.text)
            else:
                # Sub-token overlap for multi-word entities (e.g. 'Einstein' in 'Albert Einstein')
                ent_tokens = [t for t in re.findall(r"\b\w+\b", ent_clean) if len(t) > 2]
                if ent_tokens and any(t in cand_text_lower or t in cand_doc_lower for t in ent_tokens):
                    matched.append(ent.text)

        # Deduplicate matched names
        matched = list(dict.fromkeys(matched))
        overlap_ratio = len(matched) / len(claim_entities)

        # Multiplicative boost: 1.0 + boost_factor * overlap_ratio + title_bonus
        multiplier = 1.0 + (self.boost_factor * overlap_ratio)
        if has_title_match:
            multiplier += self.title_boost_bonus

        return round(overlap_ratio, 4), matched, round(multiplier, 4)

    def rerank(
        self,
        claim: str,
        candidates: List[RetrievalResult],
        entities: Optional[List[Entity]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Boosts candidate retrieval scores using entity matches.

        Args:
            claim: Query or claim text.
            candidates: Input list of RetrievalResult models.
            entities: Optional pre-extracted Entity models.
            top_k: Max candidates to return.

        Returns:
            Reranked list of RetrievalResult models sorted by boosted score descending.
        """
        if not candidates:
            return []

        claim_ents = entities if entities is not None else extract_entities(claim)
        if not claim_ents:
            logger.debug("No entities found in claim '%s'; returning original candidates.", claim)
            return candidates[:top_k] if top_k else candidates

        boosted_results: List[Tuple[float, RetrievalResult]] = []

        for cand in candidates:
            overlap_ratio, matched, multiplier = self.compute_overlap(claim_ents, cand)
            boosted_score = cand.score * multiplier

            # Copy result with updated score and entity metadata
            meta = dict(cand.metadata)
            meta["entity_matches"] = matched
            meta["entity_overlap_ratio"] = overlap_ratio
            meta["entity_multiplier"] = multiplier
            meta["original_score"] = cand.score

            boosted_cand = RetrievalResult(
                doc_id=cand.doc_id,
                sentence_id=cand.sentence_id,
                text=cand.text,
                score=round(boosted_score, 6),
                retriever=f"{cand.retriever}+entity",
                metadata=meta,
            )
            boosted_results.append((boosted_score, boosted_cand))

        # Sort by boosted score descending
        boosted_results.sort(key=lambda x: x[0], reverse=True)
        final_list = [item[1] for item in boosted_results]

        if top_k is not None:
            final_list = final_list[:top_k]

        return final_list
