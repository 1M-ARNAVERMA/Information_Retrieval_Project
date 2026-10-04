"""Claim decomposition into atomic, verifiable subclaims with subject propagation."""

import re
from typing import List, Optional

from src.claim_analysis.ner import extract_entities
from src.common.types import Entity, SubClaim
from src.utils.logger import get_logger

logger = get_logger("claim_decomposition")

# Coordinating conjunctions and clause boundary splitters
CONJUNCTION_PATTERNS = [
    re.compile(r",\s*and\s+", re.IGNORECASE),
    re.compile(r"\s+and\s+(?:also\s+)?", re.IGNORECASE),
    re.compile(r",\s*but\s+", re.IGNORECASE),
    re.compile(r"\s+as\s+well\s+as\s+", re.IGNORECASE),
    re.compile(r",\s*who\s+", re.IGNORECASE),
    re.compile(r",\s*which\s+", re.IGNORECASE),
    re.compile(r";\s*", re.IGNORECASE),
]


class ClaimDecomposer:
    """Decomposes compound or multi-hop claims into atomic subclaims."""

    def __init__(self) -> None:
        pass

    def extract_subject(self, text: str) -> Optional[str]:
        """Heuristically extracts the grammatical subject (initial noun phrase/named entity)."""
        entities = extract_entities(text)
        if entities:
            # First entity is usually the subject
            return entities[0].text

        # Fallback: take leading words up to the first verb
        words = text.strip().split()
        if len(words) >= 2:
            return " ".join(words[:2])
        return words[0] if words else None

    def has_subject(self, clause: str) -> bool:
        """Determines whether a clause already begins with a candidate noun phrase or pronoun."""
        words = clause.strip().split()
        if not words:
            return False
        first_word = words[0].lower()
        pronouns = {"he", "she", "it", "they", "we", "i", "who", "which"}
        if first_word in pronouns:
            return True

        # Check if first word is capitalized (likely a noun/entity)
        if words[0][0].isupper() and len(words[0]) > 1:
            return True

        return False

    def decompose(self, claim: str) -> List[SubClaim]:
        """Splits a claim into atomic subclaims, propagating the subject if necessary.

        Example:
            "Marie Curie was born in Poland and won two Nobel Prizes."
            -> [
                SubClaim(id=1, text="Marie Curie was born in Poland."),
                SubClaim(id=2, text="Marie Curie won two Nobel Prizes.")
            ]

        Args:
            claim: Compound or simple claim string.

        Returns:
            List of SubClaim models.
        """
        claim = claim.strip()
        if not claim:
            return []

        # Remove trailing period for clean clause processing
        has_period = claim.endswith(".")
        clean_claim = claim[:-1].strip() if has_period else claim

        subject = self.extract_subject(clean_claim)
        raw_clauses: List[str] = [clean_claim]

        # Try splitting on conjunctions
        for pattern in CONJUNCTION_PATTERNS:
            new_clauses = []
            for cl in raw_clauses:
                splits = pattern.split(cl)
                for s in splits:
                    s_clean = s.strip()
                    if s_clean:
                        new_clauses.append(s_clean)
            if len(new_clauses) > len(raw_clauses):
                raw_clauses = new_clauses
                break  # Split on primary conjunction

        # If no split occurred, return atomic claim
        if len(raw_clauses) <= 1:
            text = clean_claim + ("." if not clean_claim.endswith(".") else "")
            return [
                SubClaim(
                    id=1,
                    text=text,
                    entities=extract_entities(text),
                )
            ]

        subclaims: List[SubClaim] = []
        for idx, clause in enumerate(raw_clauses, start=1):
            clause_text = clause.strip()

            # Propagate subject to subsequent clauses if they lack one
            if idx > 1 and subject and not self.has_subject(clause_text):
                clause_text = f"{subject} {clause_text}"

            # Ensure proper capitalization and sentence termination
            if clause_text and clause_text[0].islower():
                clause_text = clause_text[0].upper() + clause_text[1:]
            if not clause_text.endswith("."):
                clause_text += "."

            subclaims.append(
                SubClaim(
                    id=idx,
                    text=clause_text,
                    entities=extract_entities(clause_text),
                )
            )

        return subclaims
