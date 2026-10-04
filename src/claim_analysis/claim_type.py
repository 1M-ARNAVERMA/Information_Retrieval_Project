"""Claim classification and linguistic feature analysis for FEVER-X."""

from enum import Enum
import re
from typing import Any, Dict, List, Optional

from src.claim_analysis.ner import extract_entities
from src.common.types import Entity


class ClaimType(str, Enum):
    """Categorical taxonomy of claim structure."""
    NUMERICAL = "NUMERICAL"
    TEMPORAL = "TEMPORAL"
    ENTITY_RELATION = "ENTITY_RELATION"
    GENERAL = "GENERAL"


# Patterns for numeric and ordinal indicators
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "first", "second", "third", "fourth", "fifth", "single", "double", "triple",
}
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b", re.IGNORECASE)

# Patterns for temporal markers
TEMPORAL_KEYWORDS = {
    "born", "died", "during", "before", "after", "since", "until", "year", "century",
}


def classify_claim_type(
    claim_text: str,
    entities: Optional[List[Entity]] = None,
) -> Dict[str, Any]:
    """Classifies a claim into structural types (numerical, temporal, relation, general).

    Args:
        claim_text: Claim string.
        entities: Optional pre-extracted named entities.

    Returns:
        Dictionary with 'primary_type', 'is_numerical', 'is_temporal', 'entity_count'.
    """
    if not claim_text:
        return {
            "primary_type": ClaimType.GENERAL.value,
            "is_numerical": False,
            "is_temporal": False,
            "entity_count": 0,
        }

    ents = entities if entities is not None else extract_entities(claim_text)
    lower_text = claim_text.lower()
    tokens = set(re.findall(r"\b\w+\b", lower_text))

    has_digits = bool(NUMBER_PATTERN.search(claim_text))
    has_num_words = bool(tokens.intersection(NUMBER_WORDS))
    is_numerical = has_digits or has_num_words

    has_date_ent = any(e.label == "DATE" for e in ents)
    has_temp_words = bool(tokens.intersection(TEMPORAL_KEYWORDS))
    is_temporal = has_date_ent or has_temp_words

    # Determine primary type
    if is_temporal:
        primary_type = ClaimType.TEMPORAL
    elif is_numerical:
        primary_type = ClaimType.NUMERICAL
    elif len(ents) >= 2:
        primary_type = ClaimType.ENTITY_RELATION
    else:
        primary_type = ClaimType.GENERAL

    return {
        "primary_type": primary_type.value,
        "is_numerical": is_numerical,
        "is_temporal": is_temporal,
        "entity_count": len(ents),
    }
