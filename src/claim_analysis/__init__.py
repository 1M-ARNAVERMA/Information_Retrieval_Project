"""Claim analysis package for FEVER-X: NER, decomposition, and claim taxonomy."""

from src.claim_analysis.claim_type import ClaimType, classify_claim_type
from src.claim_analysis.decomposition import ClaimDecomposer
from src.claim_analysis.ner import EntityExtractor, extract_entities, get_entity_extractor
from src.claim_analysis.query_expansion import QueryExpander

__all__ = [
    "EntityExtractor",
    "extract_entities",
    "get_entity_extractor",
    "ClaimType",
    "classify_claim_type",
    "ClaimDecomposer",
    "QueryExpander",
]
