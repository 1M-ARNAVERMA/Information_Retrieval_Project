"""Unit tests for Named Entity Recognition and Claim Type Analysis."""

import pytest

from src.claim_analysis.claim_type import ClaimType, classify_claim_type
from src.claim_analysis.ner import EntityExtractor, extract_entities
from src.common.types import Entity


def test_ner_persons_and_orgs():
    """Validates extraction of PERSON and ORG entities."""
    claim = "Elon Musk founded SpaceX in 2002."
    entities = extract_entities(claim)

    assert len(entities) >= 2
    ent_texts = [e.text for e in entities]
    assert any("Elon Musk" in t for t in ent_texts)
    assert any("SpaceX" in t for t in ent_texts)


def test_ner_gpe_and_date():
    """Validates extraction of GPE and DATE entities."""
    claim = "Marie Curie was born in Poland on 7 November 1867."
    entities = extract_entities(claim)

    ent_labels = {e.label for e in entities}
    ent_texts = [e.text for e in entities]

    assert "GPE" in ent_labels or any("Poland" in t for t in ent_texts)
    assert "DATE" in ent_labels or any("1867" in t for t in ent_texts)


def test_ner_empty():
    """Validates empty and whitespace handling."""
    assert extract_entities("") == []
    assert extract_entities("   ") == []


def test_claim_type_classification():
    """Validates taxonomy classification of different claim types."""
    # 1. Temporal
    c_temp = "Albert Einstein was born in 1879."
    res_temp = classify_claim_type(c_temp)
    assert res_temp["is_temporal"] is True
    assert res_temp["primary_type"] == ClaimType.TEMPORAL.value

    # 2. Numerical
    c_num = "Marie Curie won two Nobel Prizes."
    res_num = classify_claim_type(c_num)
    assert res_num["is_numerical"] is True
    assert res_num["primary_type"] in (ClaimType.NUMERICAL.value, ClaimType.ENTITY_RELATION.value)

    # 3. Entity Relation
    c_rel = "Elon Musk founded SpaceX."
    res_rel = classify_claim_type(c_rel)
    assert res_rel["primary_type"] == ClaimType.ENTITY_RELATION.value
    assert res_rel["entity_count"] >= 2

    # 4. General / Empty
    res_empty = classify_claim_type("")
    assert res_empty["primary_type"] == ClaimType.GENERAL.value
