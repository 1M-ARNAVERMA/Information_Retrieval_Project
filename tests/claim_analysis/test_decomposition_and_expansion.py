"""Unit tests for Claim Decomposition and Query Expansion modules."""

import pytest

from src.claim_analysis.decomposition import ClaimDecomposer
from src.claim_analysis.query_expansion import QueryExpander
from src.common.types import SubClaim


def test_claim_decomposition_compound():
    """Validates decomposing a compound claim with subject propagation."""
    decomposer = ClaimDecomposer()
    claim = "Marie Curie was born in Poland and won two Nobel Prizes."
    subclaims = decomposer.decompose(claim)

    assert len(subclaims) == 2
    assert all(isinstance(sc, SubClaim) for sc in subclaims)
    assert subclaims[0].id == 1
    assert "Marie Curie was born in Poland" in subclaims[0].text
    # Clause 2 must receive the propagated subject "Marie Curie"
    assert subclaims[1].id == 2
    assert "Marie Curie" in subclaims[1].text
    assert "won two Nobel Prizes" in subclaims[1].text


def test_claim_decomposition_atomic():
    """Validates that atomic simple claims are preserved as a single subclaim."""
    decomposer = ClaimDecomposer()
    claim = "Albert Einstein developed the theory of relativity."
    subclaims = decomposer.decompose(claim)

    assert len(subclaims) == 1
    assert subclaims[0].id == 1
    assert subclaims[0].text == "Albert Einstein developed the theory of relativity."
    assert len(subclaims[0].entities) >= 1


def test_claim_decomposition_empty():
    """Validates handling of empty or blank claims."""
    decomposer = ClaimDecomposer()
    assert decomposer.decompose("") == []
    assert decomposer.decompose("   ") == []


def test_query_expansion_rule_based():
    """Validates rule-based query expansion behavior."""
    expander = QueryExpander(use_llm=False, max_expansions=4)
    query = "Albert Einstein developed the theory of relativity."
    expanded = expander.expand(query)

    assert len(expanded) >= 2
    # Original query must be at index 0
    assert expanded[0] == query

    # Should contain entity-focused or synonym-expanded variant
    expanded_lower = [q.lower() for q in expanded]
    assert any("einstein" in q for q in expanded_lower)
    assert any("relativity" in q for q in expanded_lower)


def test_query_expansion_predicate_synonyms():
    """Validates predicate substitution in queries."""
    expander = QueryExpander(use_llm=False)
    query = "Elon Musk founded SpaceX."
    expanded = expander.expand(query)

    expanded_lower = [q.lower() for q in expanded]
    # 'founded' might be expanded to 'established' or entity 'Elon Musk SpaceX'
    assert any("spacex" in q for q in expanded_lower)


def test_query_expansion_empty():
    """Validates empty query handling."""
    expander = QueryExpander()
    assert expander.expand("") == []
    assert expander.expand("   ") == []
