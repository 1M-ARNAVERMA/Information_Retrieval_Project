"""Comprehensive end-to-end integration tests for the full FEVER-X verification pipeline."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.common.exceptions import PipelineError
from src.common.types import VeracityLabel, VerificationResult
from src.pipeline.full_pipeline import FactVerificationPipeline


@pytest.fixture(scope="module")
def pipeline():
    """Initializes a shared FactVerificationPipeline instance."""
    return FactVerificationPipeline(use_fallback_models=True)


@pytest.fixture(scope="module")
def api_client():
    """Provides a FastAPI test client."""
    return TestClient(app)


def test_e2e_supports_claim(pipeline):
    """Validates complete flow for a factual supported claim."""
    claim = "Albert Einstein was a German physicist."
    result = pipeline.verify(claim=claim, top_k=3)

    assert isinstance(result, VerificationResult)
    assert result.claim == claim
    assert result.verdict in [VeracityLabel.SUPPORTS, VeracityLabel.NOT_ENOUGH_INFO]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_evidence) > 0
    assert len(result.entities) > 0

    # Knowledge graph integrity
    assert len(result.graph.nodes) >= 3
    assert len(result.graph.edges) >= 2
    claim_nodes = [n for n in result.graph.nodes if n.label == "CLAIM"]
    assert len(claim_nodes) == 1
    assert claim_nodes[0].id == "claim:root"

    # Explainability integrity
    exp = result.explanation
    assert "summary" in exp
    assert "reasoning_steps" in exp
    assert len(exp["reasoning_steps"]) >= 4


def test_e2e_refutes_claim(pipeline):
    """Validates complete flow for a contradictory claim."""
    claim = "Albert Einstein was born in France and never studied physics."
    result = pipeline.verify(claim=claim, top_k=3)

    assert isinstance(result, VerificationResult)
    assert result.claim == claim
    # Contradiction detector should flag country mismatch or never studied physics
    assert result.verdict in [VeracityLabel.REFUTES, VeracityLabel.NOT_ENOUGH_INFO]
    assert len(result.top_evidence) > 0

    # Verify at least one NLI prediction has contradiction or neutral score
    assert any(
        p.contradiction_prob > 0.3 or p.neutral_prob > 0.3
        for p in result.nli_predictions
    )


def test_e2e_not_enough_info_claim(pipeline):
    """Validates complete flow for an unverified/out-of-corpus claim."""
    claim = "Albert Einstein won an Olympic medal for swimming in 1912."
    result = pipeline.verify(claim=claim, top_k=3)

    assert isinstance(result, VerificationResult)
    # Since Wikipedia sample mentions physics but no swimming medal
    assert result.verdict in [VeracityLabel.NOT_ENOUGH_INFO, VeracityLabel.REFUTES]
    assert 0.0 <= result.confidence <= 1.0


def test_e2e_compound_claim_decomposition(pipeline):
    """Validates decomposition of multi-premise claims into subclaims."""
    claim = "Albert Einstein was a German physicist and he developed the theory of relativity."
    result = pipeline.verify(claim=claim, top_k=3, use_decomposition=True)

    assert len(result.subclaims) >= 2
    subclaim_nodes = [n for n in result.graph.nodes if n.label == "SUBCLAIM"]
    assert len(subclaim_nodes) >= 2

    # Check DECOMPOSED_INTO edges in graph
    decomp_edges = [
        e for e in result.graph.edges if e.relation == "DECOMPOSED_INTO"
    ]
    assert len(decomp_edges) >= 2


def test_e2e_fever_brackets_and_special_characters(pipeline):
    """Validates robust handling of FEVER normalized bracket tokens and numbers."""
    claim = "Albert Einstein -LRB- born 1879 -RRB- was awarded the Nobel Prize in Physics."
    result = pipeline.verify(claim=claim, top_k=3)

    assert isinstance(result, VerificationResult)
    assert len(result.top_evidence) > 0
    assert any("Einstein" in ev.text for ev in result.top_evidence)


def test_e2e_modular_aggregation_strategies(pipeline):
    """Validates all 3 NLI verdict aggregation strategies."""
    claim = "Albert Einstein was a German physicist."
    for strategy in ["weighted_prob", "max_confidence", "majority_vote"]:
        res = pipeline.verify(claim=claim, top_k=3, aggregation_strategy=strategy)
        assert res.verdict in [
            VeracityLabel.SUPPORTS,
            VeracityLabel.REFUTES,
            VeracityLabel.NOT_ENOUGH_INFO,
        ]
        assert res.retrieval_metadata["aggregation_strategy"] == strategy


def test_e2e_modular_feature_toggles(pipeline):
    """Validates running pipeline with different feature toggle combinations."""
    claim = "Albert Einstein developed relativity."

    # Without decomposition and query expansion
    res_minimal = pipeline.verify(
        claim=claim,
        top_k=2,
        use_decomposition=False,
        use_query_expansion=False,
        use_colbert=False,
    )
    assert len(res_minimal.subclaims) == 0
    assert len(res_minimal.retrieval_metadata["queries_used"]) == 1

    # With ColBERT late-interaction
    res_colbert = pipeline.verify(
        claim=claim,
        top_k=2,
        use_colbert=True,
    )
    assert len(res_colbert.top_evidence) > 0


def test_e2e_explainability_score_bounds(pipeline):
    """Validates that all evidence factor scores fall strictly within [0.0, 1.0]."""
    claim = "Albert Einstein was a physicist."
    res = pipeline.verify(claim=claim, top_k=3)

    breakdowns = res.explanation.get("evidence_breakdowns", [])
    assert len(breakdowns) > 0

    for bd in breakdowns:
        assert 0.0 <= bd["lexical_score"] <= 1.0
        assert 0.0 <= bd["semantic_score"] <= 1.0
        assert 0.0 <= bd["entity_score"] <= 1.0
        assert 0.0 <= bd["reranker_score"] <= 1.0
        assert 0.0 <= bd["nli_confidence"] <= 1.0
        assert 0.0 <= bd["composite_score"] <= 1.0


def test_e2e_error_handling_invalid_claims(pipeline):
    """Validates that empty claims raise PipelineError."""
    with pytest.raises(PipelineError):
        pipeline.verify("")

    with pytest.raises(PipelineError):
        pipeline.verify("   ")


def test_e2e_api_pipeline_parity(api_client, pipeline):
    """Validates that FastAPI POST /verify produces consistent results with direct pipeline."""
    claim = "Albert Einstein was a German physicist."

    # Direct execution
    direct_res = pipeline.verify(claim=claim, top_k=3)

    # API execution
    api_resp = api_client.post(
        "/verify",
        json={"claim": claim, "top_k": 3, "include_graph": True},
    )
    assert api_resp.status_code == 200
    api_data = api_resp.json()

    assert api_data["claim"] == direct_res.claim
    assert api_data["verdict"] == direct_res.verdict.value
    assert len(api_data["top_evidence"]) == len(direct_res.top_evidence)
    assert len(api_data["graph"]["nodes"]) == len(direct_res.graph.nodes)
