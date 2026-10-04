"""Unit tests for Phase 16: Streamlit UI components and BackendClient."""

import pytest
import plotly.graph_objects as go

from frontend.api_client import BackendClient
from frontend.components.evidence_card import render_evidence_breakdown_chart
from frontend.components.graph_view import build_plotly_network


@pytest.fixture
def backend_client():
    """Provides a BackendClient running in direct in-process mode."""
    return BackendClient(prefer_direct=True)


def test_backend_client_health(backend_client):
    """Validates health check retrieval via BackendClient."""
    health = backend_client.get_health()
    assert health["status"] == "healthy"
    assert "indices_loaded" in health
    assert health["num_indexed_documents"] > 0


def test_backend_client_verify(backend_client):
    """Validates end-to-end claim verification via BackendClient."""
    res = backend_client.verify_claim(
        claim="Albert Einstein was a German physicist.",
        top_k=3,
        include_graph=True,
    )
    assert res["verdict"] in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_evidence"]) > 0
    assert len(res["graph"]["nodes"]) > 0


def test_backend_client_retrieve(backend_client):
    """Validates candidate retrieval via BackendClient."""
    res = backend_client.retrieve_candidates(
        query="Albert Einstein relativity",
        top_k=3,
    )
    assert res["total_results"] > 0
    assert len(res["results"]) <= 3


def test_backend_client_ablation(backend_client):
    """Validates ablation results retrieval via BackendClient."""
    res = backend_client.get_ablation_results()
    assert res["num_configurations"] >= 5
    assert len(res["results"]) > 0


def test_render_evidence_breakdown_chart():
    """Validates Plotly horizontal bar chart generation for evidence scoring."""
    breakdown = {
        "lexical_score": 0.45,
        "semantic_score": 0.82,
        "entity_score": 1.0,
        "reranker_score": 0.91,
        "nli_confidence": 0.85,
        "composite_score": 0.81,
    }
    fig = render_evidence_breakdown_chart(breakdown)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 5


def test_build_plotly_network():
    """Validates Plotly 2D interactive network graph construction."""
    nodes = [
        {"id": "claim:root", "label": "CLAIM", "attributes": {"text": "Einstein physicist"}},
        {"id": "entity:albert_einstein", "label": "ENTITY", "attributes": {"text": "Albert Einstein"}},
        {"id": "evidence:Albert_Einstein:0", "label": "EVIDENCE", "attributes": {"text": "German physicist", "verdict": "SUPPORTS"}},
    ]
    edges = [
        {"source": "claim:root", "target": "entity:albert_einstein", "relation": "MENTIONS"},
        {"source": "claim:root", "target": "evidence:Albert_Einstein:0", "relation": "SUPPORTS"},
    ]
    fig = build_plotly_network(nodes, edges)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # edge trace + node trace
