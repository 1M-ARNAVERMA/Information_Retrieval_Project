"""Unit and integration tests for Phase 15: FastAPI Microservice Endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_api_root():
    """Validates the root endpoint metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "FEVER-X" in data["name"]
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_health_endpoint():
    """Validates system health and index readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indices_loaded" in data
    assert data["indices_loaded"]["tfidf"] is True
    assert data["indices_loaded"]["bm25"] is True
    assert data["indices_loaded"]["dense"] is True
    assert data["num_indexed_documents"] > 0
    assert data["uptime_seconds"] >= 0.0


def test_retrieve_endpoint():
    """Validates candidate retrieval via POST /retrieve."""
    payload = {
        "query": "Albert Einstein theoretical physics",
        "top_k": 3,
        "methods": ["bm25", "sentence_transformer", "tfidf"],
        "use_entity_boost": True,
        "use_rrf": True,
    }
    response = client.post("/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["total_results"] > 0
    assert len(data["results"]) <= 3
    assert data["execution_time_ms"] > 0
    top = data["results"][0]
    assert "doc_id" in top
    assert "sentence_id" in top
    assert "text" in top
    assert "score" in top


def test_verify_endpoint_supports():
    """Validates end-to-end claim verification resulting in SUPPORTS."""
    payload = {
        "claim": "Albert Einstein was a German physicist.",
        "top_k": 3,
        "include_graph": True,
        "aggregation_strategy": "weighted_prob",
    }
    response = client.post("/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["claim"] == payload["claim"]
    assert data["verdict"] in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_evidence"]) > 0
    assert "nodes" in data["graph"]
    assert "edges" in data["graph"]
    assert len(data["graph"]["nodes"]) > 0
    assert "summary" in data["explanation"]
    assert "reasoning_steps" in data["explanation"]


def test_verify_endpoint_refutes():
    """Validates verification of a contradictory claim."""
    payload = {
        "claim": "Albert Einstein was born in France and never studied physics.",
        "top_k": 3,
        "include_graph": True,
    }
    response = client.post("/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]
    assert "execution_time_ms" in data


def test_explain_endpoint():
    """Validates granular explainability breakdown via POST /explain."""
    payload = {
        "claim": "Albert Einstein developed the theory of relativity.",
        "evidence_text": "Albert Einstein was best known for developing the theory of relativity.",
        "evidence_doc_id": "Albert_Einstein",
        "evidence_sentence_id": 1,
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["claim"] == payload["claim"]
    assert data["evidence_text"] == payload["evidence_text"]
    assert data["verdict_prediction"] == "SUPPORTS"
    assert data["confidence"] > 0.5
    breakdown = data["breakdown"]
    assert "lexical_score" in breakdown
    assert "semantic_score" in breakdown
    assert "entity_score" in breakdown
    assert "composite_score" in breakdown
    assert len(data["audit_steps"]) > 0


def test_evaluate_ablation_endpoint():
    """Validates GET /evaluate/ablation benchmark retrieval."""
    response = client.get("/evaluate/ablation")
    assert response.status_code == 200
    data = response.json()
    assert data["num_configurations"] >= 5
    assert len(data["results"]) == data["num_configurations"]
    config_names = [r["Configuration"] for r in data["results"]]
    assert any("TF-IDF" in c for c in config_names)
    assert any("BM25" in c for c in config_names)


def test_verify_validation_error():
    """Validates 422 Unprocessable Entity on empty or short claim."""
    response = client.post("/verify", json={"claim": ""})
    assert response.status_code == 422


def test_retrieve_validation_error():
    """Validates 422 Unprocessable Entity on invalid top_k parameter."""
    response = client.post("/retrieve", json={"query": "test", "top_k": -5})
    assert response.status_code == 422
