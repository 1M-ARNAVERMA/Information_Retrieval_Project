"""FastAPI Pydantic request and response schemas for FEVER-X microservice."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.common.types import (
    Entity,
    EvidenceGraphData,
    EvidenceScoreBreakdown,
    NLIPrediction,
    RerankedResult,
    RetrievalResult,
    SubClaim,
    VeracityLabel,
)


class VerifyRequest(BaseModel):
    """Payload for POST /verify endpoint."""
    claim: str = Field(..., min_length=2, description="The textual claim to verify.")
    top_k: int = Field(5, ge=1, le=50, description="Number of evidence sentences to return.")
    include_graph: bool = Field(True, description="Whether to generate the evidence graph.")
    aggregation_strategy: str = Field(
        "weighted_prob",
        description="Verdict aggregation strategy: 'weighted_prob', 'max_confidence', or 'majority_vote'.",
    )
    use_decomposition: bool = Field(True, description="Whether to decompose complex claims into subclaims.")
    use_query_expansion: bool = Field(True, description="Whether to expand queries with synonyms and entities.")
    use_colbert: bool = Field(False, description="Whether to include ColBERT late-interaction retrieval.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "claim": "Albert Einstein was a German physicist.",
                "top_k": 5,
                "include_graph": True,
                "aggregation_strategy": "weighted_prob",
            }
        }
    }


class VerifyResponse(BaseModel):
    """Response returned by POST /verify endpoint."""
    claim: str
    verdict: str
    confidence: float
    top_evidence: List[RerankedResult] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    subclaims: List[SubClaim] = Field(default_factory=list)
    nli_predictions: List[NLIPrediction] = Field(default_factory=list)
    explanation: Dict[str, Any] = Field(default_factory=dict)
    graph: EvidenceGraphData = Field(default_factory=EvidenceGraphData)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float


class RetrieveRequest(BaseModel):
    """Payload for POST /retrieve endpoint."""
    query: str = Field(..., min_length=1, description="Search query or claim.")
    top_k: int = Field(10, ge=1, le=100, description="Max candidate sentences to retrieve.")
    methods: List[str] = Field(
        default=["bm25", "sentence_transformer", "tfidf"],
        description="Retrievers to query: 'bm25', 'sentence_transformer', 'tfidf', 'colbert'.",
    )
    use_entity_boost: bool = Field(True, description="Apply entity overlap boosting.")
    use_rrf: bool = Field(True, description="Combine results with Reciprocal Rank Fusion.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Albert Einstein physics Nobel Prize",
                "top_k": 5,
                "methods": ["bm25", "sentence_transformer", "tfidf"],
                "use_entity_boost": True,
                "use_rrf": True,
            }
        }
    }


class RetrieveResponse(BaseModel):
    """Response returned by POST /retrieve endpoint."""
    query: str
    total_results: int
    results: List[RetrievalResult] = Field(default_factory=list)
    execution_time_ms: float


class ExplainRequest(BaseModel):
    """Payload for POST /explain endpoint."""
    claim: str = Field(..., min_length=2, description="Claim statement.")
    evidence_text: str = Field(..., min_length=2, description="Candidate evidence sentence.")
    evidence_doc_id: str = Field("custom_doc", description="Document ID.")
    evidence_sentence_id: int = Field(0, description="Sentence index.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "claim": "Albert Einstein was born in Germany.",
                "evidence_text": "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879.",
                "evidence_doc_id": "Albert_Einstein",
                "evidence_sentence_id": 0,
            }
        }
    }


class ExplainResponse(BaseModel):
    """Response returned by POST /explain endpoint."""
    claim: str
    evidence_text: str
    verdict_prediction: str
    confidence: float
    breakdown: EvidenceScoreBreakdown
    explanation_summary: str
    audit_steps: List[str]
    execution_time_ms: float


class AblationResponse(BaseModel):
    """Response returned by GET /evaluate/ablation endpoint."""
    results: List[Dict[str, Any]]
    num_configurations: int
    source: str


class HealthResponse(BaseModel):
    """Response returned by GET /health endpoint."""
    status: str
    app_name: str
    version: str
    device: str
    indices_loaded: Dict[str, bool]
    num_indexed_documents: int
    uptime_seconds: float
