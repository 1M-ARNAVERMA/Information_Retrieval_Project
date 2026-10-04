"""Shared Pydantic data schemas and contracts for FEVER-X.

All module boundaries must use these standardized types.
Avoid passing raw untyped dictionaries between components.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class VeracityLabel(str, Enum):
    """FEVER benchmark verification labels."""
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT ENOUGH INFO"


class PreprocessMode(str, Enum):
    """Modes supported by the text preprocessing pipeline."""
    LEXICAL = "lexical"     # Lowercase, punctuation removal, stop-words filtered, stemmed
    SEMANTIC = "semantic"   # Preserves case, grammar, and sentence structure for neural models
    RAW = "raw"             # Whitespace cleanup only


class Entity(BaseModel):
    """Named entity extracted from claim or evidence."""
    text: str
    label: str               # e.g., "PERSON", "ORG", "GPE", "DATE", "WORK_OF_ART"
    start_char: int = 0
    end_char: int = 0

    model_config = {"frozen": True}


class SubClaim(BaseModel):
    """Atomic proposition generated from claim decomposition."""
    id: int
    text: str
    entities: List[Entity] = Field(default_factory=list)


class Document(BaseModel):
    """A Wikipedia document/sentence candidate in the corpus."""
    doc_id: str                          # Wikipedia page title / ID
    sentence_id: int                     # Line number / sentence index in Wikipedia page
    text: str
    entities: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Gold or predicted single evidence piece."""
    doc_id: str
    sentence_id: int
    text: Optional[str] = None


class EvidenceSet(BaseModel):
    """Set of evidence sentences supporting or refuting a claim."""
    items: List[EvidenceItem] = Field(default_factory=list)


class Claim(BaseModel):
    """FEVER dataset claim instance."""
    id: Optional[int] = None
    claim: str
    label: Optional[VeracityLabel] = None
    evidence: List[EvidenceSet] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Candidate sentence returned by any retriever (TF-IDF, BM25, Dense, etc.)."""
    doc_id: str
    sentence_id: int
    text: str
    score: float
    retriever: str                       # e.g., "tfidf", "bm25", "sentence_transformer", "colbert", "rrf"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RerankedResult(BaseModel):
    """Candidate evaluated by the Cross-Encoder neural reranker."""
    doc_id: str
    sentence_id: int
    text: str
    retrieval_score: float
    rerank_score: float
    retriever_origin: str


class NLIPrediction(BaseModel):
    """NLI model inference result for a (claim, evidence) pair."""
    doc_id: str
    sentence_id: int
    text: str
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float
    predicted_label: VeracityLabel


class EvidenceScoreBreakdown(BaseModel):
    """Explainable component breakdown of evidence relevance and veracity."""
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    entity_score: float = 0.0
    reranker_score: float = 0.0
    nli_confidence: float = 0.0
    composite_score: float = 0.0


class GraphNode(BaseModel):
    """Node in the explainability / evidence graph."""
    id: str
    label: str                           # "CLAIM", "ENTITY", "EVIDENCE"
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Directed edge in the evidence graph."""
    source: str
    target: str
    relation: str                        # "MENTIONS", "SUPPORTS", "REFUTES", "CO_OCCURS"
    weight: float = 1.0


class EvidenceGraphData(BaseModel):
    """Complete serialized evidence graph for frontend rendering or reasoning."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Structured end-to-end outcome of the FEVER-X verification pipeline."""
    claim: str
    verdict: VeracityLabel
    confidence: float
    top_evidence: List[RerankedResult] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    subclaims: List[SubClaim] = Field(default_factory=list)
    nli_predictions: List[NLIPrediction] = Field(default_factory=list)
    explanation: Dict[str, Any] = Field(default_factory=dict)
    graph: EvidenceGraphData = Field(default_factory=EvidenceGraphData)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
