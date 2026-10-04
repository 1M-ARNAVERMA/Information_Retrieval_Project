"""Unified API and in-process backend client for Streamlit frontend."""

import httpx
from typing import Any, Dict, List, Optional

from src.api.schemas import (
    AblationResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    VerifyRequest,
    VerifyResponse,
)
from src.api.service import APIService
from src.utils.logger import get_logger

logger = get_logger("frontend_client")


class BackendClient:
    """Client for frontend to interact with backend either via HTTP or in-process singleton."""

    def __init__(self, base_url: str = "http://localhost:8000", prefer_direct: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.prefer_direct = prefer_direct
        self._direct_service: Optional[APIService] = None

    def _get_direct_service(self) -> APIService:
        if self._direct_service is None:
            self._direct_service = APIService.get_instance()
        return self._direct_service

    def get_health(self) -> Dict[str, Any]:
        """Fetches health diagnostics."""
        if not self.prefer_direct:
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                logger.debug("FastAPI backend unreachable; falling back to direct service.")

        # Fallback to direct service
        srv = self._get_direct_service()
        res = srv.get_health()
        return res.model_dump()

    def verify_claim(
        self,
        claim: str,
        top_k: int = 5,
        include_graph: bool = True,
        aggregation_strategy: str = "weighted_prob",
        use_decomposition: bool = True,
        use_query_expansion: bool = True,
        use_colbert: bool = False,
    ) -> Dict[str, Any]:
        """Verifies a claim."""
        payload = {
            "claim": claim,
            "top_k": top_k,
            "include_graph": include_graph,
            "aggregation_strategy": aggregation_strategy,
            "use_decomposition": use_decomposition,
            "use_query_expansion": use_query_expansion,
            "use_colbert": use_colbert,
        }

        if not self.prefer_direct:
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(f"{self.base_url}/verify", json=payload)
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                logger.debug("FastAPI endpoint unreachable; using direct pipeline.")

        # Direct in-process execution
        req = VerifyRequest(**payload)
        srv = self._get_direct_service()
        res = srv.verify(req)
        return res.model_dump()

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        use_entity_boost: bool = True,
        use_rrf: bool = True,
    ) -> Dict[str, Any]:
        """Retrieves candidate sentences."""
        active_methods = methods or ["bm25", "sentence_transformer", "tfidf"]
        payload = {
            "query": query,
            "top_k": top_k,
            "methods": active_methods,
            "use_entity_boost": use_entity_boost,
            "use_rrf": use_rrf,
        }

        if not self.prefer_direct:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(f"{self.base_url}/retrieve", json=payload)
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                logger.debug("FastAPI /retrieve unreachable; using direct pipeline.")

        req = RetrieveRequest(**payload)
        srv = self._get_direct_service()
        res = srv.retrieve(req)
        return res.model_dump()

    def explain_evidence(
        self,
        claim: str,
        evidence_text: str,
        evidence_doc_id: str = "custom_doc",
        evidence_sentence_id: int = 0,
    ) -> Dict[str, Any]:
        """Computes granular score breakdown for a claim and evidence pair."""
        payload = {
            "claim": claim,
            "evidence_text": evidence_text,
            "evidence_doc_id": evidence_doc_id,
            "evidence_sentence_id": evidence_sentence_id,
        }

        if not self.prefer_direct:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(f"{self.base_url}/explain", json=payload)
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                logger.debug("FastAPI /explain unreachable; using direct pipeline.")

        req = ExplainRequest(**payload)
        srv = self._get_direct_service()
        res = srv.explain(req)
        return res.model_dump()

    def get_ablation_results(self) -> Dict[str, Any]:
        """Fetches ablation benchmark results."""
        if not self.prefer_direct:
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(f"{self.base_url}/evaluate/ablation")
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                logger.debug("FastAPI /evaluate/ablation unreachable; using direct service.")

        srv = self._get_direct_service()
        res = srv.get_ablation_results()
        return res.model_dump()
