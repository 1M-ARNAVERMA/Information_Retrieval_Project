"""Explainability breakdown endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.routes.health import get_service
from src.api.schemas import ExplainRequest, ExplainResponse
from src.api.service import APIService
from src.common.exceptions import FeverXException

router = APIRouter(tags=["Explainability"])


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Evidence Relevance & NLI Score Breakdown",
    description="Provides granular explainability breakdown across lexical, semantic, entity, reranker, and NLI confidence dimensions for a pair.",
)
def explain_evidence(
    request: ExplainRequest,
    service: APIService = Depends(get_service),
) -> ExplainResponse:
    """Calculates multi-factor explainability score breakdown for a claim and evidence sentence."""
    try:
        return service.explain(request)
    except FeverXException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Explainability evaluation failed: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal explainability failure: {str(exc)}",
        )
