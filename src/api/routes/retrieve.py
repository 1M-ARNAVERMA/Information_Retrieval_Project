"""Evidence candidate retrieval endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.routes.health import get_service
from src.api.schemas import RetrieveRequest, RetrieveResponse
from src.api.service import APIService
from src.common.exceptions import FeverXException

router = APIRouter(tags=["Retrieval"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Multi-Stage Evidence Retrieval",
    description="Runs TF-IDF, BM25, and Sentence Transformer neural retrievers with RRF fusion and entity overlap boosting.",
)
def retrieve_evidence(
    request: RetrieveRequest,
    service: APIService = Depends(get_service),
) -> RetrieveResponse:
    """Retrieves ranked candidate evidence sentences for an arbitrary query."""
    try:
        return service.retrieve(request)
    except FeverXException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Retrieval error: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal retrieval failure: {str(exc)}",
        )
