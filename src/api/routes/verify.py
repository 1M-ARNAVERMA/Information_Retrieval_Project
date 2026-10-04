"""Fact verification endpoint performing end-to-end inference."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.routes.health import get_service
from src.api.schemas import VerifyRequest, VerifyResponse
from src.api.service import APIService
from src.common.exceptions import FeverXException

router = APIRouter(tags=["Verification"])


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="End-to-End Fact Verification",
    description="Analyzes input claim, retrieves relevant Wikipedia evidence sentences, runs neural reranking, performs NLI inference, constructs the evidence graph, and delivers an explainable verdict.",
)
def verify_claim(
    request: VerifyRequest,
    service: APIService = Depends(get_service),
) -> VerifyResponse:
    """Verifies a claim against indexed Wikipedia corpus."""
    try:
        return service.verify(request)
    except FeverXException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Verification failed: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal verification error: {str(exc)}",
        )
