"""Evaluation and ablation benchmarks endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.routes.health import get_service
from src.api.schemas import AblationResponse
from src.api.service import APIService

router = APIRouter(tags=["Evaluation"])


@router.get(
    "/evaluate/ablation",
    response_model=AblationResponse,
    summary="Ablation Study Benchmarks",
    description="Retrieves the 9-stage architectural ablation evaluation comparison (Precision@10, Recall@10, MAP@10, nDCG@10).",
)
def get_ablation_benchmarks(
    service: APIService = Depends(get_service),
) -> AblationResponse:
    """Returns comparative metrics across all 9 FEVER-X ablation configurations."""
    try:
        return service.get_ablation_results()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch ablation study results: {str(exc)}",
        )
