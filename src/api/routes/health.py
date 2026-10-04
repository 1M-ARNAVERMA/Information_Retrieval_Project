"""Health check and system diagnostics endpoint."""

from fastapi import APIRouter, Depends

from src.api.schemas import HealthResponse
from src.api.service import APIService

router = APIRouter(tags=["System"])


def get_service() -> APIService:
    return APIService.get_instance()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Indices Status",
    description="Returns backend status, loaded retrieval indices, document counts, and compute device.",
)
def check_health(service: APIService = Depends(get_service)) -> HealthResponse:
    """Returns application health and model index readiness."""
    return service.get_health()
