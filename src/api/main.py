"""FastAPI application entrypoint for FEVER-X fact verification microservice."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import evaluate, explain, health, retrieve, verify
from src.api.service import APIService
from src.common.exceptions import FeverXException
from src.utils.logger import get_logger

logger = get_logger("fastapi_app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup warm-up and shutdown."""
    logger.info("Initializing FEVER-X backend service and warming models...")
    try:
        service = APIService.get_instance()
        logger.info(
            "Service ready with %d indexed documents.",
            len(service.pipeline.documents),
        )
    except Exception as exc:
        logger.error("Failed to initialize pipeline during startup: %s", exc)
    yield
    logger.info("Shutting down FEVER-X service.")


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI instance."""
    app = FastAPI(
        title="FEVER-X Fact Verification API",
        version="1.0.0",
        description=(
            "Explainable Multi-Stage Fact Verification using Entity-Aware Neural Retrieval "
            "and Knowledge Graph Reasoning on the FEVER Benchmark."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for frontend applications (e.g., Streamlit, React, or browser clients)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom Exception Handlers
    @app.exception_handler(FeverXException)
    async def feverx_exception_handler(request: Request, exc: FeverXException) -> JSONResponse:
        logger.warning("FeverX domain exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": exc.__class__.__name__, "message": str(exc)},
        )

    # Include all modular route controllers
    app.include_router(health.router)
    app.include_router(verify.router)
    app.include_router(retrieve.router)
    app.include_router(explain.router)
    app.include_router(evaluate.router)

    @app.get("/", tags=["System"], summary="API Root & Info")
    def root():
        return {
            "name": "FEVER-X Fact Verification API",
            "version": "1.0.0",
            "status": "online",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
