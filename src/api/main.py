"""FastAPI application for the local restaurant assistant."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.orchestration import APIBackend
from src.api.routes import router
from src.core.config import config

logger = logging.getLogger(__name__)


def create_app(backend: APIBackend | None = None) -> FastAPI:
    app = FastAPI(title=config.project_name)
    app.state.backend = backend if backend is not None else APIBackend(
        db_path=config.db_path, max_upload_bytes=config.api_max_upload_bytes
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Transcript", "X-Response-Text", "X-Session-ID"],
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "Invalid request parameters"})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled API error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(router)
    return app


app = create_app()
