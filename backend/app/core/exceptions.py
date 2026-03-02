from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("app.error")


class AppError(Exception):
    pass


class InvalidImageError(AppError):
    pass


class InferenceError(AppError):
    pass


def _build_error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    error: Exception,
    include_detail: bool,
) -> JSONResponse:
    request_id = None
    payload = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if include_detail:
        payload["error"]["detail"] = str(error)

    logger.exception(
        "request failed: %s",
        code,
        exc_info=error,
        extra={
            "extras": {
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "error_code": code,
                "status_code": status_code,
            }
        },
    )
    return JSONResponse(status_code=status_code, content=payload)


def register_exception_handlers(app: FastAPI) -> None:
    def include_detail(request: Request) -> bool:
        settings = request.app.state.container.settings
        return bool(settings.debug or settings.expose_error_details)

    @app.exception_handler(InvalidImageError)
    async def invalid_image_handler(request: Request, exc: InvalidImageError) -> JSONResponse:
        return _build_error_response(
            request,
            status_code=400,
            code="INVALID_IMAGE",
            message="Image could not be decoded or validated.",
            error=exc,
            include_detail=include_detail(request),
        )

    @app.exception_handler(InferenceError)
    async def inference_handler(request: Request, exc: InferenceError) -> JSONResponse:
        return _build_error_response(
            request,
            status_code=500,
            code="INFERENCE_ERROR",
            message="Inference failed while processing the request.",
            error=exc,
            include_detail=include_detail(request),
        )

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return _build_error_response(
            request,
            status_code=500,
            code="APP_ERROR",
            message="Application error occurred.",
            error=exc,
            include_detail=include_detail(request),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return _build_error_response(
            request,
            status_code=500,
            code="UNHANDLED_EXCEPTION",
            message="Unexpected server error.",
            error=exc,
            include_detail=include_detail(request),
        )
