import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.api import health, image, news, video
from backend.config import API_VERSION, APP_NAME, FRONTEND_DIR
from backend.schemas import ErrorResponse
from backend.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title=APP_NAME, version=API_VERSION)
app.include_router(health.router)
app.include_router(news.router)
app.include_router(image.router)
app.include_router(video.router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.middleware("http")
async def add_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error request_id=%s", request_id)
        response = JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected server error occurred.",
                request_id=request_id,
            ).model_dump(),
        )
    response.headers["X-Request-ID"] = request_id
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = _request_id(request)
    logger.warning("HTTP error status=%s request_id=%s", exc.status_code, request_id)
    message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="request_error", message=message, request_id=request_id).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _request_id(request)
    logger.warning("Validation error request_id=%s errors=%s", request_id, exc.errors())
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            message="Request input is invalid.",
            request_id=request_id,
        ).model_dump(),
    )


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
