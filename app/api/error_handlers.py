"""글로벌 예외 핸들러. FastAPI 앱에 등록하여 일관된 에러 응답 형식을 보장한다."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


def _error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    """통일된 에러 응답 형식을 생성한다."""
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error_code": error_code,
            "message": message,
        },
    )


def _has_missing_field_error(errors: list[dict]) -> bool:
    """Pydantic 검증 에러 목록에 필수 파라미터 누락(missing)이 포함되어 있는지 확인한다."""
    return any(e.get("type") == "missing" for e in errors)


def _format_validation_message(errors: list[dict]) -> str:
    """Pydantic 검증 에러 목록에서 사용자 친화적 메시지를 생성한다."""
    details = []
    for e in errors:
        loc = " → ".join(str(loc_part) for loc_part in e.get("loc", []))
        msg = e.get("msg", "")
        details.append(f"{loc}: {msg}")
    return "; ".join(details)


async def _handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Pydantic 검증 에러를 처리한다.

    - 필수 파라미터 누락(missing) → 400 BAD_REQUEST
    - 타입 불일치/제약조건 위반 → 422 VALIDATION_ERROR
    """
    errors = exc.errors()
    message = _format_validation_message(errors)

    if _has_missing_field_error(errors):
        return _error_response(400, "BAD_REQUEST", message)

    return _error_response(422, "VALIDATION_ERROR", message)


async def _handle_value_error(
    request: Request, exc: ValueError
) -> JSONResponse:
    """ValueError를 처리한다.

    Pydantic v2의 ValidationError는 ValueError의 서브클래스이므로,
    ValidationError인 경우 RequestValidationError와 동일한 분기 로직을 적용한다.
    그 외 비즈니스 검증 ValueError는 400으로 처리한다.
    """
    if isinstance(exc, PydanticValidationError):
        errors = exc.errors()
        message = _format_validation_message(errors)
        if _has_missing_field_error(errors):
            return _error_response(400, "BAD_REQUEST", message)
        return _error_response(422, "VALIDATION_ERROR", message)

    return _error_response(400, "BAD_REQUEST", str(exc))


async def _handle_service_unavailable(
    request: Request, exc: ServiceUnavailableError
) -> JSONResponse:
    """외부 서비스 장애를 처리한다."""
    logger.error("외부 서비스 장애: %s", exc, exc_info=True)
    return _error_response(503, "SERVICE_UNAVAILABLE", str(exc))


async def _handle_unexpected_error(
    request: Request, exc: Exception
) -> JSONResponse:
    """예상치 못한 내부 오류를 처리한다."""
    logger.error("예상치 못한 내부 오류: %s", exc, exc_info=True)
    return _error_response(500, "INTERNAL_ERROR", "서버 내부 오류가 발생했습니다.")


def register_error_handlers(app: FastAPI) -> None:
    """FastAPI 앱에 글로벌 예외 핸들러를 등록한다."""
    app.add_exception_handler(RequestValidationError, _handle_request_validation_error)
    app.add_exception_handler(ValueError, _handle_value_error)
    app.add_exception_handler(ServiceUnavailableError, _handle_service_unavailable)
    app.add_exception_handler(Exception, _handle_unexpected_error)
