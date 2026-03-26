"""글로벌 예외 핸들러 테스트. 입력 검증 에러 코드 정합성 + 서비스 장애 + 내부 오류."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.api.error_handlers import register_error_handlers
from app.exceptions import ServiceUnavailableError


# ── 테스트용 Pydantic 모델 (모듈 레벨) ──


class SampleRequest(BaseModel):
    name: str = Field(..., min_length=1)
    age: int


# ── 테스트용 FastAPI 앱 (모듈 레벨 싱글턴) ──

_app = FastAPI()
register_error_handlers(_app)


@_app.post("/test/validation")
async def validation_endpoint(req: SampleRequest):
    return {"ok": True}


@_app.post("/test/value-error")
async def value_error_endpoint():
    raise ValueError("비즈니스 검증 실패")


@_app.post("/test/service-unavailable")
async def service_unavailable_endpoint():
    raise ServiceUnavailableError(service="Bedrock", detail="timeout")


@_app.post("/test/internal-error")
async def internal_error_endpoint():
    raise RuntimeError("예상치 못한 오류")


@pytest.fixture
def client():
    return TestClient(_app, raise_server_exceptions=False)


# ── Property 9: 입력 검증 에러 코드 정합성 ──
# Validates: Requirements 7.1, 7.2


class TestInputValidationErrorCodes:
    """Property 9: 필수 파라미터 누락 시 400, 타입 불일치 시 422 반환 검증."""

    def test_missing_required_field_returns_400(self, client):
        """필수 파라미터 누락 → 400 BAD_REQUEST."""
        response = client.post("/test/validation", json={})
        assert response.status_code == 400

        body = response.json()
        assert body["status"] == "error"
        assert body["error_code"] == "BAD_REQUEST"
        assert "message" in body

    def test_missing_single_field_returns_400(self, client):
        """일부 필수 파라미터만 누락 → 400 BAD_REQUEST."""
        response = client.post("/test/validation", json={"name": "홍길동"})
        assert response.status_code == 400

        body = response.json()
        assert body["error_code"] == "BAD_REQUEST"

    def test_type_mismatch_returns_422(self, client):
        """타입 불일치 → 422 VALIDATION_ERROR."""
        response = client.post(
            "/test/validation", json={"name": "홍길동", "age": "스물"}
        )
        assert response.status_code == 422

        body = response.json()
        assert body["status"] == "error"
        assert body["error_code"] == "VALIDATION_ERROR"
        assert "message" in body

    def test_constraint_violation_returns_422(self, client):
        """제약조건 위반 (min_length) → 422 VALIDATION_ERROR."""
        response = client.post("/test/validation", json={"name": "", "age": 20})
        assert response.status_code == 422

        body = response.json()
        assert body["error_code"] == "VALIDATION_ERROR"

    def test_missing_and_type_error_mixed_returns_400(self, client):
        """누락 + 타입 불일치 동시 발생 시 → 400 (missing 우선)."""
        response = client.post("/test/validation", json={"age": "스물"})
        assert response.status_code == 400

        body = response.json()
        assert body["error_code"] == "BAD_REQUEST"

    def test_valid_request_returns_200(self, client):
        """정상 요청 → 200."""
        response = client.post(
            "/test/validation", json={"name": "홍길동", "age": 20}
        )
        assert response.status_code == 200


# ── ValueError 핸들러 테스트 ──


class TestValueErrorHandler:
    """Pydantic 바깥의 비즈니스 ValueError → 400 BAD_REQUEST."""

    def test_value_error_returns_400(self, client):
        response = client.post("/test/value-error")
        assert response.status_code == 400

        body = response.json()
        assert body["status"] == "error"
        assert body["error_code"] == "BAD_REQUEST"
        assert "비즈니스 검증 실패" in body["message"]


# ── ServiceUnavailableError 핸들러 테스트 ──


class TestServiceUnavailableHandler:
    """외부 서비스 장애 → 503 SERVICE_UNAVAILABLE."""

    def test_service_unavailable_returns_503(self, client):
        response = client.post("/test/service-unavailable")
        assert response.status_code == 503

        body = response.json()
        assert body["status"] == "error"
        assert body["error_code"] == "SERVICE_UNAVAILABLE"
        assert "Bedrock" in body["message"]


# ── Exception 핸들러 테스트 ──


class TestUnexpectedErrorHandler:
    """예상치 못한 내부 오류 → 500 INTERNAL_ERROR."""

    def test_unexpected_error_returns_500(self, client):
        response = client.post("/test/internal-error")
        assert response.status_code == 500

        body = response.json()
        assert body["status"] == "error"
        assert body["error_code"] == "INTERNAL_ERROR"
        assert "서버 내부 오류" in body["message"]

    def test_unexpected_error_hides_detail(self, client):
        """내부 오류 상세는 응답에 노출되지 않는다."""
        response = client.post("/test/internal-error")
        body = response.json()
        assert "예상치 못한 오류" not in body["message"]


# ── 응답 형식 일관성 ──


class TestErrorResponseFormat:
    """모든 에러 응답이 통일된 형식을 갖추는지 검증."""

    @pytest.mark.parametrize(
        "path,payload,expected_keys",
        [
            ("/test/validation", {}, {"status", "error_code", "message"}),
            ("/test/validation", {"name": "a", "age": "x"}, {"status", "error_code", "message"}),
            ("/test/value-error", None, {"status", "error_code", "message"}),
            ("/test/service-unavailable", None, {"status", "error_code", "message"}),
            ("/test/internal-error", None, {"status", "error_code", "message"}),
        ],
    )
    def test_error_response_has_consistent_format(
        self, client, path, payload, expected_keys
    ):
        if payload is not None:
            response = client.post(path, json=payload)
        else:
            response = client.post(path)

        body = response.json()
        assert set(body.keys()) == expected_keys
