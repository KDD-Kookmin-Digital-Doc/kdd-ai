"""커스텀 예외 클래스 정의."""


class ServiceUnavailableError(Exception):
    """외부 서비스(Bedrock, Supabase) 연결 실패 시 발생하는 예외."""

    def __init__(self, service: str, detail: str = ""):
        self.service = service
        self.detail = detail
        super().__init__(
            f"{service} 서비스 연결 실패: {detail}" if detail else f"{service} 서비스 연결 실패"
        )
