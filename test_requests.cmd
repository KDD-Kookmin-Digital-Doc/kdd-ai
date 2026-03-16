@echo off
chcp 65001 >nul
echo ============================================
echo  국민대학교 RAG 챗봇 API 테스트
echo ============================================
echo.

echo [테스트 1] 인사말 (LLM 호출 없이 즉시 응답)
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"안녕\"}"
echo.
echo.

echo [테스트 2] 싱글턴 학사규정 질의
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"휴학 규정 알려줘\"}"
echo.
echo.

echo [테스트 3] 시맨틱 캐시 테스트 (유사 질문)
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"휴학규정 알려줘\"}"
echo.
echo.

echo [테스트 4] 시맨틱 캐시 테스트 (다른 표현)
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"휴학 규정을 알고싶어\"}"
echo.
echo.

echo [테스트 5] 멀티턴 대화 - 첫 번째 질문
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"휴학 규정 알려줘\", \"session_id\": \"test-session-1\"}"
echo.
echo.

echo [테스트 6] 멀티턴 대화 - 후속 질문 (이전 맥락 참조)
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"그러면 복학은 어떻게 해?\", \"session_id\": \"test-session-1\"}"
echo.
echo.

echo [테스트 7] 멀티턴 대화 - 추가 후속 질문
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"기간 제한이 있어?\", \"session_id\": \"test-session-1\"}"
echo.
echo.

echo [테스트 8] 다른 세션 (독립된 대화)
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"졸업 요건이 뭐야?\", \"session_id\": \"test-session-2\"}"
echo.
echo.

echo [테스트 9] 규정에 없는 질문
echo -------------------------------------------
curl -s -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"학교 근처 맛집 추천해줘\"}"
echo.
echo.

echo ============================================
echo  테스트 완료
echo ============================================
pause
