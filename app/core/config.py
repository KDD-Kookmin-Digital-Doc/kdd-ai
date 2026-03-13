import os
#API 키나 DB 주소 같은 '환경 변수'들을 한 곳에서 관리하는 파일
# 나중에는 .env 파일과 python-dotenv 라이브러리를 써서 숨길 예정
# 현재는 프로토타입 구동을 위해 명시적으로 세팅
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING", "postgresql+psycopg2://user:password@localhost:5432/rag_db")
COLLECTION_NAME = "kookmin_rules"

# FAISS 임계값 세팅 (튜닝용)
CACHE_THRESHOLD = 0.35

# 멀티턴 대화 설정
MAX_TURNS_BEFORE_SUMMARY = 5  # 이 턴 수 이상이면 대화 요약 실행