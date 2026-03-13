from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import GOOGLE_API_KEY
#제미나이(Gemini) API를 호출하고 스트리밍을 준비
# 최신 2.5 Flash 모델 사용!
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 국민대학교 학사규정을 안내하는 AI 어시스턴트입니다. "
            "반드시 아래 제공된 [관련 규정]만을 바탕으로 친절하게 답변하세요. "
            "규정에 없는 내용이면 '제공된 문서에서 확인할 수 없습니다'라고 답변하세요.\n\n"
            "[관련 규정]\n{context}"),
    ("human", "{question}")
])

rag_chain = rag_prompt | llm