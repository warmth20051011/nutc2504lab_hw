from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0.1
)


def rewrite_query(question: str) -> str:
    prompt = f"""
請將以下問題改寫成更適合「文件檢索」的搜尋查詢：

問題：
{question}

只輸出改寫後的查詢句。
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()

