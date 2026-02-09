from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)


def answer_question(question: str, contexts: list[str]) -> str:
    prompt = f"""
請根據以下文件內容回答問題，不可編造、不使用文件外資訊。

文件內容：
{chr(10).join(contexts)}

問題：
{question}
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()

