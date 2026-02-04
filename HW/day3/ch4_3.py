from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool


llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="vllm-token",
    model="Qwen/Qwen3-VL-8B-Instruct"
)


@tool
def generate_tech_summary(article_content: str) -> str:
    """將科技文章內容歸納出 3 個重點摘要"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個專業的科技文章編輯，請將使用者提供的文章內容，歸納出 3 個重點，並以繁體中文條列式輸出。"),
        ("human", "{text}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": article_content})



llm_with_tools = llm.bind_tools([generate_tech_summary])

router_prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    chain = router_prompt | llm_with_tools
    ai_msg = chain.invoke({"input": user_input})

    if ai_msg.tool_calls:
        print("✅ [決策] 判斷為科技文章")
        tool_args = ai_msg.tool_calls[0]['args']
        final_result = generate_tech_summary.invoke(tool_args)
        print(f"📝 [執行結果]:\n{final_result}")
    else:
        print("❌ [決策] 判斷為閒聊/非科技文章，直接回覆。")
        print(f"💬 [AI 回應]: {ai_msg.content}")

