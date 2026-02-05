from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)


class state(TypedDict):
    original_text: str
    translated_text: str
    critique: str
    attempts: int


def translator_node(state: state):
    """負責翻譯的節點"""
    print(f"\n--- 翻譯嘗試(第 {state['attempts'] + 1} 次)---")

    prompt = (
        f"你是一名翻譯員,請將以下中文翻譯成英文,不須任何解釋:"
        f"'{state['original_text']}'"
    )

    if state["critique"]:
        prompt += f"\n\n上一輪的審査意見是:{state['critique']}。請根據意見修正翻譯。"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "translated_text": response.content,
        "attempts": state["attempts"] + 1
    }


def reflector_node(state: state):
    """負責審查的節點(Critique)"""
    print("--- 審查中 (Reflection) ---")
    print(f"翻譯: {state['translated_text']}")

    prompt = f"""
你是一個嚴格的翻譯審查員。
原文: {state['original_text']}
翻譯: {state['translated_text']}

請檢查翻譯是否準確且通順。
如果翻譯很完美，請只回覆 "PASS"。
如果需要修改，請給出簡短的具體建議。
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"critique": response.content}


def should_continue(state: state) -> Literal["translator", "end"]:
    critique = state["critique"].strip().upper()

    if "PASS" in critique:
        print("--- 審查通過! ---")
        return "end"

    elif state["attempts"] >= 3:
        print("--- 達到最大重試次數，強制結束 ---")
        return "end"

    else:
        print(f"--- 審查未通過: {state['critique']} ---")
        print("--- 退回重寫 ---")
        return "translator"


workflow = StateGraph(state)

workflow.add_node("translator", translator_node)
workflow.add_node("reflector", reflector_node)

workflow.set_entry_point("translator")
workflow.add_edge("translator", "reflector")

workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "translator": "translator",
        "end": END
    }
)

app = workflow.compile()

print(app.get_graph().draw_ascii())


if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "q"]:
            break

        inputs = {
            "original_text": user_input,
            "attempts": 0,
            "critique": ""
        }

        result = app.invoke(inputs)

        print("\n========== 最終結果 ==========")
        print(f"原文: {result['original_text']}")
        print(f"最終翻譯: {result['translated_text']}")
        print(f"最終次數: {result['attempts']}")

