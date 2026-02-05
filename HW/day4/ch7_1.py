import json
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0.7
)

CACHE_FILE = "translation_cache.json"


# ===== State =====
class state(TypedDict):
    original_text: str
    translated_text: str
    critique: str
    attempts: int
    is_cache_hit: bool


# ===== Cache Utils =====
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def save_cache(original: str, translated: str):
    data = load_cache()
    data[original] = translated
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ===== Nodes =====
def check_cache_node(state: state):
    print("\n--- 檢查快取 (Check Cache) ---")
    data = load_cache()
    original = state["original_text"]

    if original in data:
        print("命中快取！直接回傳結果。")
        return {
            "translated_text": data[original],
            "is_cache_hit": True
        }
    else:
        print("未命中快取，準備翻譯流程…")
        return {"is_cache_hit": False}


def translator_node(state: state):
    print(f"\n--- 翻譯嘗試 (第 {state['attempts'] + 1} 次) ---")

    prompt = (
        f"你是一名翻譯員，請將以下中文翻譯成英文，不須任何解釋："
        f"'{state['original_text']}'"
    )

    if state["critique"]:
        prompt += f"\n\n上一輪審查意見：{state['critique']}，請修正翻譯。"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "translated_text": response.content,
        "attempts": state["attempts"] + 1
    }


def reflector_node(state: state):
    print("--- 審查中 (Reflection) ---")
    prompt = (
        f"原文: {state['original_text']}\n"
        f"翻譯: {state['translated_text']}\n"
        f"請檢查翻譯是否正確，如正確請回 PASS，否則提出修改建議。"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"critique": response.content}


# ===== Routers =====
def cache_router(state: state) -> Literal["end", "translator"]:
    if state["is_cache_hit"]:
        return "end"
    return "translator"


def should_continue(state: state) -> Literal["translator", "end"]:
    critique = state["critique"].strip().upper()

    if "PASS" in critique:
        print("--- 審查通過 ---")
        return "end"
    elif state["attempts"] >= 3:
        print("--- 達到最大重試次數 ---")
        return "end"
    else:
        print(f"--- 審查未通過：{state['critique']} ---")
        return "translator"


# ===== Workflow =====
workflow = StateGraph(state)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("translator", translator_node)
workflow.add_node("reflector", reflector_node)

workflow.set_entry_point("check_cache")

workflow.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "end": END,
        "translator": "translator"
    }
)

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


# ===== Main =====
if __name__ == "__main__":
    print(f"快取檔案：{CACHE_FILE}")
    while True:
        user_input = input("\n請輸入要翻譯的中文 (exit/q 離開): ")
        if user_input.lower() in ["exit", "q"]:
            break

        inputs = {
            "original_text": user_input,
            "translated_text": "",
            "critique": "",
            "attempts": 0,
            "is_cache_hit": False
        }

        result = app.invoke(inputs)

        if not result["is_cache_hit"]:
            save_cache(result["original_text"], result["translated_text"])
            print("(已寫入快取)")

        print("\n========== 最終結果 ==========")
        print(f"原文: {result['original_text']}")
        print(f"翻譯: {result['translated_text']}")
        print(f"來源: {'快取(Cache)' if result['is_cache_hit'] else '生成(LLM)'}")

