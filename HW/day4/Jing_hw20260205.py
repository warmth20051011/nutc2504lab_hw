import requests
import time
import base64
import uuid

from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from playwright.sync_api import sync_playwright

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# 基本設定
# ============================================================

SEARXNG_URL = "https://puli-8080.huannago.com/search"

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)


# ============================================================
# 工具：搜尋
# ============================================================

def search_searxng(query: str, time_range: str = None, limit: int = 1):
    print(f"🔍 正在搜尋: {query}")

    params = {
        "q": query,
        "format": "json",
        "language": "zh-TW"
    }

    if time_range:
        params["time_range"] = time_range

    try:
        res = requests.get(SEARXNG_URL, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return [r for r in data.get("results", []) if "url" in r][:limit]
    except Exception as e:
        print("❌ 搜尋失敗:", e)
        return []


# ============================================================
# State 定義
# ============================================================

class QAState(TypedDict):
    question: str
    query: Optional[str]
    url: Optional[str]
    title: Optional[str]
    answer: Optional[str]


# ============================================================
# Nodes
# ============================================================

def check_cache_node(state: QAState) -> QAState:
    # 目前沒有真的 cache，單純當入口
    return state


def route_from_cache(state: QAState) -> str:
    return "query_gen"


def query_gen(state: QAState) -> QAState:
    print("✏️ query_gen：產生搜尋關鍵字")
    return {
        **state,
        "query": state["question"]
    }


def route_from_query(state: QAState) -> str:
    return "search_tool"


def search_tool(state: QAState) -> QAState:
    print("🔍 search_tool：呼叫 SearXNG")

    results = search_searxng(state["query"], time_range="day", limit=1)
    if not results:
        return {**state, "answer": "找不到搜尋結果"}

    first = results[0]
    return {
        **state,
        "url": first["url"],
        "title": first.get("title", "搜尋結果")
    }


def vlm_read_website(url: str, title: str) -> str:
    print(f"[VLM] 啟動視覺閱讀: {url}")

    screenshots = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 1200}
            )

            context.route(
                "**/*",
                lambda route, req: route.abort()
                if req.resource_type in ["image", "font", "stylesheet", "media"]
                else route.continue_()
            )

            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(1500)

            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(1500)

            img = base64.b64encode(page.screenshot()).decode("utf-8")
            screenshots.append(img)

            browser.close()

    except Exception as e:
        return f"❌ 截圖失敗: {e}"

    print("[LLM] 正在分析截圖...")

    msgs = [
        HumanMessage(content=[
            {"type": "text", "text": f"這是網頁截圖，請整理與「{title}」相關的重點。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshots[0]}"}}
        ])
    ]

    return llm.invoke(msgs).content


def final_answer(state: QAState) -> QAState:
    print("📝 final_answer：VLM 閱讀網頁")

    if not state.get("url"):
        return state

    answer = vlm_read_website(state["url"], state["title"])
    return {**state, "answer": answer}


# ============================================================
# LangGraph 建立（✅ 重點改在這）
# ============================================================

workflow = StateGraph(QAState)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("query_gen", query_gen)
workflow.add_node("search_tool", search_tool)
workflow.add_node("final_answer", final_answer)

workflow.set_entry_point("check_cache")

workflow.add_conditional_edges(
    "check_cache",
    route_from_cache,
    {
        "query_gen": "query_gen",
    }
)

workflow.add_conditional_edges(
    "query_gen",
    route_from_query,
    {
        "search_tool": "search_tool",
    }
)

workflow.add_edge("search_tool", "final_answer")
workflow.add_edge("final_answer", END)

# ✅ 關鍵：加上 checkpointer，讓它能長期運行
checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer
)

print(app.get_graph().draw_ascii())


# ============================================================
# 主程式（✅ 正確 invoke）
# ============================================================

if __name__ == "__main__":
    while True:
        try:
            question = input("\n請輸入要查詢的問題（q 離開）：").strip()
            if not question:
                continue
            if question.lower() == "q":
                break

            thread_id = str(uuid.uuid4())

            result = app.invoke(
                {
                    "question": question,
                    "query": None,
                    "url": None,
                    "title": None,
                    "answer": None
                },
                config={
                    "configurable": {
                        "thread_id": thread_id
                    },
                    "recursion_limit": 20
                }
            )

            print("\n" + "=" * 40)
            print("📌 最終回答：")
            print(result.get("answer"))
            print("=" * 40)

        except KeyboardInterrupt:
            print("\n👋 中斷結束")
            break

