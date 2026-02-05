import requests
import json
import time
import os
import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from playwright.sync_api import sync_playwright


SEARXNG_URL = "https://puli-8080.huannago.com/search"


def search_searxng(query: str, time_range: str = None, limit: int = 3):
    print(f"🔍 正在搜尋: {query} (範圍: {time_range if time_range else '全部'})")

    params = {
        "q": query,
        "format": "json",
        "language": "zh-TW"
    }

    if time_range and time_range != "all":
        params["time_range"] = time_range

    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        return [r for r in results if "url" in r][:limit]

    except Exception as e:
        print(f"❌ 搜尋失敗: {e}")
        return []


llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)


def vlm_read_website(url: str, title: str = "網頁內容") -> str:
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

            for i in range(1):
                scroll_y = i * 1000
                page.evaluate(f"window.scrollTo(0, {scroll_y})")
                page.wait_for_timeout(1500)

                img = base64.b64encode(page.screenshot()).decode("utf-8")
                screenshots.append(img)

                print(f" - 截圖 {i+1} 完成 (Scroll: {scroll_y})")

            browser.close()

    except Exception as e:
        return f"❌ 截圖失敗: {e}"


    print(f"[LLM] 正在分析 {len(screenshots)} 張圖片...")

    per_image_results = []


    for idx, img in enumerate(screenshots):
        msg = [
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": f"這是一張網頁截圖，請擷取與「{title}」相關的重點資訊。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img}"
                    }
                }
            ])
        ]

        try:
            result = llm.invoke(msg).content
            per_image_results.append(result)
        except Exception as e:
            per_image_results.append(f"分析失敗: {e}")

    summary_msg = [
        HumanMessage(content=f"""
請根據以下多張網頁截圖的分析內容，整理一份完整的重點摘要：

{chr(10).join(per_image_results)}
""")
    ]

    try:
        return llm.invoke(summary_msg).content
    except Exception as e:
        return f"彙整失敗: {e}"


if __name__ == "__main__":
    question = input("請輸入要查詢的問題：")

    start_time = time.time()

    results = search_searxng(question, time_range="day", limit=1)
    if not results:
        print("找不到搜尋結果")
        exit()

    first = results[0]
    url = first["url"]
    title = first.get("title", "搜尋結果")

    print("✏️ query_gen：產生搜尋關鍵字")
    print("🔍 search_tool：呼叫 SearXNG")
    print("📝 final_answer：VLM 閱讀網頁")

    answer = vlm_read_website(url, title)

    print("\n" + "=" * 40)
    print("📌 最終回答：")
    print(answer)
    print(f"\n⏱️ 總耗時：{time.time() - start_time:.2f} 秒")

