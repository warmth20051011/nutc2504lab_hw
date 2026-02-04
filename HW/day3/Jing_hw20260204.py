import time
import requests
from pathlib import Path
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ================== 基本設定 ==================
BASE = "https://3090api.huannago.com"
CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
AUTH = ("nutc2504", "nutc2504")

AUDIO_PATH = "./Podcast_EP14_30s.wav"

OUT_DIR = Path("./out")
OUT_DIR.mkdir(exist_ok=True)

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)


# ================== State ==================
class MeetingState(TypedDict):
    srt: str
    timeline: str
    summary: str
    final: str


# ================== ASR 工具 ==================
def wait_download(url: str, max_tries=600):
    for _ in range(max_tries):
        try:
            r = requests.get(url, timeout=(5, 60), auth=AUTH)
            if r.status_code == 200:
                return r.text
        except requests.exceptions.ReadTimeout:
            pass
        time.sleep(2)
    raise TimeoutError("ASR timeout")


# ================== Node 1：ASR ==================
def asr_node(state: MeetingState):
    print("\n🎧 [ASR] 上傳音檔")

    with open(AUDIO_PATH, "rb") as f:
        r = requests.post(CREATE_URL, files={"audio": f}, auth=AUTH)
    r.raise_for_status()

    task_id = r.json()["id"]
    print(f"🆔 task_id = {task_id}")

    srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"
    srt_text = wait_download(srt_url)

    return {"srt": srt_text}


# ================== Node 2：Timeline ==================
def timeline_node(state: MeetingState):
    print("\n🕒 [Timeline] 產生含時間軸逐字稿")

    prompt = f"""
請將以下 SRT 內容整理成【時間軸逐字稿】：
- 保留所有時間碼
- 不要摘要
- 純文字、依時間順序

{state['srt']}
"""

    timeline = llm.invoke(prompt).content
    return {"timeline": timeline}


# ================== Node 3：Summary ==================
def summary_node(state: MeetingState):
    print("\n📌 [Summary] 產生重點摘要")

    prompt = f"""
請根據以下內容產生【重點摘要】：
- 主題
- 核心重點（條列）
- 結論

{state['srt']}
"""

    summary = llm.invoke(prompt).content
    return {"summary": summary}


# ================== Node 4：Writer ==================
def writer_node(state: MeetingState):
    print("\n🧩 [Writer] 輸出結果")

    timeline_path = OUT_DIR / "timeline.txt"
    summary_path = OUT_DIR / "summary.txt"

    timeline_path.write_text(state["timeline"], encoding="utf-8")
    summary_path.write_text(state["summary"], encoding="utf-8")

    print("\n=====【重點摘要】=====\n")
    print(state["summary"])

    print("\n=====【詳細逐字稿（含時間軸）】=====\n")
    print(state["timeline"])

    print(f"\n✅ 已輸出檔案：")
    print(f" - {timeline_path}")
    print(f" - {summary_path}")

    return {"final": "done"}

# ================== LangGraph ==================
graph = StateGraph(MeetingState)

graph.add_node("asr", asr_node)
graph.add_node("timeline", timeline_node)
graph.add_node("summary", summary_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("asr")

# 👇 關鍵結構（跟圖片一樣）
graph.add_edge("asr", "timeline")
graph.add_edge("asr", "summary")
graph.add_edge("timeline", "writer")
graph.add_edge("summary", "writer")
graph.add_edge("writer", END)

app = graph.compile()


# ================== Graph 結構顯示 ==================
print("\n📐 LangGraph 結構：")
try:
    print(app.get_graph().draw_ascii())
except ImportError:
    print("""
        __start__
            |
           asr
          /   \\
     timeline summary
          \\   /
          writer
            |
          __end__
    """)


# ================== 執行 ==================
result = app.invoke({
    "srt": "",
    "timeline": "",
    "summary": "",
    "final": ""
})

print("\n🎉 任務完成")

