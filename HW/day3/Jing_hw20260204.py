import time
import requests
from typing import TypedDict
from pathlib import Path


from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


BASE = "https://3090api.huannago.com"
CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
AUTH = ("nutc2504", "nutc2504")

WAV_PATH = "./Podcast_EP14_30s.wav"

OUT_DIR = Path("./out")
OUT_DIR.mkdir(exist_ok=True)

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)


class MeetingState(TypedDict):
    audio_path: str
    transcript: str
    minutes: str
    summary: str
    final_report: str

def wait_download(url: str, max_tries=300):
    for _ in range(max_tries):
        r = requests.get(url, timeout=(5, 60), auth=AUTH)
        if r.status_code == 200:
            return r.text
        time.sleep(2)
    raise TimeoutError("ASR timeout")


def asr_node(state: MeetingState):
    print("\n🎧 [ASR] 上傳音檔，建立任務")
    with open(state["audio_path"], "rb") as f:
        r = requests.post(CREATE_URL, files={"audio": f}, auth=AUTH)
    r.raise_for_status()
    task_id = r.json()["id"]

    print(f"🆔 [ASR] task_id = {task_id}")
    print("⏳ [ASR] 等待轉錄完成...")

    srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"
    transcript = wait_download(srt_url)

    print("✅ [ASR] 完成")
    return {"transcript": transcript}


def minutes_node(state: MeetingState):
    print("\n📝 [Minutes] 整理詳細逐字稿")
    minutes = llm.invoke(f"""
請將以下逐字稿整理成【詳細逐字會議紀錄】：
- 依時間順序
- 保留時間軸
- 使用 Markdown 表格

{state['transcript']}
""").content
    print("✅ [Minutes] 完成")
    return {"minutes": minutes}



def summary_node(state: MeetingState):
    print("\n📌 [Summary] 產出重點摘要")
    summary = llm.invoke(f"""
請將以下會議內容整理成【重點摘要】：
- 會議主題
- 核心重點
- 結論
- Action Items

{state['transcript']}
""").content
    print("✅ [Summary] 完成")
    return {"summary": summary}


def join_node(state: MeetingState):
    return {}


def writer_node(state: MeetingState):
    print("\n🧩 [Writer] 整合最終輸出")
    final = f"""
# 📋 智慧會議記錄

## 一、重點摘要
{state['summary']}

---

## 二、詳細逐字會議紀錄
{state['minutes']}
"""
    return {"final_report": final}



graph = StateGraph(MeetingState)

graph.add_node("asr", asr_node)
graph.add_node("minutes_taker", minutes_node)
graph.add_node("summarizer", summary_node)
graph.add_node("join", join_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("asr")

graph.add_edge("asr", "minutes_taker")
graph.add_edge("asr", "summarizer")
graph.add_edge("minutes_taker", "join")
graph.add_edge("summarizer", "join")
graph.add_edge("join", "writer")
graph.add_edge("writer", END)

app = graph.compile()



print("\n📊 LangGraph 結構：")
try:
    print(app.get_graph().draw_ascii())
except ImportError:
    print("""
        __start__
            |
           asr
          /   \\
 minutes_taker  summarizer
          \\   /
          writer
            |
          __end__
    """)




result = app.invoke({
    "audio_path": WAV_PATH,
    "transcript": "",
    "minutes": "",
    "summary": "",
    "final_report": ""
})

(Path("./out/transcript.srt")).write_text(result["transcript"], encoding="utf-8")
(Path("./out/minutes.md")).write_text(result["minutes"], encoding="utf-8")
(Path("./out/summary.md")).write_text(result["summary"], encoding="utf-8")
(Path("./out/final_report.md")).write_text(result["final_report"], encoding="utf-8")



print("\n🎉 任務完成！輸出如下：\n")

print("=====【重點摘要】=====\n")
print(result["summary"])

print("\n=====【詳細逐字稿（完整）】=====\n")
print(result["transcript"])

