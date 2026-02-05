import random
from typing import Annotated, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage, BaseMessage, ToolMessage
)
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode


llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)

@tool
def get_weather(city: str):
    """查詢指定城市的天氣"""
    if random.random() < 0.5:
        return "系統錯誤:天氣資料庫連線失敗"
    if "台北" in city:
        return "台北下大雨，18 度"
    if "台中" in city:
        return "台中晴天，26 度"
    if "高雄" in city:
        return "高雄多雲，30 度"
    return "資料庫沒有這個城市的資料"


tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)


def fallback_node(state: AgentState):
    last = state["messages"][-1]
    tool_call_id = last.tool_calls[0]["id"]

    return {
        "messages": [
            ToolMessage(
                content="系統警示:已達到最大重試次數，服務暫時無法使用",
                tool_call_id=tool_call_id,
            )
        ]
    }


def router(state: AgentState) -> Literal["tools", "fallback", "end"]:
    messages = state["messages"]
    last = messages[-1]

    if not last.tool_calls:
        return "end"

    retry_count = 0
    for msg in reversed(messages[:-1]):
        if isinstance(msg, ToolMessage):
            if "系統錯誤" in msg.content:
                retry_count += 1
            else:
                break
        else:
            break

    print(f"DEBUG: retry_count={retry_count}")

    if retry_count >= 5:
        return "fallback"

    return "tools"


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {"tools": "tools", "fallback": "fallback", "end": END},
)

workflow.add_edge("tools", "agent")
workflow.add_edge("fallback", "agent")

app = workflow.compile()
print(app.get_graph().draw_ascii())


if __name__ == "__main__":
    while True:
        user_input = input("user: ")
        if user_input in {"exit", "q"}:
            break

        for event in app.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            for node_name, output in event.items():
                msg = output["messages"][-1]

                if node_name == "agent":
                    if msg.tool_calls:
                        print("->[Agent] 呼叫工具中…")
                    else:
                        print(f"->[Agent] {msg.content}")

                elif node_name == "tools":
                    print("->[Tools] 工具執行")

                elif node_name == "fallback":
                    print("->[Fallback] 觸發熔斷")
        print()

