import json
from typing import Annotated, TypedDict, Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode


llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)

VIP_LIST = ["AI哥", "一龍馬"]


@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """資料提取專用工具"""
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }


llm_with_tools = llm.bind_tools([extract_order_data])


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode([extract_order_data])


def entry_router(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def post_tool_router(state: AgentState) -> Literal["human_review", "agent"]:
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, ToolMessage):
        try:
            data = json.loads(last_message.content)
            user_name = data.get("name", "")

            if user_name in VIP_LIST:
                print(f"DEBUG: 發現 VIP [{user_name}] → 轉向人工審核")
                return "human_review"

        except Exception as e:
            print(f"JSON 解析錯誤: {e}")

    return "agent"


def human_review_node(state: AgentState):
    print("\n" + "=" * 30)
    print("觸發人工審核機制：檢測到 VIP 客戶！")
    print("=" * 30)

    last_msg = state["messages"][-1]
    print(f"待審核資料: {last_msg.content}")

    review = input(">>> 管理員請批示 (輸入 ok 通過，其它拒絕): ")

    if review.lower() == "ok":
        return {
            "messages": [
                AIMessage(content="已收到訂單資料，因偵測到 VIP 客戶，系統將轉交人工審核。"),
                HumanMessage(content="[系統公告] 管理員已通過此 VIP 訂單，請繼續後續流程。")
            ]
        }
    else:
        return {
            "messages": [
                AIMessage(content="已收到訂單資料，等待人工審核結果。"),
                HumanMessage(content="[系統公告] 管理員拒絕了此訂單，請取消交易並告知用戶。")
            ]
        }


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("human_review", human_review_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    entry_router,
    {"tools": "tools", END: END}
)

workflow.add_conditional_edges(
    "tools",
    post_tool_router,
    {
        "human_review": "human_review",
        "agent": "agent"
    }
)

workflow.add_edge("human_review", "agent")

app = workflow.compile()

print(app.get_graph().draw_ascii())


if __name__ == "__main__":
    print(f"VIP 名單: {VIP_LIST}")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "q"]:
            break

        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][-1]
                    if not msg.tool_calls:
                        print(f"-> [Agent]: {msg.content}")
                elif key == "human_review":
                    print("-> [Human Review]: 審核完成")

