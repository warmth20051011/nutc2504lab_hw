from typing import TypedDict, Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)

@tool
def get_weather(city: str):
    """查詢指定城市的天氣。輸入參数 city 必須是城市名稱。"""
    if "台北" in city:
        return "台北下大雨，氣溫 18 度" 
    elif "台中" in city:
        return "台中晴天，氣温 26 度"
    elif "高雄" in city:
        return "高雄多雲，氣溫 30 度"
    else:
        return "資料庫沒有這個城市的資料"
        
tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chatbot_node(state: AgentState):
    """思考節點:負責呼叫 LLM"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
    
tool_node_executor = ToolNode(tools)

def router(state: AgentState) -> Literal["tools","end"]:
    """路由邏輯:決定下一步是執行工具還是結束"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    else:
        return "end"
        
workflow = StateGraph(AgentState)

workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

app = workflow.compile()
print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    while True:
        user_input = input("user: ")
        if user_input.lower() in ["exit", "q"]:
            print("Bye!")
            break

        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for node_name, node_output in event.items():
                last_msg = node_output["messages"][-1]
                if node_name == "agent":
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"[AI 呼叫工具]: {last_msg.tool_calls}")
                    else:
                        print(f"[AI]: {last_msg.content}")
                elif node_name == "tools":
                    print(f"[工具回傳]: {last_msg.content}")
        print()
