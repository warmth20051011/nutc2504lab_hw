import json
from typing import Annotated, TypedDict 
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END, add_messages  
from langgraph.prebuilt import ToolNode

llm =ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)

@tool
def extract_order_data(name: str, phone:str, product:str, quantity: int, address: str):
    """
    資料提取專用工具。
    專門用於從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。
    """
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
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode([extract_order_data])

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)  
workflow.add_node("tools", tool_node)   
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

print(app.get_graph().draw_ascii())


if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "q"]:
            break

        for event in app.stream({
            "messages": [HumanMessage(content=user_input)]
        }):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                print(value["messages"][-1].content or value["messages"][-1].tool_calls)
            
