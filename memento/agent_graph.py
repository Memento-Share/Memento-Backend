from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from typing import List, Optional, Literal, Any, Dict, TypedDict

class MementoState(MessagesState):
    request_stage : Literal["image", "audio"]


def photo_node(state: MementoState) -> Dict[Literal["messages"], Any]:
    print("Processing image...")
    return {"messages": [AIMessage(content="Processed Image")]}
def audio_node(state: MementoState) -> Dict[Literal["messages"], Any]:
    print("Processing audio...")
    return {"messages": [AIMessage(content="Processed Audio")]}
def text_node(state: MementoState) -> Dict[Literal["messages"], Any]:
    print("Processing text...")
    return {"messages": [AIMessage(content="Processed Text")]}


def router(state): 
    if state["request_stage"] == "image":
        return "photo_node"
    elif state["request_stage"] == "audio":
        return "audio_node"


graph = StateGraph(MementoState)
graph.add_node("photo_node", photo_node)
graph.add_node("audio_node", audio_node)
graph.add_node("text_node", text_node)

graph.add_conditional_edges(START, router)
graph.add_edge("photo_node", END)
graph.add_edge("audio_node", "text_node")
graph.add_edge("text_node", END)
agent = graph.compile()
def test_graph():
    state : MementoState = {"messages" : [],"request_stage": "image"}
    result = agent.invoke(state)
    print(result)

test_graph()