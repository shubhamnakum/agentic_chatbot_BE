from langgraph.graph import StateGraph, END
from typing import Any, Dict, TypedDict, List
from agents import classifier, retriever, researcher, weather_agent, writer, critic, AgentState


def setup_graph() -> Any:
    """Sets up the fully automated agent workflow graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", classifier)
    workflow.add_node("retrieval", retriever)
    workflow.add_node("researcher", researcher)
    workflow.add_node("weather", weather_agent)
    workflow.add_node("writer", writer)
    workflow.add_node("critic", critic)

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["category"],
        {
            "retrieval": "retrieval",
            "researcher": "researcher",
            "weather": "weather",
        },
    )
    workflow.add_edge("retrieval", "writer")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("weather", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", END)

    workflow.set_entry_point("supervisor")
    return workflow.compile()


def process_message(state: Dict[str, Any], message: str, domain: List[str]) -> Dict[str, Any]:
    """Process a user message through the agent system."""
    app = setup_graph()
    agent_state: AgentState = {
        "messages": state.get("messages", []),
        "current_input": message,
        "retrieved_docs": "",
        "draft_response": "",
        "final_response": "",
        "retrieval_successful": False,
        "category": "",
        "source": "",
        "city": "",
        "domain": domain,
        "cleaned_docs": [],
    }
    final_state = app.invoke(agent_state)
    return final_state
