from langgraph.graph import StateGraph, END
from langgraph_agent.nodes import (
    RAGState,
    retrieve_node,
    generate_node,
    score_node,
    should_retry,
    retry_node,
    retry_count_node
)


def build_graph():
    # Initiate the LangGraph flow builder class
    builder = StateGraph(RAGState)

    # Add nodes to the graph
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("score", score_node)
    builder.add_node("retry", retry_node)
    builder.add_node("increment_retry", retry_count_node)

    # Define the flow
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "score")

    # Conditional Edge: if score < 0.8 and retry_count < max_retries, then retry
    builder.add_conditional_edges(
        "score",
        should_retry,
        {
            "retry": "retry",
            "end": END
        }
    )

    builder.add_edge("retry", "increment_retry")
    builder.add_edge("increment_retry", "retrieve")

    # Compile the graph
    return builder.compile()
