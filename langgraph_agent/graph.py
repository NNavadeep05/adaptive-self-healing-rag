from langgraph.graph import StateGraph, END

from langgraph_agent.nodes import (
    RAGState,
    retrieve_node,
    generate_node,
    verify_node,
    score_node,
    should_retry,
    retry_node,
    retry_count_node,
)


def build_graph():
    # Initialize the LangGraph state builder.
    builder = StateGraph(RAGState)

    # -----------------------------------------------------
    # Add nodes
    # -----------------------------------------------------

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)
    builder.add_node("score", score_node)
    builder.add_node("retry", retry_node)
    builder.add_node("increment_retry", retry_count_node)

    # -----------------------------------------------------
    # Define main pipeline
    # -----------------------------------------------------

    builder.set_entry_point("retrieve")

    builder.add_edge(
        "retrieve",
        "generate"
    )

    # Self-verification happens after generation.
    builder.add_edge(
        "generate",
        "verify"
    )

    # Existing LLM judge runs after verification.
    builder.add_edge(
        "verify",
        "score"
    )

    # -----------------------------------------------------
    # Decide whether healing is required
    # -----------------------------------------------------

    builder.add_conditional_edges(
        "score",
        should_retry,
        {
            "retry": "retry",
            "end": END,
        },
    )

    # -----------------------------------------------------
    # Healing loop
    # -----------------------------------------------------

    builder.add_edge(
        "retry",
        "increment_retry"
    )

    builder.add_edge(
        "increment_retry",
        "retrieve"
    )

    # -----------------------------------------------------
    # Compile graph
    # -----------------------------------------------------

    return builder.compile()