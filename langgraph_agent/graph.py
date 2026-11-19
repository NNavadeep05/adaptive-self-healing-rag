import os

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


DEFAULT_RETRIEVAL_BUDGET = 5         # v1 (dense) starting k
HYBRID_RETRIEVAL_BUDGET = 8          # v2 (hybrid) starting k
DEFAULT_MAX_RETRIES = 3

# Strategy used by the API and the Streamlit app ("hybrid" = v2, "dense" = v1).
APP_STRATEGY = os.environ.get("RAG_STRATEGY", "hybrid")


def make_initial_state(
    query: str,
    retrieval_budget: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retrieval_mode: str = "original",
    strategy: str = "dense",
    collection: str | None = None,
) -> dict:
    """
    Build the starting state for one query.

    Shared by the API, the Streamlit UI and the evaluation scripts so
    every entry point starts the pipeline from the same configuration.
    ``strategy`` is "dense" (v1) or "hybrid" (v2); ``collection`` None
    means the default Qdrant collection.
    """
    if retrieval_budget is None:
        retrieval_budget = (HYBRID_RETRIEVAL_BUDGET if strategy == "hybrid"
                            else DEFAULT_RETRIEVAL_BUDGET)
    return {
        "text": [],
        "query": query,
        "retrieved_docs": [],
        "retrieval_strategy": strategy,
        "retrieval_mode": retrieval_mode,
        "retrieval_budget": retrieval_budget,
        "search_query": query,
        "collection": collection,
        "healing_actions": [],
        "answer": "",
        "verification_status": "none",
        "verification_score": 0.0,
        "unsupported_claims": False,
        "score": 0.0,
        "failure_reason": "none",
        "retry_count": 0,
        "max_retries": max_retries,
        "healing_trace": [],
        "vector_store": None,
    }


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
