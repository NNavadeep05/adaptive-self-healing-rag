from typing import TypedDict, List, Any

from langgraph_agent.retrieve_docs import (
    get_qdrant_client,
    get_doc_answer,
    rerank,
    generate_answer,
    llm_judge,
)


class RAGState(TypedDict):
    text: List[dict]
    query: str
    retrieved_docs: List[str]
    retrieval_mode: str
    retrieval_budget: int
    answer: str
    score: float
    failure_reason: str
    retry_count: int
    max_retries: int
    healing_trace: List[str]
    vector_store: Any


def retrieve_node(state: RAGState):
    query = state["query"]
    budget = state["retrieval_budget"]

    # Connect to the external Qdrant service.
    # QDRANT_URL is configured through the environment.
    client = get_qdrant_client()

    # Retrieve from the already-ingested persistent knowledge base.
    results = get_doc_answer(
        client=client,
        query=query,
        k=budget,
    )

    # Apply reranking when healing is triggered.
    if state["retrieval_mode"] == "dense_rerank":
        results = rerank(
            query=query,
            retrieved_docs=results,
        )

    return {
        "retrieved_docs": results,
        "healing_trace": state.get(
            "healing_trace",
            []
        ),
        "vector_store": client,
    }


def generate_node(state: RAGState):
    answer = generate_answer(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
    )

    return {
        "answer": answer
    }


def score_node(state: RAGState):
    judge = llm_judge(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
        answer=state["answer"],
    )

    score = judge["score"]
    relevant = judge["relevant_docs"]
    sufficient = judge["sufficient_context"]

    if not relevant:
        failure_reason = "irrelevant_docs"
    elif not sufficient:
        failure_reason = "missing_context"
    else:
        failure_reason = "none"

    return {
        "score": score,
        "failure_reason": failure_reason,
    }


def should_retry(state: RAGState):
    # Stop once the maximum retry limit is reached.
    if state["retry_count"] >= state["max_retries"]:
        return "end"

    failure_reason = state.get(
        "failure_reason",
        "none"
    )

    # Retry when the judge identifies a retrieval/context
    # failure or when the overall score is below the threshold.
    if state["score"] < 0.8 or failure_reason != "none":
        return "retry"

    return "end"


def retry_node(state: RAGState):
    failure = state["failure_reason"]
    trace = state.get("healing_trace", [])

    if failure == "missing_context":
        trace.append(
            "Missing context → increased retrieval budget by 3 + rerank"
        )

        return {
            "retrieval_budget": state["retrieval_budget"] + 3,
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace,
        }

    if failure == "irrelevant_docs":
        trace.append(
            "Irrelevant docs → enabled rerank + increased retrieval budget by 2"
        )

        return {
            "retrieval_budget": state["retrieval_budget"] + 2,
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace,
        }

    trace.append("No healing needed")

    return {
        "healing_trace": trace
    }


def retry_count_node(state: RAGState):
    return {
        "retry_count": state["retry_count"] + 1
    }