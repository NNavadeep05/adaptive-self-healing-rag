"""
LangGraph nodes for the self-healing RAG loop.

Two retrieval strategies share the same graph:

    "dense"  (v1)  dense top-k, reranking switched on by the first retry,
                   every failure healed by "retrieve a few more chunks".
    "hybrid" (v2)  company filter + dense/BM25 fusion + rerank-truncate,
                   grounded answer prompt, and a failure-aware healing
                   policy (widen retrieval vs. rewrite the query).
"""

from typing import Any, List, TypedDict

from langgraph_agent.healing_policy import REWRITE, WIDEN_STEP, choose_healing_action
from langgraph_agent.hybrid_retrieval import hybrid_search
from langgraph_agent.query_rewrite import rewrite_query
from langgraph_agent.retrieve_docs import (
    generate_answer,
    get_doc_answer,
    get_qdrant_client,
    llm_judge,
    rerank,
    self_verify,
)

HYBRID = "hybrid"
JUDGE_PASS_SCORE = 0.8


class RAGState(TypedDict):
    text: List[dict]
    query: str
    retrieved_docs: List[str]

    # Retrieval
    retrieval_strategy: str      # "dense" (v1) or "hybrid" (v2)
    retrieval_mode: str          # v1 only: "original" or "dense_rerank"
    retrieval_budget: int
    search_query: str            # v2: may be rewritten by the healing loop
    collection: Any              # Qdrant collection (None = default)

    # Generation
    answer: str

    # Self-Verification
    verification_status: str
    verification_score: float
    unsupported_claims: bool

    # LLM-as-a-Judge
    score: float
    failure_reason: str

    # Retry / Healing
    retry_count: int
    max_retries: int
    healing_trace: List[str]
    healing_actions: List[str]   # v2: actions already tried

    # Qdrant client
    vector_store: Any


def _is_hybrid(state: RAGState) -> bool:
    return state.get("retrieval_strategy") == HYBRID


# ---------------------------------------------------------
# RETRIEVAL NODE
# ---------------------------------------------------------

def retrieve_node(state: RAGState):
    query = state["query"]
    budget = state["retrieval_budget"]
    client = get_qdrant_client()

    if _is_hybrid(state):
        results = hybrid_search(
            client=client,
            query=state.get("search_query") or query,
            k=budget,
            collection_name=state.get("collection"),
            filter_query=query,  # detect companies from the user's own words
        )
    else:
        results = get_doc_answer(
            client=client,
            query=query,
            k=budget,
            collection_name=state.get("collection"),
        )
        # v1 applies reranking only once healing has been triggered.
        if state["retrieval_mode"] == "dense_rerank":
            results = rerank(query=query, retrieved_docs=results)

    return {
        "retrieved_docs": results,
        "vector_store": client,
    }


# ---------------------------------------------------------
# GENERATION NODE
# ---------------------------------------------------------

def generate_node(state: RAGState):
    answer = generate_answer(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
        grounded=_is_hybrid(state),
    )
    return {"answer": answer}


# ---------------------------------------------------------
# SELF-VERIFICATION NODE
# ---------------------------------------------------------

def verify_node(state: RAGState):
    """
    Verify whether the generated answer is supported by the retrieved
    documents (the paper's Self-Verification Module, implemented as an
    LLM entailment / faithfulness check).
    """
    verification = self_verify(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
        answer=state["answer"],
    )
    status = verification["verification_status"]
    return {
        "verification_status": status,
        "verification_score": verification["verification_score"],
        "unsupported_claims": status != "supported",
    }


# ---------------------------------------------------------
# LLM-AS-A-JUDGE NODE
# ---------------------------------------------------------

def score_node(state: RAGState):
    """
    Judge relevance, context sufficiency and overall quality, and turn
    the signals into one failure reason. Verification has priority.
    """
    judge = llm_judge(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
        answer=state["answer"],
    )

    if state.get("unsupported_claims", False):
        failure_reason = "unsupported_claims"
    elif not judge["relevant_docs"]:
        failure_reason = "irrelevant_docs"
    elif not judge["sufficient_context"]:
        failure_reason = "missing_context"
    else:
        failure_reason = "none"

    return {
        "score": judge["score"],
        "failure_reason": failure_reason,
    }


# ---------------------------------------------------------
# RETRY DECISION
# ---------------------------------------------------------

def should_retry(state: RAGState):
    """Retry on a low judge score, a named failure, or unsupported claims."""
    if state["retry_count"] >= state["max_retries"]:
        return "end"

    if (
        state["score"] < JUDGE_PASS_SCORE
        or state.get("failure_reason", "none") != "none"
        or state.get("unsupported_claims", False)
    ):
        return "retry"

    return "end"


# ---------------------------------------------------------
# RETRY / RETRIEVAL REFINEMENT NODE
# ---------------------------------------------------------

# v1: failure -> (extra chunks, trace message)
_V1_HEALING = {
    "missing_context": (3, "Missing context → increased retrieval budget by 3 + enabled reranking"),
    "irrelevant_docs": (2, "Irrelevant docs → increased retrieval budget by 2 + enabled reranking"),
    "unsupported_claims": (2, "Unsupported claims → enabled reranking + increased retrieval budget by 2"),
}
_V1_LOW_SCORE = (2, "Low judge score → enabled reranking + increased retrieval budget by 2")


def _retry_v1(state: RAGState) -> dict:
    trace = state.get("healing_trace", [])
    step = _V1_HEALING.get(state["failure_reason"])
    if step is None and state["score"] < JUDGE_PASS_SCORE:
        step = _V1_LOW_SCORE
    if step is None:
        return {"healing_trace": [*trace, "No healing needed"]}

    extra, message = step
    return {
        "retrieval_budget": state["retrieval_budget"] + extra,
        "retrieval_mode": "dense_rerank",
        "healing_trace": [*trace, message],
    }


def _retry_v2(state: RAGState) -> dict:
    trace = state.get("healing_trace", [])
    actions = state.get("healing_actions", [])
    failure = state["failure_reason"]
    action = choose_healing_action(failure, actions)

    if action == REWRITE:
        new_query = rewrite_query(state["query"])
        note = "unchanged (rewrite failed)" if new_query == state["query"] else new_query
        return {
            "search_query": new_query,
            "healing_actions": [*actions, action],
            "healing_trace": [*trace, f"{failure} → rewrote search query: {note}"],
        }

    new_budget = state["retrieval_budget"] + WIDEN_STEP
    return {
        "retrieval_budget": new_budget,
        "healing_actions": [*actions, action],
        "healing_trace": [*trace, f"{failure} → widened retrieval to k={new_budget}"],
    }


def retry_node(state: RAGState):
    """Retrieval Refinement Engine: adjust retrieval based on the failure type."""
    return _retry_v2(state) if _is_hybrid(state) else _retry_v1(state)


# ---------------------------------------------------------
# RETRY COUNT NODE
# ---------------------------------------------------------

def retry_count_node(state: RAGState):
    return {"retry_count": state["retry_count"] + 1}
