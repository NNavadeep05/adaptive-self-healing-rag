from typing import TypedDict, List, Any

from langgraph_agent.retrieve_docs import (
    get_qdrant_client,
    get_doc_answer,
    rerank,
    generate_answer,
    llm_judge,
    self_verify,
)


class RAGState(TypedDict):
    text: List[dict]
    query: str
    retrieved_docs: List[str]

    # Retrieval
    retrieval_mode: str
    retrieval_budget: int

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

    # Qdrant client
    vector_store: Any


# ---------------------------------------------------------
# RETRIEVAL NODE
# ---------------------------------------------------------

def retrieve_node(state: RAGState):
    query = state["query"]
    budget = state["retrieval_budget"]

    # Connect to the external Qdrant service.
    client = get_qdrant_client()

    # Retrieve documents from the existing knowledge base.
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


# ---------------------------------------------------------
# GENERATION NODE
# ---------------------------------------------------------

def generate_node(state: RAGState):
    answer = generate_answer(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
    )

    return {
        "answer": answer
    }


# ---------------------------------------------------------
# SELF-VERIFICATION NODE
# ---------------------------------------------------------

def verify_node(state: RAGState):
    """
    Verify whether the generated answer is supported
    by the retrieved documents.

    This is our entailment/faithfulness-based version
    of the paper's Self-Verification Module.
    """

    verification = self_verify(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
        answer=state["answer"],
    )

    status = verification["verification_status"]
    verification_score = verification["verification_score"]

    # Keep verification as an independent signal.
    unsupported_claims = (
        status != "supported"
    )

    return {
        "verification_status": status,
        "verification_score": verification_score,
        "unsupported_claims": unsupported_claims,
    }


# ---------------------------------------------------------
# LLM-AS-A-JUDGE NODE
# ---------------------------------------------------------

def score_node(state: RAGState):
    """
    Evaluate the generated answer using GPT-OSS-20B.

    The judge checks:
        - document relevance
        - context sufficiency
        - overall answer quality

    The verification signal is kept separately and is
    given priority when determining the failure reason.
    """

    judge = llm_judge(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"],
        answer=state["answer"],
    )

    score = judge["score"]
    relevant = judge["relevant_docs"]
    sufficient = judge["sufficient_context"]

    # Preserve the independent verification signal.
    if state.get(
        "unsupported_claims",
        False
    ):
        failure_reason = "unsupported_claims"

    elif not relevant:
        failure_reason = "irrelevant_docs"

    elif not sufficient:
        failure_reason = "missing_context"

    else:
        failure_reason = "none"

    return {
        "score": score,
        "failure_reason": failure_reason,
    }


# ---------------------------------------------------------
# RETRY DECISION
# ---------------------------------------------------------

def should_retry(state: RAGState):
    """
    Decide whether the system should activate
    the self-healing retrieval process.
    """

    # Stop once the maximum retry limit is reached.
    if state["retry_count"] >= state["max_retries"]:
        return "end"

    failure_reason = state.get(
        "failure_reason",
        "none"
    )

    unsupported_claims = state.get(
        "unsupported_claims",
        False
    )

    # Retry when:
    # 1. Judge score is below threshold
    # 2. Judge identifies a retrieval/context failure
    # 3. Self-verification finds unsupported claims
    if (
        state["score"] < 0.8
        or failure_reason != "none"
        or unsupported_claims
    ):
        return "retry"

    return "end"


# ---------------------------------------------------------
# RETRY / RETRIEVAL REFINEMENT NODE
# ---------------------------------------------------------

def retry_node(state: RAGState):
    """
    Retrieval Refinement Engine.

    Adjust retrieval strategy based on the detected
    failure type before the next retrieval attempt.
    """

    failure = state["failure_reason"]
    trace = state.get(
        "healing_trace",
        []
    )

    # -----------------------------------------------------
    # Missing context
    # -----------------------------------------------------

    if failure == "missing_context":

        trace.append(
            "Missing context → increased retrieval "
            "budget by 3 + enabled reranking"
        )

        return {
            "retrieval_budget": (
                state["retrieval_budget"] + 3
            ),
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace,
        }

    # -----------------------------------------------------
    # Irrelevant documents
    # -----------------------------------------------------

    if failure == "irrelevant_docs":

        trace.append(
            "Irrelevant docs → increased retrieval "
            "budget by 2 + enabled reranking"
        )

        return {
            "retrieval_budget": (
                state["retrieval_budget"] + 2
            ),
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace,
        }

    # -----------------------------------------------------
    # Unsupported claims
    # -----------------------------------------------------

    if failure == "unsupported_claims":

        trace.append(
            "Unsupported claims → enabled reranking "
            "+ increased retrieval budget by 2"
        )

        return {
            "retrieval_budget": (
                state["retrieval_budget"] + 2
            ),
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace,
        }

    # -----------------------------------------------------
    # Generic low-score failure
    # -----------------------------------------------------

    if state["score"] < 0.8:

        trace.append(
            "Low judge score → enabled reranking "
            "+ increased retrieval budget by 2"
        )

        return {
            "retrieval_budget": (
                state["retrieval_budget"] + 2
            ),
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace,
        }

    # -----------------------------------------------------
    # No healing required
    # -----------------------------------------------------

    trace.append(
        "No healing needed"
    )

    return {
        "healing_trace": trace
    }


# ---------------------------------------------------------
# RETRY COUNT NODE
# ---------------------------------------------------------

def retry_count_node(state: RAGState):
    return {
        "retry_count": (
            state["retry_count"] + 1
        )
    }