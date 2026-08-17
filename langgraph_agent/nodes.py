from typing import TypedDict, List
from langgraph_agent.retrieve_docs import embed_docs, get_doc_answer, rerank, generate_answer, llm_judge

class RAGState(TypedDict):
    text: List[str]
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

def retrieve_node(state: RAGState):
    query = state["query"]
    budget = state["retrieval_budget"]
    mode = state["retrieval_mode"]
    text = state["text"]
    
    # Embed Documents
    docs = embed_docs(text)

    # Get Answer
    results = get_doc_answer(client=docs, query=query, k=budget)
    
    # Read retrieval model
    if state["retrieval_mode"] == "dense_rerank":
        results = rerank(query=query, retrieved_docs=results)
    
    return {"retrieved_docs": results,
            "healing_trace": state["healing_trace"]}

def generate_node(state: RAGState):
    answer = generate_answer(query=state["query"], retrieved_docs=state["retrieved_docs"])
    return {"answer": answer}

def score_node(state: RAGState):
    judge = llm_judge(query=state["query"], 
                      retrieved_docs=state["retrieved_docs"], 
                      answer=state["answer"])
    
    score = judge["score"]
    relevant = judge["relevant_docs"]
    sufficient = judge["sufficient_context"]

    # Determine failure reason
    if not relevant:
        failure_reason = "irrelevant_docs"
    elif not sufficient:
        failure_reason = "missing_context"
    else:
        failure_reason = "none"

    return {
        "score": score,
        "failure_reason": failure_reason
    }

def should_retry(state: RAGState):
    if state["score"] < 0.8 and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "end"

def retry_node(state: RAGState):
    failure = state["failure_reason"]
    trace = state.get("healing_trace", [])

    if failure == "missing_context":
        trace.append("Missing context → increased retrieval budget by 3 + rerank")
        return {
            "retrieval_budget": state["retrieval_budget"] + 3,
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace
        }

    if failure == "irrelevant_docs":
        trace.append("Irrelevant docs → enabled rerank + increased retrieval budget by 2")
        return {
            "retrieval_budget": state["retrieval_budget"] + 2,
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace
        }

    trace.append("No healing needed")
    return {"healing_trace": trace}

def retry_count_node(state: RAGState):
    return {"retry_count": state["retry_count"] + 1}
