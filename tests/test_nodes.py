"""Tests for the self-healing control logic (no LLM or Qdrant calls)."""

import pytest

from langgraph_agent.graph import build_graph, make_initial_state
from langgraph_agent.nodes import retry_count_node, retry_node, should_retry


def state(**overrides) -> dict:
    return {**make_initial_state("q"), "score": 0.95, **overrides}


# ---------------------------------------------------------
# should_retry
# ---------------------------------------------------------

def test_passing_answer_ends():
    assert should_retry(state()) == "end"


@pytest.mark.parametrize("overrides", [
    {"score": 0.5},
    {"failure_reason": "irrelevant_docs"},
    {"failure_reason": "missing_context"},
    {"unsupported_claims": True},
])
def test_any_failure_signal_triggers_retry(overrides):
    assert should_retry(state(**overrides)) == "retry"


def test_retry_budget_is_respected():
    exhausted = state(score=0.1, failure_reason="missing_context",
                      retry_count=3, max_retries=3)
    assert should_retry(exhausted) == "end"


# ---------------------------------------------------------
# retry_node
# ---------------------------------------------------------

@pytest.mark.parametrize("failure, budget_increase", [
    ("missing_context", 3),
    ("irrelevant_docs", 2),
    ("unsupported_claims", 2),
])
def test_failure_type_sets_retrieval_strategy(failure, budget_increase):
    update = retry_node(state(failure_reason=failure, retrieval_budget=5))
    assert update["retrieval_budget"] == 5 + budget_increase
    assert update["retrieval_mode"] == "dense_rerank"
    assert len(update["healing_trace"]) == 1


def test_low_score_without_reason_still_heals():
    update = retry_node(state(score=0.4, retrieval_budget=5))
    assert update["retrieval_budget"] == 7
    assert update["retrieval_mode"] == "dense_rerank"


def test_retry_count_increments():
    assert retry_count_node(state(retry_count=1)) == {"retry_count": 2}


# ---------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------

def test_graph_contains_all_pipeline_nodes():
    nodes = set(build_graph().get_graph().nodes)
    assert {"retrieve", "generate", "verify", "score",
            "retry", "increment_retry"} <= nodes


def test_initial_state_defaults():
    s = make_initial_state("What was revenue?")
    assert s["query"] == "What was revenue?"
    assert s["retrieval_budget"] == 5
    assert s["max_retries"] == 3
    assert s["retry_count"] == 0
    assert s["healing_trace"] == []


# ---------------------------------------------------------
# v2 (hybrid) healing
# ---------------------------------------------------------

def v2_state(**overrides) -> dict:
    return {**make_initial_state("What was Amazon's net sales?", strategy="hybrid"),
            "score": 0.4, **overrides}


def test_v2_initial_state():
    s = make_initial_state("q", strategy="hybrid", collection="c")
    assert s["retrieval_strategy"] == "hybrid"
    assert s["collection"] == "c"
    assert s["search_query"] == "q"
    assert s["healing_actions"] == []


def test_v2_missing_context_widens_retrieval():
    update = retry_node(v2_state(failure_reason="missing_context", retrieval_budget=8))
    assert update["retrieval_budget"] == 12
    assert update["healing_actions"] == ["widen"]


def test_v2_irrelevant_docs_rewrites_query(monkeypatch):
    monkeypatch.setattr("langgraph_agent.nodes.rewrite_query",
                        lambda q: "Amazon 10-K 2025 Total net sales")
    update = retry_node(v2_state(failure_reason="irrelevant_docs"))
    assert update["search_query"] == "Amazon 10-K 2025 Total net sales"
    assert update["healing_actions"] == ["rewrite_query"]
    assert "retrieval_budget" not in update


def test_v2_second_failure_switches_action(monkeypatch):
    monkeypatch.setattr("langgraph_agent.nodes.rewrite_query", lambda q: "x")
    update = retry_node(v2_state(failure_reason="irrelevant_docs",
                                 healing_actions=["rewrite_query"], retrieval_budget=8))
    assert update["retrieval_budget"] == 12
    assert update["healing_actions"] == ["rewrite_query", "widen"]


def test_retry_node_does_not_mutate_the_trace():
    trace = ["earlier"]
    retry_node(state(failure_reason="missing_context", healing_trace=trace))
    assert trace == ["earlier"]
