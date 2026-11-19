"""Tests for re-grading and tie handling in the evaluation harness (no LLM calls)."""

from eval.analysis import majority_correct
from eval.compare_systems import regrade_records

ITEMS = {"Q01": {"id": "Q01", "ground_truth": "331,839"},
         "Q17": {"id": "Q17", "ground_truth": "long text", "answer_facts": "40.31%",
                 "accepted_variants": ["40.3%"]}}


def record(qid: str, system: str, cell: dict) -> dict:
    return {"run": 1, "id": qid, "query": "q", "systems": {system: cell}}


def test_regrade_uses_the_current_scorer_for_every_cell():
    records = {
        (1, "Q01"): record("Q01", "plain_rag_k5",
                           {"correct": False, "answer": "Revenue was $331.839 billion", "error": None}),
        (1, "Q17"): record("Q17", "self_healing",
                           {"correct": False, "answer": "Net margin ≈ 40.3 %", "error": None,
                            "judge_passed": True, "silent_failure": True}),
    }
    regraded = regrade_records(records, ITEMS)
    assert regraded[(1, "Q01")]["systems"]["plain_rag_k5"]["correct"] is True
    loop = regraded[(1, "Q17")]["systems"]["self_healing"]
    assert loop["correct"] is True and loop["silent_failure"] is False
    # input untouched
    assert records[(1, "Q01")]["systems"]["plain_rag_k5"]["correct"] is False


def test_regrade_leaves_errored_cells_alone():
    records = {(1, "Q01"): record("Q01", "plain_rag_k5",
                                  {"correct": None, "answer": None, "error": "429"})}
    assert regrade_records(records, ITEMS) == records


def test_majority_treats_ties_as_not_correct():
    assert majority_correct(2 / 3)
    assert not majority_correct(0.5)


def test_daily_limit_detection():
    from eval.compare_systems import is_daily_limit
    assert is_daily_limit("RateLimitError: Error code: 429 - Limit 200000, tokens per day (TPD)")
    assert not is_daily_limit("RateLimitError: Error code: 429 - tokens per minute (TPM)")
    assert not is_daily_limit(None)


def test_error_messages_are_redacted_and_short():
    from eval.compare_systems import clean_error
    message = clean_error("RateLimitError: 429 in organization `org_01abcdef1234567890` " + "x" * 900)
    assert "org_01abcdef" not in message and "org_[redacted]" in message
    assert len(message) <= 400
