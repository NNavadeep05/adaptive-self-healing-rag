"""
Loading evaluation questions and their scoring targets.

Straightforward questions are scored on ``ground_truth``. Hard questions
(comparisons, computed margins, fiscal-year dates) have narrative ground
truths, so they carry two structured targets:

    answer_facts    what the final answer must contain, e.g. both
                    revenue figures, or the computed "40.31%" margin
    evidence_facts  what the retrieved chunks must contain to make the
                    answer possible, e.g. the net income AND revenue
                    figures a margin is computed from
"""

import json

from eval.config import QUERIES_PATH
from eval.scoring import contains_answer

BUCKETS = ("straightforward", "hard")


def load_queries(bucket: str = "all") -> list[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        items = json.load(f)
    return items if bucket == "all" else [q for q in items if q["bucket"] == bucket]


def _variants(item: dict) -> list[str]:
    return item.get("accepted_variants", [])


def answer_correct(item: dict, answer: str) -> bool:
    target = item.get("answer_facts", item["ground_truth"])
    return bool(contains_answer(target, [answer], _variants(item)))


def evidence_target(item: dict) -> tuple[str, list[str]]:
    """(target, variants) for retrieval recall."""
    if "evidence_facts" in item:
        return item["evidence_facts"], []
    return item["ground_truth"], _variants(item)
