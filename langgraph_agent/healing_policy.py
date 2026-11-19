"""
Failure-aware healing policy for the v2 loop.

The v1 loop answers every failure the same way (retrieve a few more
chunks and rerank). v2 maps each diagnosed failure to the action most
likely to fix it, and switches action if the first one did not help:

    missing_context     -> WIDEN    the right documents exist but too few
                                    were retrieved: increase k
    irrelevant_docs     -> REWRITE  retrieval missed the topic: rewrite the
                                    query in filing vocabulary
    unsupported_claims  -> REWRITE  the answer is not grounded in what was
                                    retrieved: look for better evidence
    low score, no reason-> WIDEN
"""

from typing import Sequence

WIDEN = "widen"
REWRITE = "rewrite_query"

_FIRST_CHOICE = {
    "missing_context": WIDEN,
    "irrelevant_docs": REWRITE,
    "unsupported_claims": REWRITE,
}

WIDEN_STEP = 4


def choose_healing_action(failure_reason: str, previous_actions: Sequence[str]) -> str:
    """Pick the next healing action; never repeat a rewrite that already failed."""
    preferred = _FIRST_CHOICE.get(failure_reason, WIDEN)
    if preferred not in previous_actions:
        return preferred
    alternative = REWRITE if preferred == WIDEN else WIDEN
    if alternative not in previous_actions:
        return alternative
    return WIDEN
