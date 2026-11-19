"""Tests for the v2 failure-aware healing policy."""

import pytest

from langgraph_agent.healing_policy import (
    REWRITE,
    WIDEN,
    choose_healing_action,
)


@pytest.mark.parametrize("failure, expected", [
    ("missing_context", WIDEN),
    ("irrelevant_docs", REWRITE),
    ("unsupported_claims", REWRITE),
    ("none", WIDEN),  # low judge score with no named reason
])
def test_first_action_depends_on_failure_type(failure, expected):
    assert choose_healing_action(failure, previous_actions=()) == expected


def test_repeated_failure_switches_to_the_other_action():
    assert choose_healing_action("irrelevant_docs", previous_actions=(REWRITE,)) == WIDEN
    assert choose_healing_action("missing_context", previous_actions=(WIDEN,)) == REWRITE


def test_after_both_actions_keep_widening():
    assert choose_healing_action("irrelevant_docs", previous_actions=(REWRITE, WIDEN)) == WIDEN
