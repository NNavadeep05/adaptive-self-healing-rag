"""Tests for the evaluation statistics helpers."""

import pytest

from eval.stats import (
    bootstrap_ci,
    first_hit_rank,
    judge_confusion,
    mcnemar_exact,
)


def test_mcnemar_no_disagreement_is_not_significant():
    assert mcnemar_exact([True, False, True], [True, False, True]) == {
        "only_a": 0, "only_b": 0, "p_value": 1.0}


def test_mcnemar_counts_discordant_pairs():
    a = [True, True, True, False]
    b = [False, False, False, False]
    result = mcnemar_exact(a, b)
    assert (result["only_a"], result["only_b"]) == (3, 0)
    # two-sided exact binomial: 2 * 0.5**3
    assert result["p_value"] == pytest.approx(0.25)


def test_mcnemar_requires_paired_lists():
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    low, high = bootstrap_ci(values, seed=0)
    assert 0 <= low <= 0.7 <= high <= 1
    assert bootstrap_ci(values, seed=0) == (low, high)


def test_bootstrap_ci_of_constant_values_is_a_point():
    assert bootstrap_ci([1, 1, 1], seed=0) == (1.0, 1.0)


def test_judge_confusion_counts_silent_failures():
    passed = [True, True, False, False]
    correct = [True, False, True, False]
    result = judge_confusion(passed, correct)
    assert result["pass_correct"] == 1
    assert result["pass_wrong"] == 1  # silent failure
    assert result["fail_correct"] == 1
    assert result["fail_wrong"] == 1
    assert result["precision_of_pass"] == 0.5


def test_first_hit_rank_finds_smallest_k_containing_the_answer():
    docs = ["nothing", "revenue 331,839", "other"]
    assert first_hit_rank("331,839", docs) == 2
    assert first_hit_rank("999", docs) is None


def test_first_hit_rank_for_multi_part_answer_needs_all_parts():
    docs = ["Microsoft revenue 331,839", "filler", "NVIDIA revenue 215,938"]
    assert first_hit_rank("331,839; 215,938", docs) == 3


def test_holm_adjustment_is_monotone_and_capped():
    from eval.stats import holm_adjust
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08->0.09 (monotone), 0.5*1=0.5
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert holm_adjust([0.6, 0.7]) == [1.0, 1.0]
    assert holm_adjust([]) == []
