"""
Statistics for comparing RAG systems on a small question set.

With 10-20 questions, "80% vs 70%" can easily be noise, so comparisons
are reported with:

- McNemar's exact test: are two systems' per-question outcomes
  significantly different? Only the discordant questions (one system
  right, the other wrong) carry information.
- Bootstrap confidence intervals: resample questions with replacement
  to show how much the accuracy could move on a different sample.
- Judge confusion matrix: how often the LLM judge's "pass" is actually
  correct (its precision), which exposes silent failures.
"""

import math
import random
from typing import Sequence

from eval.scoring import contains_answer


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Two-sided exact McNemar test on paired boolean outcomes."""
    if len(a) != len(b):
        raise ValueError("outcome lists must be paired (same length)")
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    alpha = (1 - confidence) / 2
    low = means[int(alpha * resamples)]
    high = means[min(resamples - 1, int((1 - alpha) * resamples))]
    return round(low, 3), round(high, 3)


def judge_confusion(passed: Sequence[bool], correct: Sequence[bool]) -> dict:
    """Confusion matrix of the judge's verdict against ground-truth correctness."""
    pairs = list(zip(passed, correct))
    counts = {
        "pass_correct": sum(1 for p, c in pairs if p and c),
        "pass_wrong": sum(1 for p, c in pairs if p and not c),
        "fail_correct": sum(1 for p, c in pairs if not p and c),
        "fail_wrong": sum(1 for p, c in pairs if not p and not c),
    }
    n_pass = counts["pass_correct"] + counts["pass_wrong"]
    n_wrong = counts["pass_wrong"] + counts["fail_wrong"]
    return {
        **counts,
        "precision_of_pass": round(counts["pass_correct"] / n_pass, 3) if n_pass else None,
        # share of wrong answers the judge caught (recall on the "wrong" class)
        "wrong_answers_caught": round(counts["fail_wrong"] / n_wrong, 3) if n_wrong else None,
    }


def first_hit_rank(
    ground_truth: str,
    ranked_docs: Sequence[str],
    variants: Sequence[str] = (),
) -> int | None:
    """Smallest k such that the top-k documents together contain the answer."""
    for k in range(1, len(ranked_docs) + 1):
        if contains_answer(ground_truth, ranked_docs[:k], variants):
            return k
    return None


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """
    Holm-Bonferroni adjusted p-values (same order as the input).

    Running several significance tests at once inflates the chance of a
    false "significant" result; Holm controls that family-wise error
    rate and is uniformly more powerful than plain Bonferroni.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for position, index in enumerate(order):
        running_max = max(running_max, min(1.0, (m - position) * p_values[index]))
        adjusted[index] = running_max
    return adjusted
