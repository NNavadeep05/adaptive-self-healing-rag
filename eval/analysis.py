"""
Statistical analysis of the stored evaluation results (no LLM calls).

For each bucket in eval/results/:

- per-system accuracy with a 95% bootstrap CI over questions
  (each question's score = its mean correctness over the runs)
- McNemar exact tests on per-question majority outcomes for the
  comparisons that matter
- the LLM judge's confusion matrix for each loop: is "pass" trustworthy?

Usage:
    python eval/analysis.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval.compare_systems import (  # noqa: E402
    RESULTS_DIR,
    SYSTEMS,
    load_records,
    regrade_records,
)
from eval.questions import BUCKETS, load_queries  # noqa: E402
from eval.stats import bootstrap_ci, holm_adjust, judge_confusion, mcnemar_exact  # noqa: E402

ANALYSIS_PATH = RESULTS_DIR / "analysis.json"
COMPARISONS = (
    ("plain_rag_k5", "self_healing"),
    ("wide_rerank_k11", "self_healing"),
    ("wide_rerank_k11", "hybrid_single_pass"),
    ("self_healing", "self_healing_v2"),
    ("hybrid_single_pass", "self_healing_v2"),
    ("plain_rag_k5", "self_healing_v2"),
)


def majority_correct(mean_score: float) -> bool:
    """A question counts as solved if most runs got it right (a 1-of-2 tie does not)."""
    return mean_score > 0.5


def per_question_scores(records: dict, system: str) -> dict[str, float]:
    """Question id -> mean correctness over the runs that scored it."""
    by_question: dict[str, list[bool]] = {}
    for record in records.values():
        cell = record["systems"].get(system)
        if cell and cell.get("correct") is not None:
            by_question.setdefault(record["id"], []).append(cell["correct"])
    return {qid: statistics.mean(v) for qid, v in by_question.items()}


def accuracy_table(scores: dict[str, dict[str, float]]) -> dict:
    return {
        system: {"questions": len(s), "accuracy": round(statistics.mean(s.values()), 3),
                 "ci95": bootstrap_ci(list(s.values()))}
        for system, s in scores.items() if s
    }


def comparisons(scores: dict[str, dict[str, float]]) -> list[dict]:
    rows = []
    for a, b in COMPARISONS:
        if a not in scores or b not in scores:
            continue
        shared = sorted(set(scores[a]) & set(scores[b]))
        result = mcnemar_exact([majority_correct(scores[a][q]) for q in shared],
                               [majority_correct(scores[b][q]) for q in shared])
        rows.append({"a": a, "b": b, "questions": len(shared), **result})
    # Several tests at once: Holm-adjust so "p < 0.05" keeps its meaning.
    adjusted = holm_adjust([row["p_value"] for row in rows])
    return [{**row, "p_value": round(row["p_value"], 3), "p_holm": round(p, 3)}
            for row, p in zip(rows, adjusted)]


def judge_tables(records: dict) -> dict:
    tables = {}
    for system in ("self_healing", "self_healing_v2"):
        cells = [r["systems"][system] for r in records.values()
                 if system in r["systems"] and r["systems"][system].get("correct") is not None]
        if cells:
            tables[system] = judge_confusion([c["judge_passed"] for c in cells],
                                             [c["correct"] for c in cells])
    return tables


def analyse(bucket: str) -> dict | None:
    records = load_records(bucket)
    if not records:
        return None
    records = regrade_records(records, {q["id"]: q for q in load_queries(bucket)})
    scores = {s: per_question_scores(records, s) for s in SYSTEMS}
    scores = {s: v for s, v in scores.items() if v}
    return {"accuracy": accuracy_table(scores), "mcnemar": comparisons(scores),
            "judge": judge_tables(records)}


def print_report(bucket: str, report: dict) -> None:
    print(f"\n=== {bucket} ===")
    for system, row in report["accuracy"].items():
        low, high = row["ci95"]
        print(f"  {system:<20} {row['accuracy']:.0%}  95% CI [{low:.0%}, {high:.0%}]"
              f"  ({row['questions']} questions)")
    for row in report["mcnemar"]:
        print(f"  McNemar {row['a']} vs {row['b']}: only_a={row['only_a']} "
              f"only_b={row['only_b']} p={row['p_value']} p_holm={row['p_holm']}")
    for system, table in report["judge"].items():
        print(f"  Judge ({system}): {table}")


def main() -> None:
    reports = {b: r for b in BUCKETS if (r := analyse(b))}
    ANALYSIS_PATH.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for bucket, report in reports.items():
        print_report(bucket, report)
    print(f"\nSaved {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
