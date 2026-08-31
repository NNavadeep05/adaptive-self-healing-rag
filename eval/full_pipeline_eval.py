import json
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from langgraph_agent.graph import build_graph
from eval.baseline_retrieval import (
    answer_found_in_topk,
)


def main():
    # -----------------------------------------------------
    # Load evaluation queries
    # -----------------------------------------------------

    with open(
        "eval_queries.json",
        "r",
        encoding="utf-8",
    ) as f:
        queries = json.load(f)

    # -----------------------------------------------------
    # Build the actual self-healing graph
    # -----------------------------------------------------

    graph = build_graph()

    total = 0
    baseline_success = 0
    final_success = 0
    baseline_failures = 0
    recovered = 0

    results = []

    # -----------------------------------------------------
    # Run straightforward evaluation set
    # -----------------------------------------------------

    for item in queries:

        if item["bucket"] != "straightforward":
            continue

        total += 1

        print(f"\n{'=' * 70}")
        print(f"{item['id']}: {item['query']}")
        print(f"Target: {item['target_companies']}")

        # -------------------------------------------------
        # Determine baseline status from the same
        # top-k retrieval used in Session 11.
        # -------------------------------------------------

        baseline_docs = graph.nodes["retrieve"].bound.invoke(
            {
                "query": item["query"],
                "retrieval_mode": "original",
                "retrieval_budget": 5,
                "retrieved_docs": [],
            }
        )

        retrieved_docs = baseline_docs.get(
            "retrieved_docs",
            []
        )

        baseline_found = answer_found_in_topk(
            item["ground_truth"],
            retrieved_docs,
        )

        baseline_ok = baseline_found is True

        if baseline_ok:
            baseline_success += 1
        else:
            baseline_failures += 1

        # -------------------------------------------------
        # Initialize full self-healing pipeline.
        #
        # IMPORTANT:
        # Initial retrieval budget = 5 so the first
        # attempt matches the Session 11 baseline.
        # -------------------------------------------------

        initial_state = {
            "query": item["query"],
            "text": [],
            "retrieved_docs": [],
            "retrieval_mode": "original",
            "retrieval_budget": 5,
            "answer": "",
            "score": 0.0,
            "failure_reason": "none",
            "verification_status": "none",
            "verification_score": 0.0,
            "unsupported_claims": False,
            "retry_count": 0,
            "max_retries": 3,
            "healing_trace": [],
            "vector_store": None,
        }

        # -------------------------------------------------
        # Run complete self-healing graph
        # -------------------------------------------------

        result = graph.invoke(initial_state)

        final_answer = result.get(
            "answer",
            "",
        )

        final_score = result.get(
            "score",
            0.0,
        )

        retry_count = result.get(
            "retry_count",
            0,
        )

        final_budget = result.get(
            "retrieval_budget",
            5,
        )

        verification_status = result.get(
            "verification_status",
            "unknown",
        )

        failure_reason = result.get(
            "failure_reason",
            "unknown",
        )

        healing_trace = result.get(
            "healing_trace",
            [],
        )

        # -------------------------------------------------
        # Determine whether final answer contains the
        # expected ground truth.
        # -------------------------------------------------

        final_found = answer_found_in_topk(
            item["ground_truth"],
            [final_answer],
        )

        final_ok = final_found is True

        if final_ok:
            final_success += 1

        # Recovery means:
        # baseline failed AND final pipeline succeeded.
        was_recovered = (
            not baseline_ok
            and final_ok
        )

        if was_recovered:
            recovered += 1

        # -------------------------------------------------
        # Print query-level result
        # -------------------------------------------------

        print(f"Baseline found:      {baseline_ok}")
        print(f"Final answer found:  {final_ok}")
        print(f"Final score:         {final_score:.2f}")
        print(f"Retry count:         {retry_count}")
        print(f"Final budget:        {final_budget}")
        print(f"Verification:       {verification_status}")
        print(f"Failure reason:      {failure_reason}")
        print(f"Recovered:           {was_recovered}")

        if healing_trace:
            print("Healing trace:")
            for trace in healing_trace:
                print(f"  - {trace}")

        print("Final answer:")
        print(final_answer)

        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "ground_truth": item["ground_truth"],
                "baseline_success": baseline_ok,
                "final_success": final_ok,
                "recovered": was_recovered,
                "final_score": final_score,
                "retry_count": retry_count,
                "final_retrieval_budget": final_budget,
                "verification_status": verification_status,
                "failure_reason": failure_reason,
                "healing_trace": healing_trace,
                "final_answer": final_answer,
            }
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(f"\n{'=' * 70}")
    print("SELF-HEALING PIPELINE SUMMARY")
    print(f"{'=' * 70}")

    print(f"Total straightforward queries: {total}")
    print(
        f"Baseline success:              "
        f"{baseline_success}/{total}"
    )
    print(
        f"Final pipeline success:        "
        f"{final_success}/{total}"
    )
    print(
        f"Baseline failures:             "
        f"{baseline_failures}"
    )
    print(
        f"Recovered by self-healing:     "
        f"{recovered}/{baseline_failures}"
    )

    if total > 0:
        print(
            f"Baseline success rate:         "
            f"{baseline_success / total * 100:.1f}%"
        )
        print(
            f"Final pipeline success rate:   "
            f"{final_success / total * 100:.1f}%"
        )

    if baseline_failures > 0:
        print(
            f"Recovery rate:                 "
            f"{recovered / baseline_failures * 100:.1f}%"
        )

    # -----------------------------------------------------
    # Save detailed results
    # -----------------------------------------------------

    output_path = Path(
        "eval/full_pipeline_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nDetailed results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()