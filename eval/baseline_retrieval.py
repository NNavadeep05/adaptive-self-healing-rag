import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langgraph_agent.retrieve_docs import (
    get_qdrant_client,
    get_doc_answer,
)


def answer_found_in_topk(
    ground_truth: str,
    retrieved_docs: list[str],
) -> bool | None:
    """
    Check whether all components of the ground-truth answer
    appear in the retrieved chunks.

    Returns:
        True  -> all required answer components found
        False -> at least one component missing
        None  -> ground truth has not been populated yet
    """

    if not ground_truth:
        return None

    combined = (
        " ".join(retrieved_docs)
        .replace(",", "")
        .replace("$", "")
        .lower()
    )

    # Split multi-part answers on commas or the word "and".
    # This avoids false negatives caused by phrasing differences.
    parts = [
        part.strip()
        for part in re.split(r",|\band\b", ground_truth.lower())
        if part.strip()
    ]

    return all(
        part.replace(",", "").replace("$", "").strip() in combined
        for part in parts
    )


def main():
    with open("eval_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    client = get_qdrant_client()

    total = 0
    answerable = 0
    not_found = 0
    unverified = 0

    for item in queries:
        if item["bucket"] != "straightforward":
            continue

        total += 1

        print(f"\n{'=' * 70}")
        print(f"{item['id']}: {item['query']}")
        print(f"Target: {item['target_companies']}")

        docs = get_doc_answer(
            client,
            item["query"],
            k=5,
        )

        target = item["target_companies"][0].lower()

        # Check whether the target company's filing appears
        # in the retrieved chunks.
        retrieved_target = False

        for doc in docs:
            source = doc.lower()

            if target == "jpmorgan chase":
                if "jpmorgan" in source or "jpm-2025" in source:
                    retrieved_target = True
            elif target in source:
                retrieved_target = True

        answer_found = answer_found_in_topk(
            item["ground_truth"],
            docs,
        )

        print(f"Target company retrieved: {retrieved_target}")
        print(f"Answer found in top-5: {answer_found}")

        if answer_found is True:
            answerable += 1
        elif answer_found is False:
            not_found += 1
        else:
            unverified += 1

    print(f"\n{'=' * 70}")
    print("BASELINE SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total straightforward queries: {total}")
    print(f"Answer found in top-5:        {answerable}")
    print(f"Answer NOT found in top-5:    {not_found}")
    print(f"Ground truth not populated:   {unverified}")

    if total > 0:
        verified = answerable + not_found

        if verified > 0:
            print(
                f"Verified retrieval rate:      "
                f"{answerable}/{verified} "
                f"({answerable / verified * 100:.1f}%)"
            )


if __name__ == "__main__":
    main()