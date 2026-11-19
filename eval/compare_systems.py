"""
End-to-end comparison of v1 and v2 RAG systems on one question bucket.

Systems (all answers graded by eval/scoring.py via eval/questions.py):

    plain_rag_k5        v1  dense top-5 -> generate once              (1 LLM call)
    wide_rerank_k11     v1  dense top-11 -> rerank -> generate once   (1 LLM call)
    self_healing        v1  LangGraph loop, dense, k=5 (+2/+3 per retry)
    hybrid_single_pass  v2  hybrid retrieval k=8 -> grounded prompt   (1 LLM call)
    self_healing_v2     v2  LangGraph loop, hybrid, failure-aware healing

Results are stored per (run, question, system) in eval/results/<bucket>.json.
Re-running the same command only fills cells that are missing or errored,
so a run interrupted by an API rate limit can simply be resumed later.

Usage:
    python eval/compare_systems.py --bucket straightforward --systems hybrid_single_pass self_healing_v2
    python eval/compare_systems.py --bucket hard --runs 2
    python eval/compare_systems.py --import-v1      # bring in eval/full_pipeline_results.json
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval.config import EVAL_COLLECTIONS  # noqa: E402
from eval.questions import BUCKETS, answer_correct, load_queries  # noqa: E402
from langgraph_agent.graph import build_graph, make_initial_state  # noqa: E402
from langgraph_agent.healing_policy import REWRITE  # noqa: E402
from langgraph_agent.hybrid_retrieval import hybrid_search  # noqa: E402
from langgraph_agent.nodes import JUDGE_PASS_SCORE  # noqa: E402
from langgraph_agent.retrieve_docs import (  # noqa: E402
    generate_answer,
    get_doc_answer,
    get_qdrant_client,
    rerank,
)

RESULTS_DIR = Path("eval/results")
V1_RESULTS = Path("eval/full_pipeline_results.json")
LLM_CALLS_PER_ATTEMPT = 3  # generate + verify + judge
V1, V2 = EVAL_COLLECTIONS["v1"], EVAL_COLLECTIONS["v2"]

# name -> (kind, retrieval function or loop strategy, grounded prompt)
SINGLE_PASS = {
    "plain_rag_k5": (lambda c, q: get_doc_answer(c, q, 5, V1), False),
    "wide_rerank_k11": (lambda c, q: rerank(q, get_doc_answer(c, q, 11, V1)), False),
    "hybrid_single_pass": (lambda c, q: hybrid_search(c, q, 8, V2), True),
}
LOOPS = {
    "self_healing": ("dense", V1),
    "self_healing_v2": ("hybrid", V2),
}
SYSTEMS = (*SINGLE_PASS, *LOOPS)


_ORG_ID = re.compile(r"org_[A-Za-z0-9]+")
MAX_ERROR_CHARS = 400


def clean_error(message: str) -> str:
    """Shorten provider errors and redact account identifiers before storing them."""
    return _ORG_ID.sub("org_[redacted]", message)[:MAX_ERROR_CHARS]


def timed(fn, *args):
    """(result, seconds, error message or None) — errors are recorded, not raised."""
    start = time.perf_counter()
    try:
        return fn(*args), round(time.perf_counter() - start, 2), None
    except Exception as e:  # e.g. rate limits; the cell can be retried later
        return None, round(time.perf_counter() - start, 2), clean_error(f"{type(e).__name__}: {e}")


def run_single_pass(name: str, item: dict, client) -> dict:
    retrieve, grounded = SINGLE_PASS[name]

    def answer():
        return generate_answer(item["query"], retrieve(client, item["query"]), grounded=grounded)

    text, seconds, error = timed(answer)
    return {"correct": None if error else answer_correct(item, text), "answer": text,
            "llm_calls": 1, "seconds": seconds, "error": error}


def run_loop(name: str, item: dict, graph) -> dict:
    strategy, collection = LOOPS[name]
    state = make_initial_state(item["query"], strategy=strategy, collection=collection)
    result, seconds, error = timed(graph.invoke, state)
    if error:
        return {"correct": None, "answer": None, "llm_calls": None,
                "seconds": seconds, "error": error}

    correct = answer_correct(item, result["answer"])
    judge_passed = (result["score"] >= JUDGE_PASS_SCORE
                    and result["failure_reason"] == "none")
    rewrites = result.get("healing_actions", []).count(REWRITE)
    return {
        "correct": correct,
        "answer": result["answer"],
        "llm_calls": LLM_CALLS_PER_ATTEMPT * (result["retry_count"] + 1) + rewrites,
        "seconds": seconds,
        "error": None,
        "retry_count": result["retry_count"],
        "judge_score": result["score"],
        "failure_reason": result["failure_reason"],
        "judge_passed": judge_passed,
        "silent_failure": judge_passed and not correct,
        "healing_trace": result["healing_trace"],
    }


def results_path(bucket: str) -> Path:
    return RESULTS_DIR / f"{bucket}.json"


def load_records(bucket: str) -> dict[tuple[int, str], dict]:
    path = results_path(bucket)
    if not path.exists():
        return {}
    return {(r["run"], r["id"]): r for r in json.loads(path.read_text(encoding="utf-8"))}


def save_records(bucket: str, records: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda r: (r["run"], r["id"]))
    # Write to a temp file, then rename: an interrupted run never corrupts results.
    target = results_path(bucket)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def is_daily_limit(error: str | None) -> bool:
    """Groq's tokens-per-day limit: retrying today is pointless."""
    return bool(error) and "429" in error and "per day" in error.lower()


def needs_run(record: dict, system: str) -> bool:
    cell = record["systems"].get(system)
    return cell is None or cell.get("error") is not None


def run(bucket: str, runs: int, systems: list[str], ids: list[str] | None = None) -> dict:
    items = [q for q in load_queries(bucket) if not ids or q["id"] in ids]
    records = load_records(bucket)
    exhausted: set[str] = set()  # systems whose model hit the daily token limit
    graph = build_graph()
    client = get_qdrant_client()
    try:
        for run_no in range(1, runs + 1):
            for item in items:
                key = (run_no, item["id"])
                record = records.get(key, {"run": run_no, "id": item["id"],
                                           "query": item["query"], "systems": {}})
                for system in (s for s in systems
                               if s not in exhausted and needs_run(record, s)):
                    cell = (run_single_pass(system, item, client) if system in SINGLE_PASS
                            else run_loop(system, item, graph))
                    if is_daily_limit(cell["error"]):
                        exhausted.add(system)
                        print(f"[{bucket}] {system}: daily token limit reached; "
                              "skipping it for the rest of this run (re-run later to resume).")
                    record = {**record, "systems": {**record["systems"], system: cell}}
                    print(f"[{bucket} run {run_no}] {item['id']} {system}: "
                          f"correct={cell['correct']} error={cell['error']}", flush=True)
                    records = {**records, key: record}
                    save_records(bucket, records)  # checkpoint after every cell
    finally:
        client.close()
    return records


def regrade_records(records: dict, items_by_id: dict) -> dict:
    """
    Re-grade every stored answer with the current scorer (no LLM calls).

    Keeps all systems graded by exactly the same rules, including cells
    imported from older result files or stored before a scorer fix.
    """
    regraded = {}
    for key, record in records.items():
        item = items_by_id.get(record["id"])
        systems = {}
        for name, cell in record["systems"].items():
            if item is None or cell.get("error") is not None or cell.get("answer") is None:
                systems[name] = cell
                continue
            correct = answer_correct(item, cell["answer"])
            updated = {**cell, "correct": correct}
            if "judge_passed" in cell:
                updated["silent_failure"] = cell["judge_passed"] and not correct
            systems[name] = updated
        regraded[key] = {**record, "systems": systems}
    return regraded


def import_v1() -> None:
    """Convert the published v1 straightforward results into this format."""
    records = load_records("straightforward")
    for old in json.loads(V1_RESULTS.read_text(encoding="utf-8")):
        key = (old["run"], old["id"])
        base = records.get(key, {"run": old["run"], "id": old["id"],
                                 "query": old["query"], "systems": {}})
        v1_cells = {s: old[s] for s in ("plain_rag_k5", "wide_rerank_k11", "self_healing")}
        records = {**records, key: {**base, "systems": {**base["systems"], **v1_cells}}}
    save_records("straightforward", records)
    print(f"Imported {V1_RESULTS} into {results_path('straightforward')}")


def summarize(records: dict) -> dict:
    summary = {}
    for system in SYSTEMS:
        cells = [r["systems"][system] for r in records.values() if system in r["systems"]]
        scored = [c for c in cells if c.get("correct") is not None]
        if not cells:
            continue
        loops = [c for c in scored if "retry_count" in c]
        summary[system] = {
            "cells": len(cells),
            "errors": len(cells) - len(scored),
            "accuracy": round(statistics.mean(c["correct"] for c in scored), 3) if scored else None,
            "mean_llm_calls": round(statistics.mean(c["llm_calls"] for c in scored), 2) if scored else None,
            "mean_seconds": round(statistics.mean(c["seconds"] for c in scored), 1) if scored else None,
            **({"retry_rate": round(statistics.mean(c["retry_count"] > 0 for c in loops), 3),
                "silent_failures": sum(c["silent_failure"] for c in loops)} if loops else {}),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v1 and v2 RAG systems.")
    parser.add_argument("--bucket", choices=BUCKETS, default="straightforward")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--ids", nargs="*", help="only these question ids (default: all)")
    parser.add_argument("--import-v1", action="store_true")
    args = parser.parse_args()

    if args.import_v1:
        import_v1()
        bucket = "straightforward"
        records = load_records(bucket)
    else:
        bucket = args.bucket
        records = run(bucket, args.runs, args.systems, args.ids)

    records = regrade_records(records, {q["id"]: q for q in load_queries(bucket)})
    save_records(bucket, records)

    summary = summarize(records)
    (RESULTS_DIR / f"{bucket}_summary.json").write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")
    print(f"\n{bucket}:")
    for system, row in summary.items():
        print(f"  {system:<20} {row}")


if __name__ == "__main__":
    main()
