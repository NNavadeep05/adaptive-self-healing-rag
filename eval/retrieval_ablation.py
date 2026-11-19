"""
Retrieval ablation: which retrieval change actually helps? (no LLM calls)

Each configuration ranks the top-30 chunks for every question. For each
question we record the first rank k at which the top-k chunks together
contain all of the evidence facts, then report:

    recall@k  share of questions whose evidence is inside the top k
    MRR       mean of 1 / first-hit-rank (0 when never found in the top 30)

Configurations add one change at a time, so each row's gain can be
attributed to that change:

    v1_dense                  original parsing, dense only (the v1 baseline)
    v2_dense                  + table-aware parsing + company chunk headers
    v2_dense_filter           + company metadata filter
    v2_bm25_filter            BM25 keyword search alone (with filter)
    v2_hybrid_filter          dense + BM25 fused with RRF
    v2_hybrid_filter_rerank   + cross-encoder rerank of the fused pool
    v2_full                   + per-company query decomposition        (= v2)
    v2_full_nofilter          v2 without the company filter (so no decomposition)

Entity precision@5 is the share of the top-5 chunks that come from a
company the question is about (cross-company leakage shows up here).

Usage:
    python eval/ingest_corpus.py --version v1   # if not built yet
    python eval/ingest_corpus.py --version v2
    python eval/retrieval_ablation.py
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval.config import EVAL_COLLECTIONS  # noqa: E402
from eval.questions import BUCKETS, evidence_target, load_queries  # noqa: E402
from eval.stats import bootstrap_ci, first_hit_rank  # noqa: E402
from langgraph_agent.hybrid_retrieval import (  # noqa: E402
    RetrievalConfig,
    entity_precision,
    hybrid_search,
    load_corpus,
)
from langgraph_agent.metadata import filing_metadata  # noqa: E402
from langgraph_agent.retrieve_docs import get_qdrant_client  # noqa: E402

RESULTS_PATH = Path("eval/retrieval_ablation.json")
POOL = 30
RECALL_KS = (1, 3, 5, 8, 11, 20, 30)

_NO_DECOMPOSE = {"decompose": False}
CONFIGS: dict[str, tuple[str, RetrievalConfig]] = {
    "v1_dense": ("v1", RetrievalConfig(sparse=False, company_filter=False, rerank=False)),
    "v2_dense": ("v2", RetrievalConfig(sparse=False, company_filter=False, rerank=False)),
    "v2_dense_filter": ("v2", RetrievalConfig(sparse=False, rerank=False, **_NO_DECOMPOSE)),
    "v2_bm25_filter": ("v2", RetrievalConfig(dense=False, rerank=False, **_NO_DECOMPOSE)),
    "v2_hybrid_filter": ("v2", RetrievalConfig(rerank=False, **_NO_DECOMPOSE)),
    "v2_hybrid_filter_rerank": ("v2", RetrievalConfig(**_NO_DECOMPOSE)),
    "v2_full": ("v2", RetrievalConfig()),
    "v2_full_nofilter": ("v2", RetrievalConfig(company_filter=False)),
}
ENTITY_K = 5


def company_lookup(client) -> dict[str, str | None]:
    """Chunk text -> company, derived from each chunk's source file name."""
    lookup = {}
    for collection in EVAL_COLLECTIONS.values():
        corpus = load_corpus(client, collection)
        lookup.update({text: filing_metadata(source)["company"]
                       for text, source in zip(corpus.texts, corpus.sources)})
    return lookup


def evaluate(client, item: dict, version: str, config: RetrievalConfig,
             companies: dict[str, str | None]) -> dict:
    docs = hybrid_search(client, item["query"], k=POOL,
                         collection_name=EVAL_COLLECTIONS[version], config=config)
    target, variants = evidence_target(item)
    return {
        "first_hit_rank": first_hit_rank(target, docs, variants),
        "entity_precision_at_5": entity_precision(
            [companies.get(doc) for doc in docs[:ENTITY_K]], item["target_companies"]),
    }


def summarize(results: list[dict]) -> dict:
    ranks = [r["first_hit_rank"] for r in results]
    reciprocal = [1 / r if r else 0.0 for r in ranks]
    precision = [r["entity_precision_at_5"] for r in results
                 if r["entity_precision_at_5"] is not None]
    return {
        "entity_precision@5": round(statistics.mean(precision), 3) if precision else None,
        **{f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / len(ranks), 3)
           for k in RECALL_KS},
        "mrr": round(statistics.mean(reciprocal), 3),
        "mrr_95ci": bootstrap_ci(reciprocal),
    }


def print_table(summary: dict) -> None:
    for bucket, configs in summary.items():
        n = configs["_n"]
        print(f"\n{bucket} ({n} questions)")
        header = "".join(f"{'R@' + str(k):>7}" for k in RECALL_KS)
        print(f"{'config':<27}{header}{'MRR':>7}{'EntP@5':>8}")
        for name in (n for n in CONFIGS if n in configs):
            row = configs[name]
            cells = "".join(f"{row[f'recall@{k}']:>7.0%}" for k in RECALL_KS)
            print(f"{name:<27}{cells}{row['mrr']:>7.2f}{row['entity_precision@5']:>8.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only ablation.")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    args = parser.parse_args()

    items = load_queries("all")
    client = get_qdrant_client()
    try:
        companies = company_lookup(client)
        results = {
            name: {item["id"]: evaluate(client, item, *CONFIGS[name], companies)
                   for item in items}
            for name in args.configs
        }
    finally:
        client.close()

    summary = {}
    for bucket in (*BUCKETS, "all"):
        ids = [q["id"] for q in items if bucket == "all" or q["bucket"] == bucket]
        summary[bucket] = {"_n": len(ids), **{
            name: summarize([results[name][i] for i in ids]) for name in args.configs}}

    RESULTS_PATH.write_text(json.dumps({"per_question": results, "summary": summary},
                                       indent=2), encoding="utf-8")
    print_table(summary)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
