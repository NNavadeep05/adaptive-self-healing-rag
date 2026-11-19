"""
Build the evaluation knowledge bases from the local SEC filings.

Two dedicated Qdrant collections, so evaluation never depends on what
else has been ingested into the application's "documents" collection:

    v1  sec_filings_eval     original parsing (one line per table cell),
                             no chunk headers. Reproduces the v1 results.
    v2  sec_filings_eval_v2  table-aware parsing (one line per table row)
                             + "[Company | Form 10-K | fiscal year]" header
                             + company / fiscal-year payload for filtering.

Usage:
    python eval/ingest_corpus.py --version v2
    python eval/ingest_corpus.py --version v1
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval.config import CORPUS_DIR, EVAL_COLLECTIONS  # noqa: E402
from langgraph_agent.document_loader import load_documents  # noqa: E402
from langgraph_agent.retrieve_docs import embed_docs, get_qdrant_client  # noqa: E402

PROTECTED_COLLECTIONS = {"documents"}


def build(version: str) -> None:
    collection = EVAL_COLLECTIONS[version]
    if collection in PROTECTED_COLLECTIONS:
        sys.exit(f"Refusing to rebuild the application's main collection '{collection}'.")

    filings = sorted(CORPUS_DIR.glob("*.htm*")) + sorted(CORPUS_DIR.glob("*.pdf"))
    if not filings:
        sys.exit(f"No filings found in {CORPUS_DIR}/. See README 'Repository Data'.")

    is_v2 = version == "v2"
    chunks = load_documents([str(path) for path in filings],
                            table_aware=is_v2, contextual_headers=is_v2)
    print(f"[{version}] Loaded {len(filings)} filings -> {len(chunks)} chunks")

    # Rebuild from scratch so the collection exactly matches the corpus.
    client = get_qdrant_client()
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
    finally:
        client.close()

    client = embed_docs(chunks, collection_name=collection)
    try:
        stored = client.count(collection_name=collection).count
    finally:
        client.close()
    print(f"[{version}] Collection '{collection}' now holds {stored} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evaluation collection.")
    parser.add_argument("--version", choices=sorted(EVAL_COLLECTIONS), default="v2")
    build(parser.parse_args().version)


if __name__ == "__main__":
    main()
