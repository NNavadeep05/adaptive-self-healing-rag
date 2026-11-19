"""Shared paths and collection names for the evaluation scripts."""

from pathlib import Path

CORPUS_DIR = Path("data/sec_filings")
QUERIES_PATH = Path("eval_queries.json")

EVAL_COLLECTIONS = {
    "v1": "sec_filings_eval",
    "v2": "sec_filings_eval_v2",
}
