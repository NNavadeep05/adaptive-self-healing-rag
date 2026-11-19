"""
Document metadata for company-aware retrieval.

Every chunk of a known SEC filing is tagged with its company and fiscal
year, and gets a short header such as "[Amazon | Form 10-K | fiscal year
2025]" prepended before embedding. Without it, a chunk from a segment
table carries no clue about *which* company it describes, which caused
the "Microsoft segments for an Amazon question" failure (Q09).
"""

import hashlib
import re
import uuid
from pathlib import Path

# filename prefix -> (display name, query aliases)
COMPANIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "amazon": ("Amazon", ("amazon", "amzn", "aws")),
    "jpmorgan": ("JPMorgan Chase", ("jpmorgan", "jp morgan", "jpm", "chase")),
    "microsoft": ("Microsoft", ("microsoft", "msft")),
    "nvidia": ("NVIDIA", ("nvidia", "nvda")),
    "walmart": ("Walmart", ("walmart", "wmt")),
}

_FILENAME = re.compile(r"^([a-z]+)_(\d{4})$")
_POINT_NAMESPACE = uuid.UUID("6f1c3c1e-6a43-4c0e-9a59-1f1d6b0a7e21")


def filing_metadata(filename: str) -> dict:
    """Company and fiscal year for files named like "amazon_2025.html"."""
    match = _FILENAME.match(Path(filename).stem.lower())
    if not match or match.group(1) not in COMPANIES:
        return {"company": None, "fiscal_year": None}
    return {"company": COMPANIES[match.group(1)][0], "fiscal_year": match.group(2)}


def detect_companies(query: str) -> list[str]:
    """Companies named in a query, in registry order (whole-word match)."""
    text = query.lower()
    return [
        name
        for name, aliases in COMPANIES.values()
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases)
    ]


def chunk_header(metadata: dict, source: str) -> str:
    """Short context line prepended to every chunk before embedding."""
    if metadata.get("company"):
        return f"[{metadata['company']} | Form 10-K | fiscal year {metadata['fiscal_year']}]"
    return f"[Source: {source}]"


def point_id(source: str, chunk_index: int, text: str) -> str:
    """
    Deterministic Qdrant point id.

    Re-ingesting the same file produces the same ids, so Qdrant upserts
    (overwrites) instead of storing duplicate chunks.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{source}:{chunk_index}:{digest}"))
