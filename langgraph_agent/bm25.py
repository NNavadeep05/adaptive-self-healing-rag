"""
Sparse keyword retrieval (BM25) and Reciprocal Rank Fusion.

Dense embeddings capture meaning ("revenue" ~ "net sales") but are weak
at exact tokens such as line-item names and numbers. BM25 is the
classic term-weighting ranker that is strong exactly there, so the two
are combined (hybrid retrieval) and their rankings fused with RRF.

BM25 score of document d for query q:

    sum over terms t in q of
        IDF(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""

import math
import re
from collections import Counter, defaultdict
from typing import Hashable, Iterable, Sequence

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or "
    "that the this to was were what which with".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word/number tokens; "716,924" stays one token "716924"."""
    text = _THOUSANDS.sub("", text.lower())
    return [token for token in _TOKEN.findall(text) if token not in _STOPWORDS]


class BM25Index:
    """In-memory BM25 (Okapi) index over a fixed list of documents."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        tokenized = [tokenize(doc) for doc in documents]
        self._lengths = [len(tokens) for tokens in tokenized]
        self._avg_length = (sum(self._lengths) / len(tokenized)) if tokenized else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, tokens in enumerate(tokenized):
            for term, count in Counter(tokens).items():
                self._postings[term].append((doc_id, count))
        n_docs = len(tokenized)
        self._idf = {
            term: math.log(1 + (n_docs - len(posts) + 0.5) / (len(posts) + 0.5))
            for term, posts in self._postings.items()
        }

    def scores(self, query: str) -> dict[int, float]:
        """BM25 score for every document sharing at least one query term."""
        totals: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self._postings[term]:
                norm = 1 - self.b + self.b * self._lengths[doc_id] / self._avg_length
                totals[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return totals

    def search(self, query: str, limit: int, allowed: set[int] | None = None) -> list[int]:
        """Indices of the top-``limit`` documents, optionally within ``allowed``."""
        scored = [
            (score, doc_id)
            for doc_id, score in self.scores(query).items()
            if allowed is None or doc_id in allowed
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for _, doc_id in scored[:limit]]


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[Hashable]],
    k: int = 60,
) -> list[Hashable]:
    """
    Fuse several ranked lists: score(d) = sum 1 / (k + rank_i(d)).

    Uses ranks only, so the incomparable score scales of BM25 and cosine
    similarity never need to be normalised. k=60 is the standard constant
    from Cormack et al. (2009).
    """
    fused: dict[Hashable, float] = defaultdict(float)
    first_seen: dict[Hashable, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(fused, key=lambda item: (-fused[item], first_seen[item]))
