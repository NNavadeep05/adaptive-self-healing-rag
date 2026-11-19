r"""
v2 retrieval pipeline.

    query
      |-- company detection --> Qdrant payload filter (only that company's chunks)
      |                         2+ companies: one search per company, interleaved
      |-- dense search (MiniLM embeddings, top `pool`)   \
      |-- BM25 keyword search (top `pool`)               /-- Reciprocal Rank Fusion
      v
    fused candidates (top `pool`) --> cross-encoder rerank --> keep top k

Retrieving a wide candidate pool and then truncating after reranking is
the standard "retrieve wide, rerank, truncate" pattern: recall comes
from the cheap retrievers, precision from the expensive cross-encoder,
and the LLM still only sees k chunks.
"""

import logging
import threading
from dataclasses import dataclass, field

from langgraph_agent.bm25 import BM25Index, reciprocal_rank_fusion
from langgraph_agent.metadata import detect_companies
from langgraph_agent.retrieve_docs import COLLECTION_NAME, dense_search, rerank

logger = logging.getLogger(__name__)

DEFAULT_POOL = 30
_SCROLL_PAGE = 512


@dataclass(frozen=True)
class RetrievalConfig:
    """Switches used by the retrieval ablation; the defaults are the v2 pipeline."""
    dense: bool = True
    sparse: bool = True
    company_filter: bool = True
    rerank: bool = True
    decompose: bool = True
    pool: int = DEFAULT_POOL


@dataclass(frozen=True)
class Corpus:
    """All chunks of a collection, plus a BM25 index over them."""
    ids: tuple[str, ...]
    texts: tuple[str, ...]
    companies: tuple[str | None, ...]
    sources: tuple[str, ...]
    index: BM25Index = field(repr=False)

    def text_of(self) -> dict[str, str]:
        return dict(zip(self.ids, self.texts))

    def indices_for(self, companies: list[str]) -> set[int] | None:
        """Chunks of the given companies plus untagged chunks (None = all)."""
        if not companies:
            return None
        wanted = set(companies)
        return {i for i, company in enumerate(self.companies)
                if company is None or company in wanted}


_MAX_CACHED_CORPORA = 4
_corpus_cache: dict[tuple[str, int], Corpus] = {}
_corpus_lock = threading.Lock()


def load_corpus(client, collection_name: str) -> Corpus:
    """
    Read every chunk of the collection once and build the BM25 index.

    Cached per (collection, point count), so it is rebuilt automatically
    after new documents are ingested.
    """
    count = client.count(collection_name=collection_name).count
    key = (collection_name, count)
    with _corpus_lock:  # concurrent API requests build the index only once
        if key not in _corpus_cache:
            _corpus_cache[key] = _build_corpus(client, collection_name)
            stale = [k for k in _corpus_cache if k[0] == collection_name and k != key]
            for old in stale + list(_corpus_cache)[:-_MAX_CACHED_CORPORA]:
                _corpus_cache.pop(old, None)
        return _corpus_cache[key]


def _build_corpus(client, collection_name: str) -> Corpus:
    """Scroll every point of the collection and index the texts with BM25."""
    ids, texts, companies, sources = [], [], [], []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_PAGE,
            offset=offset,
            with_payload=["description", "company", "source_pdf"],
            with_vectors=False,
        )
        for point in points:
            ids.append(str(point.id))
            texts.append(point.payload["description"])
            companies.append(point.payload.get("company"))
            sources.append(point.payload.get("source_pdf", ""))
        if offset is None:
            break

    corpus = Corpus(tuple(ids), tuple(texts), tuple(companies), tuple(sources),
                    BM25Index(texts))
    logger.info("BM25 index built over %d chunks of '%s'", len(ids), collection_name)
    return corpus


def resolve_companies(query: str, corpus: Corpus) -> list[str]:
    """
    Companies named in the query that the collection actually has chunks for.

    Collections without company metadata (e.g. user uploads) therefore
    never get filtered down to nothing.
    """
    known = {company for company in corpus.companies if company}
    return [company for company in detect_companies(query) if company in known]


def interleave(rankings: list[list[str]]) -> list[str]:
    """Round-robin merge of ranked lists, dropping duplicates."""
    merged: list[str] = []
    for position in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if position < len(ranking) and ranking[position] not in merged:
                merged.append(ranking[position])
    return merged


def entity_precision(chunk_companies: list[str | None], targets: list[str]) -> float | None:
    """Share of retrieved chunks that belong to one of the target companies."""
    if not chunk_companies:
        return None
    wanted = set(targets)
    return sum(1 for c in chunk_companies if c in wanted) / len(chunk_companies)


def hybrid_search(
    client,
    query: str,
    k: int,
    collection_name: str | None = None,
    config: RetrievalConfig = RetrievalConfig(),
    filter_query: str | None = None,
) -> list[str]:
    """
    Return the top-k chunk texts for ``query`` using the v2 pipeline.

    ``filter_query`` is the text used for company detection; the healing
    loop passes the user's original question here when ``query`` is a
    rewritten search query.

    Questions naming several companies ("Compare Microsoft's revenue with
    NVIDIA's") are decomposed: one filtered search per company, results
    interleaved, so every company's evidence reaches the top k instead of
    one company crowding out the other.
    """
    corpus = load_corpus(client, collection_name or COLLECTION_NAME)
    companies = (resolve_companies(filter_query or query, corpus)
                 if config.company_filter else [])

    if config.decompose and len(companies) > 1:
        per_company = [
            _search(client, query, k, collection_name, corpus, [company], config)
            for company in companies
        ]
        return interleave(per_company)[:k]
    return _search(client, query, k, collection_name, corpus, companies, config)


def _search(
    client,
    query: str,
    k: int,
    collection_name: str | None,
    corpus: Corpus,
    companies: list[str],
    config: RetrievalConfig,
) -> list[str]:
    """One hybrid retrieval: filtered dense + BM25 -> RRF -> rerank -> top k."""
    collection = collection_name or COLLECTION_NAME
    texts = corpus.text_of()

    rankings = []
    if config.dense:
        dense = dense_search(client, query, config.pool, collection, companies or None)
        texts.update(dense)
        rankings.append([point_id for point_id, _ in dense])
    if config.sparse:
        hits = corpus.index.search(query, config.pool, corpus.indices_for(companies))
        rankings.append([corpus.ids[i] for i in hits])

    candidates = [texts[point_id] for point_id in reciprocal_rank_fusion(rankings)[:config.pool]]
    if config.rerank and candidates:
        candidates = rerank(query, candidates)
    return candidates[:k]
