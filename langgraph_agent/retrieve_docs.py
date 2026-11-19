import os
import json
import logging
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    IsNullCondition,
    MatchAny,
    PayloadField,
    PointStruct,
    VectorParams,
)
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from langgraph_agent.metadata import point_id


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
EMBED_BATCH_SIZE = 32


# ---------------------------------------------------------
# MODEL CACHE
# ---------------------------------------------------------
# Loading these models is expensive (the reranker is ~1 GB), so each
# is loaded once per process instead of on every retrieval call.

@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANKER_MODEL_NAME)


# ---------------------------------------------------------
# QDRANT CONFIGURATION
# ---------------------------------------------------------

QDRANT_URL = os.environ.get(
    "QDRANT_URL",
    "http://localhost:6333"
)

# Overridable so evaluation can use a dedicated, reproducible collection.
COLLECTION_NAME = os.environ.get(
    "QDRANT_COLLECTION",
    "documents"
)


def get_qdrant_client():
    """
    Create a client connected to the Qdrant server.

    Local development:
        http://localhost:6333

    Docker:
        http://qdrant:6333
    """

    return QdrantClient(
        url=QDRANT_URL
    )


# ---------------------------------------------------------
# EMBEDDING + STORAGE
# ---------------------------------------------------------

def _payload_filter(companies: list[str] | None) -> Filter | None:
    """
    Restrict a search to chunks of the given companies (None = no filter).

    Chunks without a company tag (e.g. user uploads with arbitrary file
    names) stay eligible, so a filter can never hide them.
    """
    if not companies:
        return None
    return Filter(should=[
        FieldCondition(key="company", match=MatchAny(any=companies)),
        IsNullCondition(is_null=PayloadField(key="company")),
        IsEmptyCondition(is_empty=PayloadField(key="company")),
    ])


def embed_docs(chunks, collection_name: str | None = None):
    """
    Embed document chunks and store them in Qdrant.

    Each point stores the chunk text ("description"), its source file and,
    when known, its company / fiscal year / chunk index. Point ids are
    deterministic, so re-ingesting the same file overwrites its chunks
    instead of duplicating them.
    """
    collection = collection_name or COLLECTION_NAME
    client = get_qdrant_client()

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "embedding": VectorParams(
                    size=client.get_embedding_size(EMBEDDING_MODEL_NAME),
                    distance=Distance.COSINE,
                )
            },
        )

    texts = [chunk["text"] for chunk in chunks]
    # Small batches keep peak memory low (a 256-chunk batch needs ~200 MB).
    vectors = get_embedding_model().embed(texts, batch_size=EMBED_BATCH_SIZE)

    points = [
        PointStruct(
            id=point_id(chunk["source_pdf"], chunk.get("chunk_index", i), chunk["text"]),
            payload={
                "description": chunk["text"],
                "source_pdf": chunk["source_pdf"],
                "company": chunk.get("company"),
                "fiscal_year": chunk.get("fiscal_year"),
                "chunk_index": chunk.get("chunk_index", i),
            },
            vector={"embedding": vector},
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    client.upload_points(collection_name=collection, points=points)
    return client


# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

def dense_search(
    client,
    query: str,
    k: int,
    collection_name: str | None = None,
    companies: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Top-k (point id, chunk text) pairs by embedding similarity."""
    query_embedding = next(iter(get_embedding_model().query_embed(query)))
    results = client.query_points(
        collection_name=collection_name or COLLECTION_NAME,
        using="embedding",
        query=query_embedding,
        query_filter=_payload_filter(companies),
        with_payload=True,
        limit=k,
    )

    for i, point in enumerate(results.points):
        logger.debug(
            "Rank %d | Score: %.4f | Source: %s | Text: %.200s",
            i + 1,
            point.score,
            point.payload.get("source_pdf"),
            point.payload["description"],
        )

    return [(str(point.id), point.payload["description"]) for point in results.points]


def get_doc_answer(
    client,
    query: str,
    k: int = 2,
    collection_name: str | None = None,
) -> list[str]:
    """
    Retrieve the top-k document chunks relevant to a query (v1: dense only).
    """
    return [text for _, text in dense_search(client, query, k, collection_name)]


# ---------------------------------------------------------
# RERANKING
# ---------------------------------------------------------

def rerank(
    query: str,
    retrieved_docs: list[str]
) -> list[str]:
    """
    Rerank retrieved documents using a cross-encoder.
    """

    scores = list(
        get_reranker().rerank(
            query,
            retrieved_docs
        )
    )

    scored_docs = list(
        zip(
            retrieved_docs,
            scores
        )
    )

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score in scored_docs
    ]


# ---------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------

def get_groq_client():
    """
    Create the OpenAI-compatible client used to access
    Groq-hosted models.
    """

    return OpenAI(
        api_key=os.environ.get(
            "GROQ_API_KEY"
        ),
        base_url="https://api.groq.com/openai/v1",
        # Back off and retry on rate limits / transient errors
        # (the SDK honours Groq's retry-after headers).
        max_retries=6,
    )


# ---------------------------------------------------------
# ANSWER GENERATION
# ---------------------------------------------------------

# v2 answer rules. They target the failure modes seen in evaluation:
# summing segment figures instead of quoting the reported total (Q05),
# answering with another company's figures (Q09), and restating numbers
# in rounded billions so they no longer match the filing.
GROUNDED_ANSWER_RULES = (
    "Rules:\n"
    "- Each document starts with a header naming its company and fiscal "
    "year. Use only documents about the company the question asks about.\n"
    "- Quote figures exactly as reported, with the filing's units "
    "(e.g. '$57,048 million'). Do not round or convert them.\n"
    "- Prefer a consolidated total that the filing reports directly. "
    "Never compute a total by adding segments when the total is reported.\n"
    "- If a calculation is asked for, show the reported inputs and the "
    "formula, then the result.\n"
)


def generate_answer(
    query: str,
    retrieved_docs: list[str],
    grounded: bool = False,
) -> str:
    """
    Generate an answer using retrieved documents.

    Generation model:
        openai/gpt-oss-120b via Groq

    ``grounded=True`` (v2) adds GROUNDED_ANSWER_RULES to the prompt.
    """

    client = get_groq_client()

    formatted_docs = "\n\n".join(
        f"[Document {i + 1}]\n{doc}"
        for i, doc in enumerate(retrieved_docs)
    )

    system_content = (
        "Use the following documents to answer "
        "the user question:\n\n"
        f"{formatted_docs}\n\n"
        "Answer the question using only the "
        "provided documents. "
        "If the answer cannot be found in the "
        "documents, respond with "
        "'I didn't find any relevant documents.'"
    )
    if grounded:
        system_content = f"{system_content}\n\n{GROUNDED_ANSWER_RULES}"

    ai_answer = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    return ai_answer.choices[0].message.content


# ---------------------------------------------------------
# SELF-VERIFICATION
# ---------------------------------------------------------

self_verification_prompt = """
You are a verification module for a
Retrieval-Augmented Generation system.

Your task is to determine whether the generated answer
is actually supported by the retrieved documents.

User question:
{query}

Retrieved documents:
{retrieved_docs}

Generated answer:
{answer}

Evaluate whether the claims made in the generated answer
are supported by the retrieved documents.

Return ONLY valid JSON in this format:

{{
  "verification_status": "supported" | "partially_supported" | "unsupported" | "contradicted",
  "verification_score": number between 0 and 1
}}

Guidelines:

- "supported":
  The retrieved documents provide sufficient evidence
  for the important claims in the answer.

- "partially_supported":
  Some claims are supported, but important information
  in the answer is missing from the retrieved documents.

- "unsupported":
  The answer contains claims that cannot be supported
  by the retrieved documents.

- "contradicted":
  The retrieved documents directly contradict important
  claims in the generated answer.

- verification_score represents how strongly the
  retrieved documents support the generated answer.

- A fully supported answer should receive a high score.

- An answer containing unsupported or hallucinated claims
  should receive a low score.

- Do not use outside knowledge.
- Judge only whether the retrieved documents support
  the generated answer.
"""


def self_verify(
    query: str,
    retrieved_docs: list[str],
    answer: str
) -> dict:
    """
    Verify whether the generated answer is supported
    by the retrieved documents.

    Uses GPT-OSS-20B via Groq as an LLM-based
    entailment/faithfulness checker.

    Returns:
        {
            "verification_status": ...,
            "verification_score": ...
        }
    """

    prompt = self_verification_prompt.format(
        query=query,
        retrieved_docs=retrieved_docs,
        answer=answer
    )

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    result = json.loads(
        response.choices[0].message.content
    )

    return {
        "verification_status": result.get(
            "verification_status",
            "unsupported"
        ),
        "verification_score": float(
            result.get(
                "verification_score",
                0.0
            )
        )
    }


# ---------------------------------------------------------
# LLM-AS-A-JUDGE
# ---------------------------------------------------------

llm_judge_prompt = """
You are an expert evaluator of Retrieval-Augmented Generation systems.

User question:
{query}

Retrieved documents:
{retrieved_docs}

Generated answer:
{answer}

Evaluate the answer using the retrieved documents.

Answer the following in JSON:
{{
  "relevant_docs": true | false,
  "sufficient_context": true | false,
  "score": number between 0 and 1
}}

Guidelines:
- relevant_docs = false if the retrieved documents do not address the user question
- sufficient_context = false if the retrieved documents are related but incomplete
- sufficient_context = false if the generated answer says "I didn't find any relevant documents."
  or otherwise states that it cannot answer the question
- If the generated answer is a fallback response saying that no relevant documents
  were found, treat it as an unsuccessful answer even if some retrieved documents
  appear relevant
- score should reflect the overall quality, relevance, and faithfulness of the answer
- A correct answer supported by the retrieved documents should receive a high score
- An answer that cannot answer the user's question should receive a low score
"""


def llm_judge(
    query,
    retrieved_docs,
    answer
):
    """
    Evaluate the generated answer using the retrieved documents.

    Uses GPT-OSS-20B via Groq.
    """

    prompt = llm_judge_prompt.format(
        query=query,
        retrieved_docs=retrieved_docs,
        answer=answer
    )

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    return json.loads(
        response.choices[0].message.content
    )
