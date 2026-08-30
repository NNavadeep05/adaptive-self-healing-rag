import os
import json
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


# Load environment variables
load_dotenv()


# ---------------------------------------------------------
# QDRANT CONFIGURATION
# ---------------------------------------------------------

QDRANT_URL = os.environ.get(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION_NAME = "documents"


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

def embed_docs(chunks):
    """
    Embed document chunks and store them in Qdrant.

    Each chunk stores:
        - description
        - source_pdf

    Qdrant provides persistent storage through
    the external Qdrant service.
    """

    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Initialize embedding model
    embedding_model = TextEmbedding(
        model_name=encoder_name
    )

    # Connect to Qdrant server
    client = get_qdrant_client()

    # Create collection if it does not exist
    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "embedding": VectorParams(
                    size=client.get_embedding_size(
                        encoder_name
                    ),
                    distance=Distance.COSINE
                )
            }
        )

    # Extract chunk text
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings
    vectors = list(
        embedding_model.embed(texts)
    )

    # Create Qdrant points
    points = []

    for chunk, vector in zip(chunks, vectors):

        point_id = str(
            uuid.uuid4()
        )

        points.append(
            PointStruct(
                id=point_id,
                payload={
                    "description": chunk["text"],
                    "source_pdf": chunk["source_pdf"]
                },
                vector={
                    "embedding": vector
                }
            )
        )

    # Upload to Qdrant
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return client


# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

def get_doc_answer(
    client,
    query: str,
    k: int = 2
) -> list[str]:
    """
    Retrieve the top-k document chunks relevant to a query.
    """

    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Initialize embedding model
    embedding_model = TextEmbedding(
        model_name=encoder_name
    )

    # Embed user query
    query_embedding = list(
        embedding_model.query_embed(query)
    )[0]

    # Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        using="embedding",
        query=query_embedding,
        with_payload=True,
        limit=k
    )

    print("\n--- Qdrant Retrieval Results ---")

    for i, point in enumerate(results.points):

        print(
            f"Rank {i + 1} | "
            f"Score: {point.score:.4f} | "
            f"Source: {point.payload.get('source_pdf')} | "
            f"Text: {point.payload['description']}"
        )

    print("--------------------------------\n")

    # Return retrieved text
    retrieved_docs = [
        point.payload["description"]
        for point in results.points
    ]

    return retrieved_docs


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

    reranker = TextCrossEncoder(
        model_name="jinaai/jina-reranker-v2-base-multilingual"
    )

    scores = list(
        reranker.rerank(
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
        base_url="https://api.groq.com/openai/v1"
    )


# ---------------------------------------------------------
# ANSWER GENERATION
# ---------------------------------------------------------

def generate_answer(
    query: str,
    retrieved_docs: list[str]
) -> str:
    """
    Generate an answer using retrieved documents.

    Generation model:
        openai/gpt-oss-120b via Groq
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