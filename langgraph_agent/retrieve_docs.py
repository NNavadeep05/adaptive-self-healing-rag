import os
import json

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


def embed_docs(text):
    """
    Embed document chunks and store them in an in-memory Qdrant collection.

    Args:
        text (list[str]): Document chunks to embed.

    Returns:
        QdrantClient: Client containing the embedded document collection.
    """

    # Initialize the embedding model
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = TextEmbedding(model_name=encoder_name)

    # Convert every text chunk into an embedding vector
    vectors = list(embedding_model.embed(text))

    # Create an in-memory Qdrant database
    client = QdrantClient(":memory:")

    # Create the vector collection
    if not client.collection_exists("test_collection"):
        client.create_collection(
            collection_name="test_collection",
            vectors_config={
                "embedding": VectorParams(
                    size=client.get_embedding_size(encoder_name),
                    distance=Distance.COSINE
                )
            }
        )

    # Store each chunk together with its embedding
    client.upload_points(
        collection_name="test_collection",
        points=[
            PointStruct(
                id=idx,
                payload={"description": chunk},
                vector={"embedding": vector}
            )
            for idx, (chunk, vector) in enumerate(zip(text, vectors))
        ]
    )

    return client

def get_doc_answer(client, query: str, k: int = 2) -> list[str]:
    """
    Retrieve the top-k document chunks relevant to a query.

    Args:
        client: Qdrant client containing the document vectors.
        query: User's search query.
        k: Number of documents to retrieve.

    Returns:
        List of retrieved document chunks.
    """

    # Initialize the same embedding model used for documents
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = TextEmbedding(model_name=encoder_name)

    # Convert the user query into an embedding
    query_embedding = list(
        embedding_model.query_embed(query)
    )[0]

    # Search Qdrant for the most similar document vectors
    results = client.query_points(
        collection_name="test_collection",
        using="embedding",
        query=query_embedding,
        with_payload=True,
        limit=k
    )

    # Print retrieved documents and their similarity scores for inspection
    print("\n--- Qdrant Retrieval Results ---")
    for i, point in enumerate(results.points):
        print(f"Rank {i+1} | Score: {point.score:.4f} | Text: {point.payload['description']}")
    print("--------------------------------\n")

    # Extract the original text from the retrieved points
    retrieved_docs = [
        point.payload["description"]
        for point in results.points
    ]

    return retrieved_docs

def rerank(query: str, retrieved_docs: list[str]) -> list[str]:
    """
    Rerank a list of retrieved documents using a cross-encoder model.

    This function takes the user's query and the candidate documents from 
    dense retrieval, scores each pair simultaneously for deep semantic 
    relevance using a cross-encoder, and returns the documents sorted 
    from most relevant to least relevant.

    Args:
        query (str): The user's search query.
        retrieved_docs (list[str]): The candidate documents from dense retrieval.

    Returns:
        list[str]: The reordered documents based on cross-encoder scores.
    """
    # Initialize the cross-encoder reranker
    reranker = TextCrossEncoder(model_name="jinaai/jina-reranker-v2-base-multilingual")

    # Get relevance scores for each document
    scores = list(reranker.rerank(query, retrieved_docs))

    # Pair each document with its score
    scored_docs = list(zip(retrieved_docs, scores))

    # Sort the paired list by score in descending order
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Extract and return just the documents in the new sorted order
    reranked_docs = [doc for doc, score in scored_docs]

    return reranked_docs

def generate_answer(query: str, retrieved_docs: list[str]) -> str:
    """
    Generate an answer to a query based on retrieved documents using an LLM.

    Args:
        query (str): The user query.
        retrieved_docs (list[str]): The candidate documents from retrieval/reranking.

    Returns:
        str: The generated answer from the LLM.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    formatted_docs = "\n\n".join(f"[Document {i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs))
    system_content = (
        f"Use the following documents to answer the user question:\n\n{formatted_docs}\n\n"
        "If the answer cannot be found in the documents, respond with 'I didn't find any relevant documents.'"
    )

    ai_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer", 
                "content": system_content
            },
            {
                "role": "user", 
                "content": query
            }
        ]
    )

    return ai_answer.choices[0].message.content

# LLM-as-a-Judge Prompt
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
- relevant_docs = false if documents do not address the user question
- sufficient_context = false if documents are related but incomplete
- score should reflect overall answer quality and faithfulness
"""

# Function LLM-as-a-Judge
def llm_judge(query, retrieved_docs, answer):
    """
    Evaluate the answer using the retrieved documents.

    Args:
        query (str): The user query.
        retrieved_docs (list[str]): The retrieved documents.
        answer (str): The generated answer.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    prompt = llm_judge_prompt.format(query=query, retrieved_docs=retrieved_docs, answer=answer)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)