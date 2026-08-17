from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding


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