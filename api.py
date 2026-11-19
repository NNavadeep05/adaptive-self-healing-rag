import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from langgraph_agent.document_loader import load_pdf_from_url
from langgraph_agent.retrieve_docs import (
    COLLECTION_NAME,
    embed_docs,
    get_qdrant_client,
)
from langgraph_agent.graph import APP_STRATEGY, build_graph, make_initial_state


load_dotenv()


app = FastAPI(
    title="Adaptive Self-Healing RAG API",
    description="FastAPI backend for the Adaptive Self-Healing RAG system.",
    version="1.0.0",
)


class IngestRequest(BaseModel):
    url: str


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    """Check whether the API service is running."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest_document(request: IngestRequest):
    """
    Download a PDF from an external URL, process it,
    and persist its chunks and embeddings in Qdrant.
    """

    if not request.url.strip():
        raise HTTPException(
            status_code=400,
            detail="PDF URL cannot be empty."
        )

    try:
        # Download, parse, and chunk the external PDF
        chunks = load_pdf_from_url(
            request.url
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the PDF."
            )

        # Embed and persist the document chunks
        client = embed_docs(chunks)

        # Check how many chunks are now stored
        total_stored_chunks = client.count(
            collection_name=COLLECTION_NAME
        ).count

        # Close the ingestion client
        client.close()

        return {
            "status": "success",
            "source": request.url,
            "chunks_ingested": len(chunks),
            "total_stored_chunks": total_stored_chunks
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(e)}"
        )


@app.post("/query")
def query_documents(request: QueryRequest):
    """
    Query the existing persistent knowledge base.

    No PDF is uploaded or downloaded during querying.
    """

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    client = None

    try:
        # Client used only to check that the knowledge base exists.
        client = get_qdrant_client()
        # Make sure the knowledge base exists
        if not client.collection_exists(
            COLLECTION_NAME
        ):
            raise HTTPException(
                status_code=400,
                detail="Knowledge base is empty. Ingest a document first."
            )

        stored_count = client.count(
            collection_name=COLLECTION_NAME
        ).count

        if stored_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Knowledge base is empty. Ingest a document first."
            )

        # Build the self-healing LangGraph workflow
        app_graph = build_graph()

        # Run the complete self-healing RAG pipeline
        result = app_graph.invoke(
            make_initial_state(request.question, strategy=APP_STRATEGY)
        )

        return {
            "answer": result.get("answer", ""),
            "score": result.get("score", 0.0),
            "retrieval_strategy": result.get("retrieval_strategy"),
            "retrieval_mode": result.get("retrieval_mode"),
            "search_query": result.get("search_query"),
            "retrieval_budget": result.get("retrieval_budget"),
            "retry_count": result.get("retry_count"),
            "healing_trace": result.get(
                "healing_trace",
                []
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline execution failed: {str(e)}"
        )

    finally:
        if client is not None:
            client.close()
