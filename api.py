import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from langgraph_agent.document_loader import load_document
from langgraph_agent.graph import build_graph


# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="Adaptive Self-Healing RAG API",
    description="FastAPI backend for the Adaptive Self-Healing RAG system.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """Check whether the API service is running."""
    return {"status": "ok"}


@app.post("/query")
async def query_document(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Upload a PDF and ask a question about its contents.

    The request is processed through the complete
    Adaptive Self-Healing RAG LangGraph pipeline.
    """

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    tmp_path = None

    try:
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        # Load and chunk the document
        chunks = load_document(tmp_path)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the PDF."
            )

        # Build the existing LangGraph RAG workflow
        app_graph = build_graph()

        # Initialize the same state used by the Streamlit application
        initial_state = {
            "text": chunks,
            "query": question,
            "retrieval_mode": "original",
            "retrieval_budget": 2,
            "retrieved_docs": [],
            "answer": "",
            "score": 0.0,
            "failure_reason": "none",
            "retry_count": 0,
            "max_retries": 3,
            "healing_trace": []
        }

        # Run the complete self-healing RAG pipeline
        result = app_graph.invoke(initial_state)

        # Return the final answer and execution information
        return {
            "answer": result.get("answer", ""),
            "score": result.get("score", 0.0),
            "retrieval_mode": result.get("retrieval_mode"),
            "retrieval_budget": result.get("retrieval_budget"),
            "retry_count": result.get("retry_count"),
            "healing_trace": result.get("healing_trace", [])
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline execution failed: {str(e)}"
        )

    finally:
        # Remove temporary PDF
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)