# Adaptive Self-Healing RAG

**Author:** Navadeep Nandedapu

A Retrieval-Augmented Generation (RAG) service that evaluates its own retrieval quality at runtime and attempts to recover from failed retrievals — without requiring a manual retry or a redeployment.

Built with **Python, LangGraph, Qdrant, FastEmbed, Groq, FastAPI, Docker, and Streamlit**, the system exposes both a **REST API** for programmatic/service-to-service integration and a **Streamlit UI** for interactive use, with the backend shipped as a **Docker container** for consistent deployment.

---

## Why It's Different

Most RAG pipelines are single-shot: retrieve once, generate once, return whatever comes out — even if the retrieved context was irrelevant or incomplete.

This system treats retrieval as a step that can fail, and attempts to recover from that failure automatically:

1. Retrieve context and generate an answer.
2. An LLM judge scores the retrieved context and the generated answer for relevance and sufficiency.
3. If the result is inadequate, the workflow **attempts to recover** — it re-enters retrieval with cross-encoder reranking enabled and a larger retrieval budget, then regenerates. This retry is bounded by a maximum retry count; if retries are exhausted, the workflow still returns the current result rather than retrying indefinitely.
4. The vector store is reused across retries, so no document is re-embedded mid-request.

The result is a service that attempts to degrade gracefully instead of silently returning a wrong answer — which is the behavior you want from something running behind an API, not just a notebook.

---

## Architecture

```text
                         PDF
                          │
                          ▼
                  Document Loading
                          │
                          ▼
                   Dense Embedding
                          │
                          ▼
                    Qdrant Store
                          │
                          ▼
                  Dense Retrieval
                          │
                          ▼
                  Answer Generation
                          │
                          ▼
                     LLM Judge
                          │
                          ▼
                     Evaluation
                          │
                   ┌──────┴──────┐
                   │             │
                 Good          Not Good
                   │             │
                   ▼             ▼
             Final Answer     Self-Healing
                                  │
                                  ▼
                         Increase Retrieval
                                  │
                                  ▼
                       Enable Cross-Encoder
                           Reranking
                                  │
                                  └──────► Retrieval
```

This cyclic control flow — retrieve → judge → conditionally re-route — is implemented as a **LangGraph** graph, which is what allows retries to be a first-class part of the workflow rather than an ad-hoc `try/except` around the pipeline.

---

## Key Capabilities

| Capability | Description |
|---|---|
| **PDF-based Q&A** | Upload a PDF and query its contents. |
| **Dense vector retrieval** | Qdrant-backed retrieval using cosine similarity over embedded chunks. |
| **Cross-encoder reranking** | Applied selectively via `jinaai/jina-reranker-v2-base-multilingual`, only when initial retrieval is judged insufficient — keeps normal-path latency low. |
| **LLM-as-a-judge evaluation** | A separate LLM call (`gpt-oss-20b` via Groq) scores retrieved context and generated answers for relevance, sufficiency, and overall quality. |
| **Self-healing control flow** | LangGraph detects retrieval failure and re-routes the workflow automatically, up to a fixed retry limit. |
| **Adaptive retrieval budget** | Retrieval count increases on retry instead of using a fixed top-k for every query. |
| **Vector-store reuse** | The in-memory Qdrant store persists across retries — no redundant re-embedding. |
| **REST API (FastAPI)** | `GET /health` and `POST /query` for integration into other services or pipelines. |
| **Dockerized backend** | Consistent, portable deployment across environments. |
| **Streamlit UI** | Interactive interface with visibility into judge scores and healing traces. |
| **Execution tracing** | Every response reports final retrieval mode, retrieval budget, and retry count — useful for debugging and understanding recovery behavior. |

---

## Project Structure

```text
adaptive-self-healing-rag/
├── api.py                           # FastAPI backend: GET /health, POST /query
├── app.py                           # Streamlit UI frontend
├── requirements.txt                 # Project dependencies
├── Dockerfile                       # Container build for the backend
└── langgraph_agent/
    ├── document_loader.py           # PDF text extraction and chunking
    ├── retrieve_docs.py             # Embedding, retrieval, reranking, and LLM calls
    ├── nodes.py                     # LangGraph nodes and self-healing logic
    ├── graph.py                     # LangGraph cyclic workflow compilation
    └── __init__.py                  # Python package marker
```

---

## Getting Started

These instructions assume **Windows**, **PowerShell**, and **Python 3.11+**. Docker instructions are OS-agnostic.

### 1. Clone the repository

```powershell
git clone https://github.com/NNavadeep05/adaptive-self-healing-rag.git
cd adaptive-self-healing-rag
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

`.env` is git-ignored and should never be committed.

### 3. Run locally (Streamlit)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### 4. Run via Docker (FastAPI backend)

```powershell
docker build -t adaptive-self-healing-rag .
docker run --env-file .env -p 8000:8000 adaptive-self-healing-rag
```

The container runs the FastAPI backend (`api.py`) via Uvicorn on port 8000. The Streamlit UI is not started inside the container — it runs separately using `streamlit run app.py`.

---

## Usage

### Streamlit UI

1. Open the Streamlit app in your browser.
2. Upload a PDF document.
3. Enter a question about the document.
4. Click **Run RAG**.
5. Review the generated answer and LLM judge score.
6. Check the **Self-Healing Trace** for any retrieval recovery steps.
7. Expand **View Execution Details** for the final retrieval mode, retrieval budget, and retry count.

### REST API

Once running (locally or via Docker), the FastAPI service exposes:

- **`GET /health`** — returns service status.
- **`POST /query`** — accepts a PDF and a question, returns the answer, judge score, and execution trace.

Both endpoints return the same answer, judge score, and execution trace surfaced in the Streamlit UI — suitable for integration into other applications or automated pipelines.

---

## How Self-Healing Works

The system does not accept the first retrieval result unconditionally.

After retrieval and answer generation, an LLM judge evaluates whether the retrieved documents are relevant and whether the context is sufficient to answer the question. If not, LangGraph routes the workflow back to retrieval — enabling cross-encoder reranking and increasing the retrieval count before generating again. The vector store is reused throughout, so retries add latency but not redundant embedding cost.

**Models used:**

- Embedding: FastEmbed, `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual`
- Generation: `openai/gpt-oss-120b` via Groq
- Judge: `openai/gpt-oss-20b` via Groq

**Retry behavior:**

- Initial retrieval budget: 2
- Retry triggered when the judge's score is below 0.8, or a failure reason is present
- `irrelevant_docs` → retrieval budget increases by 2, reranking enabled
- `missing_context` → retrieval budget increases by 3, reranking enabled
- Maximum retries: 3 — once reached, the workflow returns the current result rather than retrying further

**Example trace of a recovered query:**

```text
Initial Retrieval
      ↓
Irrelevant Context
      ↓
LLM Judge Detects Failure
      ↓
Self-Healing Triggered
      ↓
Reranking Enabled
      ↓
Retrieval Count Increased
      ↓
Generate Again
      ↓
Correct Answer
```

**Resulting execution details:**

```text
Final Retrieval Mode: dense_rerank
Final Retrieval Budget: 4
Total Retries: 1
```

This gives callers — whether a human in the Streamlit UI or a service calling the API — visibility into not just the final answer, but how the system arrived at it.