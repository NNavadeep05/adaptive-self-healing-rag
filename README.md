# SentinelRAG | Adaptive Self-Healing RAG System

A Retrieval-Augmented Generation service that evaluates the quality of its own output at query time and automatically recovers when it detects a bad result — instead of silently returning an answer built on irrelevant context, missing information, or unsupported claims.

**Author:** Navadeep Nandedapu

---

## The Problem

Organizations store information across many documents and sources, and finding an answer usually means manually searching through them. A standard RAG pipeline automates the search, but it is single-shot: it retrieves once, generates once, and returns whatever comes out, even when the retrieved context was irrelevant, incomplete, or when the generated answer drifts from what the context actually supports. The user has no way to know the answer was built on bad context, and the system has no way to try again.

This project treats retrieval and generation as steps that can fail, and gives the system a way to notice and recover.

## How It Works

1. **Retrieve** relevant chunks for the query and **generate** an answer from them.
2. A **self-verification stage** checks whether the generated answer is actually supported by the retrieved context, flagging unsupported or fabricated claims.
3. An **LLM judge** independently evaluates whether the retrieved context was relevant and sufficient, and scores the answer.
4. If either stage finds the result inadequate, the workflow **routes back to retrieval** with a larger retrieval budget and cross-encoder reranking enabled, then generates, verifies, and judges again.
5. Retries are capped by a maximum retry count, so a genuinely difficult query fails gracefully instead of looping forever.
6. The vector store is **reused across retries** — documents are embedded once at ingestion time, not re-embedded on every retry.

```
              PDF / External Source
                       │
                       ▼
               Document Loading
                       │
                       ▼
                   Chunking
                       │
                       ▼
               Dense Embedding
                       │
                       ▼
               Qdrant Vector DB
                       │
                       ▼
               Dense Retrieval
                       │
                       ▼
               Answer Generation
                       │
                       ▼
               Self-Verification
                       │
                       ▼
                  LLM Judge
                       │
              ┌────────┴────────┐
              │                 │
            Good             Not Good
              │                 │
              ▼                 ▼
        Final Answer      Self-Healing
                                 │
                                 ▼
                    Increase Retrieval Budget
                                 │
                                 ▼
                    Enable Cross-Encoder
                         Reranking
                                 │
                                 └──────► Retrieve Again
```

The retrieve → generate → verify → judge → conditionally retry loop is implemented as an explicit graph using **LangGraph**, so the recovery behavior is part of the application's control flow rather than a manual retry wrapped around it.

Verification runs after generation rather than before retrieval, since checking whether a claim is supported requires a claim to check — it can only happen once an answer exists.

### Retry logic

A retry is triggered when:

```
score < 0.8
OR
failure_reason != "none"
OR
verification_status == "unsupported"
```

| Failure reason | Recovery action |
|---|---|
| `unsupported_claims` | Increase retrieval budget by 2, enable reranking |
| `irrelevant_docs` | Increase retrieval budget by 2, enable reranking |
| `missing_context` | Increase retrieval budget by 3, enable reranking |
| Max retries reached | Stop retrying, return the current result |

## Features

- **Multi-document ingestion** — loads and chunks multiple PDFs, from local files or external URLs
- **Source-aware chunks** — every chunk retains its originating document or URL
- **Dense vector retrieval** over a persistent Qdrant collection
- **Cross-encoder reranking**, applied selectively during recovery rather than on every query
- **Self-verification** — checks whether generated claims are actually supported by the retrieved context
- **LLM-as-judge evaluation** of retrieved context relevance and sufficiency
- **Self-healing control flow** with a bounded retry loop, triggered by either verification or judge failures
- **REST API** (`/health`, `/ingest`, `/query`) for service-to-service use
- **Streamlit UI** for interactive ingestion and querying
- **Execution tracing** — every response reports retrieval mode, retrieval budget, retry count, and the healing actions taken
- **Dockerized backend**, running against Qdrant as a separate service over a Docker network

## Architecture

| Component | Role |
|---|---|
| Document Loader | Loads PDFs from local paths or URLs, chunks them, retains source metadata |
| FastEmbed | Converts chunks into dense vector embeddings |
| Qdrant | Stores embeddings, performs vector similarity search |
| LangGraph | Controls retrieval, generation, verification, judging, and retry flow |
| Groq | Serves the generation, verification, and judge LLM calls |
| FastAPI | Exposes the backend over REST |
| Streamlit | Interactive frontend |
| Docker | Packages the backend into a reproducible container |

**Models:**

| Purpose | Model |
|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (via FastEmbed) |
| Reranking | `jinaai/jina-reranker-v2-base-multilingual` |
| Generation | `openai/gpt-oss-120b` (via Groq) |
| Self-verification | `openai/gpt-oss-20b` (via Groq) |
| Judge | `openai/gpt-oss-20b` (via Groq) |

Self-verification and judging are two independent checks that both use the same underlying model but evaluate different things: verification checks whether the answer's claims are supported by the retrieved context, while the judge evaluates the relevance and sufficiency of the retrieved context itself.

## Project Structure

```
adaptive-self-healing-rag/
├── api.py                    # FastAPI backend and REST endpoints
├── app.py                    # Streamlit UI
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── langgraph_agent/
    ├── document_loader.py    # PDF loading, URL download, chunking, metadata
    ├── retrieve_docs.py      # Qdrant client, embeddings, retrieval, reranking, verification, judge, generation
    ├── nodes.py               # LangGraph nodes and self-healing logic
    ├── graph.py                # LangGraph workflow construction
    └── __init__.py
```

## Getting Started

Requires Python 3.11+. Docker commands below use PowerShell syntax but are OS-agnostic in intent.

### 1. Clone and configure

```
git clone https://github.com/NNavadeep05/adaptive-self-healing-rag.git
cd adaptive-self-healing-rag
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://qdrant:6333
```

`.env` is git-ignored and should never be committed. For a purely local setup without Dockerized Qdrant, point `QDRANT_URL` at your locally running Qdrant instance instead.

### 2. Option A — Local Streamlit + local backend

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Qdrant must be reachable at the configured `QDRANT_URL`.

### 3. Option B — Dockerized backend + Qdrant

```
docker network create rag-network

docker run -d `
  --name qdrant `
  --network rag-network `
  -p 6333:6333 -p 6334:6334 `
  qdrant/qdrant

docker build -t adaptive-self-healing-rag .

docker run -d `
  --name adaptive-self-healing-rag `
  --env-file .env `
  --network rag-network `
  -p 8000:8000 `
  adaptive-self-healing-rag
```

Inside the Docker network, the backend reaches Qdrant via its service name (`http://qdrant:6333`) — `localhost` inside the backend container refers to the backend itself, not Qdrant.

The Streamlit UI runs as a separate process (`streamlit run app.py`) regardless of which backend option is used.

## API Usage

**Health check**

```
GET /health
```

**Ingest a document**

```
POST /ingest
{ "source": "https://arxiv.org/pdf/2312.10997" }
```

Returns the number of chunks ingested and the total stored in the collection.

**Query**

```
POST /query
{ "question": "..." }
```

Returns (example of a healed response, after a retry):

```json
{
  "answer": "...",
  "score": 0.9,
  "verification_status": "supported",
  "verification_score": 0.9,
  "retrieval_mode": "dense_rerank",
  "retrieval_budget": 4,
  "retry_count": 1,
  "healing_trace": [
    "Unsupported claims → enabled reranking + increased retrieval budget by 2"
  ]
}
```

Exposing these fields makes the recovery process observable instead of hiding it behind the final answer.
