# Adaptive Self-Healing RAG

**Author:** Navadeep Nandedapu

## Overview

Adaptive Self-Healing RAG is a document Q&A application built with Python, LangGraph, Qdrant, FastEmbed, Groq, and Streamlit.

Unlike a single-shot RAG pipeline, the system evaluates the retrieved context and generated answer before returning the result. If the retrieved documents are irrelevant or insufficient, the LangGraph workflow automatically retries retrieval with reranking enabled and an increased retrieval budget.

## How It Works

```text
PDF
 ↓
Document Loading
 ↓
Dense Embedding
 ↓
Qdrant Vector Store
 ↓
Dense Retrieval
 ↓
Generate Answer
 ↓
LLM Judge
 ↓
Good → Return Answer
 ↓
Not Good
 ↓
Enable Reranking
 ↓
Increase Retrieval
 ↓
Generate Again
```

## Key Features

* **PDF-based question answering:** Upload a PDF and ask questions about its contents.
* **Dense vector retrieval with Qdrant:** Stores document embeddings and retrieves relevant chunks using cosine similarity.
* **Cross-encoder reranking:** Reorders retrieved documents using deeper query-document relevance scoring when retrieval fails.
* **LLM-based answer generation:** Uses Groq's OpenAI-compatible API for answer generation.
* **LLM-as-a-judge evaluation:** Evaluates retrieved context and generated answers for relevance, sufficiency, and overall quality.
* **Conditional self-healing:** LangGraph detects retrieval failures and routes the workflow back through retrieval.
* **Adaptive retrieval:** Failed retrieval can trigger reranking and an increased retrieval count.
* **Vector-store reuse across retries:** Reuses the in-memory Qdrant store instead of re-embedding the document for every retry.
* **Streamlit interface:** Provides an interactive PDF Q&A interface with evaluation results and healing traces.
* **Execution trace:** Displays the final retrieval mode, retrieval budget, and number of retries.

## Project Structure

```text
adaptive-self-healing-rag/
├── app.py                           # Streamlit UI frontend
├── requirements.txt                 # Project dependencies
└── langgraph_agent/
    ├── document_loader.py           # PDF text extraction and chunking
    ├── retrieve_docs.py             # Embedding, retrieval, reranking, and LLM calls
    ├── nodes.py                     # LangGraph nodes and self-healing logic
    ├── graph.py                     # LangGraph cyclic workflow compilation
    └── __init__.py                  # Python package marker
```

## How to Run

These instructions assume **Windows**, **PowerShell**, and **Python 3.11+**.

### 1. Clone the repository

```powershell
git clone https://github.com/NNavadeep05/adaptive-self-healing-rag.git
cd adaptive-self-healing-rag
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Create the `.env` file

Create a file named `.env` in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

The `.env` file is ignored by Git and should never be committed to the repository.

### 6. Run the Streamlit application

```powershell
streamlit run app.py
```

## Usage

1. Open the Streamlit application in your browser.
2. Upload a PDF document.
3. Enter a question about the document.
4. Click **Run RAG**.
5. View the generated answer.
6. View the LLM judge score.
7. View the **Self-Healing Trace** to see whether retrieval recovery was triggered.
8. Open **View Execution Details** to inspect the final retrieval mode, retrieval budget, and retry count.

## What Makes It Self-Healing?

The system does not blindly accept the first retrieval result.

After retrieval and answer generation, an LLM judge evaluates whether the retrieved documents are relevant and whether the context is sufficient for answering the question.

If the result is inadequate, LangGraph routes the workflow back to retrieval. The system can enable cross-encoder reranking and increase the number of retrieved documents before generating the answer again.

The vector store is reused across retries, avoiding unnecessary re-embedding of the uploaded document.

For example:

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

## Example Self-Healing Execution

A failed initial retrieval can produce a trace such as:

```text
Irrelevant docs → enabled rerank + increased retrieval budget by 2
```

The execution details then show:

```text
Final Retrieval Mode: dense_rerank
Final Retrieval Budget: 4
Total Retries: 1
```

This allows the user to see not only the final answer, but also how the system recovered from an unsuccessful initial retrieval.