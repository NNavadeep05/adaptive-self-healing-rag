# Adaptive Self-Healing RAG

## Overview
Adaptive Self-Healing RAG is an intelligent document Q&A application built with Python, LangGraph, and Streamlit. 

In a standard Retrieval-Augmented Generation (RAG) system, if the initial document retrieval pulls weak or irrelevant context, the Large Language Model (LLM) blindly generates a weak or hallucinated answer. 

**Our system solves this by introducing a feedback loop.** It evaluates the quality of the retrieved context and the generated answer. If the result is poor, it dynamically "heals" itself by adjusting the retrieval budget, enabling semantic reranking, and trying again before presenting the final answer to the user.

## How It Works

```text
PDF
 ↓
Retrieve
 ↓
Rerank
 ↓
Generate Answer
 ↓
Evaluate
 ↓
Good → Return Answer
 ↓
Not Good
 ↓
Retry Retrieval
 ↓
Generate Again
```

## Key Features
- **PDF-based question answering:** Upload your own documents to query.
- **Dense vector retrieval with Qdrant:** Fast semantic search over embedded chunks.
- **Cross-encoder reranking:** Deeper semantic re-ordering when simple retrieval fails.
- **LLM-based answer generation:** Powered by OpenAI's GPT-4o.
- **LLM-based answer evaluation:** Powered by OpenAI's GPT-4o-mini acting as a judge.
- **Conditional self-healing retrieval:** LangGraph orchestration loops retrieval upon failure.
- **Vector-store reuse across retries:** Prevents redundant and expensive re-embedding.
- **Streamlit interface:** Clean, interactive UI.
- **Healing trace / evaluation score:** Transparently shows why and how the system healed itself.

## Project Structure
```text
adaptive-self-healing-rag/
├── app.py                           # Streamlit UI frontend
├── requirements.txt                 # Project dependencies
└── langgraph_agent/
    ├── document_loader.py           # PyPDF text extraction and chunking
    ├── retrieve_docs.py             # Qdrant embedding, reranking, and LLM calls
    ├── nodes.py                     # LangGraph node definitions and state mutation
    ├── graph.py                     # LangGraph cyclic workflow compilation
    └── __init__.py                  # Python package marker
```

## How to Run

These instructions assume you are using **Windows** and **PowerShell** with **Python 3.11+**.

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/NNavadeep05/adaptive-self-healing-rag.git
   cd adaptive-self-healing-rag
   ```

2. **Create a virtual environment:**
   ```powershell
   python -m venv .venv
   ```

3. **Activate it:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Set your OpenAI API key:**
   ```powershell
   $env:OPENAI_API_KEY="your_api_key_here"
   ```

6. **Run Streamlit:**
   ```powershell
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit application in your browser.
2. Upload a PDF.
3. Enter a question regarding the contents of the PDF.
4. Click **Run RAG**.
5. View the generated answer.
6. View the judge's evaluation score and the system's healing trace to see how many retries were required.

## What makes it self-healing?

The system does not blindly accept the first retrieval result. 

After generating an answer, an LLM judge evaluates whether the retrieved documents are relevant/sufficient and whether the answer quality is acceptable. 

If the result fails the retry condition, LangGraph sends the workflow back through retrieval with an increased retrieval budget and reranking enabled. This gives the system a simple feedback loop instead of a fragile, single-shot RAG pipeline.

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
          Cross-Encoder Reranking
                     │
                     ▼
             GPT-4o Generation
                     │
                     ▼
          GPT-4o-mini LLM Judge
                     │
                     ▼
              Evaluation
                     │
              ┌──────┴──────┐
              │             │
            Good          Not Good
              │             │
              ▼             ▼
          Final Answer    Retry
                            │
                            ▼
                   Increased Retrieval
                     Budget + Reranking
                            │
                            └──────► Retrieval
```
