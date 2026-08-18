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

## Implementation Improvements

- **Reuse vector store during retries:** The vector store is kept in LangGraph state so retries do not recreate the entire embedding store.
- **More reliable judge output:** The LLM judge uses enforced JSON response formatting to prevent parsing crashes.
- **Stronger judge model:** The evaluation judge was upgraded to `gpt-4o-mini` to match generation intelligence.
- **Cleaner document context:** Retrieved documents are clearly separated by distinct headers before being passed to the answer generator.

## Limitations / Future Work
- **Persistent vector storage:** Storing Qdrant data to disk to avoid re-embedding on system restarts.
- **Query rewriting:** Rather than just increasing retrieval count, using an LLM to rewrite the user's failed query before searching again.
- **Hybrid dense + keyword retrieval:** Fusing BM25 search with dense embeddings for better baseline retrieval.
- **More systematic evaluation:** Integrating rigorous RAG evaluation frameworks like RAGAS to track quantitative improvements across datasets.
