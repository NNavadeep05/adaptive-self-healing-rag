import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langgraph_agent.document_loader import load_documents
from langgraph_agent.graph import APP_STRATEGY, build_graph, make_initial_state
from langgraph_agent.retrieve_docs import COLLECTION_NAME, embed_docs, get_qdrant_client

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Adaptive Self-Healing RAG", layout="centered")

st.title("Adaptive Self-Healing RAG")
st.markdown("""
This system:
1. Ingests documents into a persistent Qdrant knowledge base.
2. Retrieves relevant context and generates an answer with a Groq-hosted LLM.
3. Verifies the answer against the context and scores it with an LLM judge.
4. **Self-heals**: if the checks fail, it changes its retrieval strategy and retries.
""")

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.warning("GROQ_API_KEY is not set. You can ingest documents but not run queries.")

if "ingested_hashes" not in st.session_state:
    st.session_state.ingested_hashes = set()


def knowledge_base_size() -> int:
    """Return the number of chunks stored, or 0 if the collection is missing."""
    client = get_qdrant_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            return 0
        return client.count(collection_name=COLLECTION_NAME).count
    finally:
        client.close()


def ingest_uploaded_pdf(uploaded_file) -> int:
    """Chunk an uploaded PDF, embed it into Qdrant and return the chunk count."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Record the user's file name as the source instead of the temp path.
        chunks = load_documents([tmp_path], display_names={tmp_path: uploaded_file.name})
    finally:
        os.remove(tmp_path)

    if not chunks:
        return 0

    embed_docs(chunks).close()
    return len(chunks)


# ---------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------

st.header("1. Add a Document (optional)")
uploaded_file = st.file_uploader("Upload a PDF to add to the knowledge base", type=["pdf"])

if uploaded_file is not None:
    file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

    if file_hash in st.session_state.ingested_hashes:
        st.info(f"'{uploaded_file.name}' is already in the knowledge base.")
    elif st.button("Ingest document"):
        try:
            with st.spinner("Loading, chunking and embedding..."):
                count = ingest_uploaded_pdf(uploaded_file)
            if count:
                st.session_state.ingested_hashes.add(file_hash)
                st.success(f"Ingested {count} chunks from '{uploaded_file.name}'.")
            else:
                st.error("No readable text was found in the PDF.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

# ---------------------------------------------------------
# 2. Querying
# ---------------------------------------------------------

st.header("2. Ask a Question")

try:
    stored_chunks = knowledge_base_size()
except Exception as e:
    stored_chunks = 0
    st.error(f"Could not reach Qdrant: {e}")

st.caption(f"Knowledge base: {stored_chunks} chunks in collection '{COLLECTION_NAME}'.")

query = st.text_input("Enter your question:")

if st.button("Run RAG"):
    if not query.strip():
        st.error("Please enter a question.")
    elif not api_key:
        st.error("GROQ_API_KEY is missing. Cannot execute the RAG pipeline.")
    elif stored_chunks == 0:
        st.error("The knowledge base is empty. Ingest a document first.")
    else:
        try:
            with st.spinner("Running adaptive RAG pipeline..."):
                result = build_graph().invoke(make_initial_state(query, strategy=APP_STRATEGY))
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
        else:
            st.header("3. Results")

            st.subheader("Generated Answer")
            st.info(result.get("answer", "No answer generated."))

            st.subheader("Judge Evaluation")
            score = result.get("score", 0.0)
            if score >= 0.8:
                st.success(f"Score: {score:.2f} (Passed)")
            else:
                st.warning(f"Score: {score:.2f} (Failed)")

            st.subheader("Self-Healing Trace")
            trace = result.get("healing_trace", [])
            if trace:
                for step in trace:
                    st.write(f"- {step}")
            else:
                st.write("- No healing actions taken.")

            with st.expander("View Execution Details"):
                st.write("**Verification:**", result.get("verification_status"))
                st.write("**Final Retrieval Mode:**", result.get("retrieval_mode"))
                st.write("**Final Retrieval Budget:**", result.get("retrieval_budget"))
                st.write("**Total Retries:**", result.get("retry_count"))
