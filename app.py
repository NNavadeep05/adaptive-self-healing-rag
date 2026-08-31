import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langgraph_agent.document_loader import load_document
from langgraph_agent.graph import build_graph

# Load environment variables from .env file
load_dotenv()

# Configure page
st.set_page_config(page_title="Adaptive Self-Healing RAG", layout="centered")

st.title("Adaptive Self-Healing RAG")
st.markdown("""
This system automatically:
1. Retrieves relevant context from your PDF.
2. Generates an answer using Groq LLM.
3. Evaluates the answer using an LLM Judge.
4. **Self-Heals**: If the context is insufficient or irrelevant, it will dynamically adjust retrieval strategies and retry.
""")

# Check for API key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.warning("⚠️ GROQ_API_KEY is not set in your environment. The system can start, but you won't be able to run queries.")

# Sidebar or main content for file upload
st.header("1. Upload Document")
uploaded_file = st.file_uploader("Upload a PDF to query", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load and chunk the document
        with st.spinner("Loading and chunking document..."):
            chunks = load_document(tmp_path)
            st.success(f"Document loaded successfully! Extracted {len(chunks)} chunks.")

        st.header("2. Ask a Question")
        query = st.text_input("Enter your question about the document:")
        
        if st.button("Run RAG"):
            if not query.strip():
                st.error("Please enter a question.")
            elif not api_key:
                st.error("GROQ_API_KEY is missing. Cannot execute the RAG pipeline.")
            else:
                with st.spinner("Running Adaptive RAG Pipeline..."):
                    # Build the graph
                    app_graph = build_graph()
                    
                    # Initialize the graph state
                    initial_state = {
                        "text": chunks,
                        "query": query,
                        "retrieval_mode": "original",
                        "retrieval_budget": 5,
                        "retrieved_docs": [],
                        "answer": "",
                        "score": 0.0,
                        "failure_reason": "none",
                        "retry_count": 0,
                        "max_retries": 3,
                        "healing_trace": []
                    }
                    
                    try:
                        # Run the graph
                        result = app_graph.invoke(initial_state)
                        
                        st.header("3. Results")
                        
                        # Display Final Answer
                        st.subheader("Generated Answer")
                        st.info(result.get("answer", "No answer generated."))
                        
                        # Display Score
                        st.subheader("Judge Evaluation")
                        score = result.get("score", 0.0)
                        if score >= 0.8:
                            st.success(f"Score: {score:.2f} (Passed)")
                        else:
                            st.warning(f"Score: {score:.2f} (Failed)")
                            
                        # Display Healing Trace
                        st.subheader("Self-Healing Trace")
                        trace = result.get("healing_trace", [])
                        if trace:
                            for step in trace:
                                st.write(f"- {step}")
                        else:
                            st.write("- No healing actions taken.")
                            
                        # Optionally Display Debug Info
                        with st.expander("View Execution Details"):
                            st.write("**Final Retrieval Mode:**", result.get("retrieval_mode"))
                            st.write("**Final Retrieval Budget:**", result.get("retrieval_budget"))
                            st.write("**Total Retries:**", result.get("retry_count"))
                            
                    except Exception as e:
                        st.error(f"An error occurred during execution: {str(e)}")
                        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
