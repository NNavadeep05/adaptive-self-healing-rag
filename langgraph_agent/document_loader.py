import os
import tempfile

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(pdfs):
    """
    Load multiple local PDFs and split them into chunks for retrieval.

    Args:
        pdfs: List of local PDF file paths.

    Returns:
        A list of dictionaries containing chunk text and source PDF.
    """

    all_docs = []

    for pdf in pdfs:
        loader = PyPDFLoader(pdf, mode="single")
        docs = loader.load()

        for doc in docs:
            all_docs.append({
                "text": doc.page_content,
                "source_pdf": os.path.basename(pdf)
            })

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []

    for doc in all_docs:
        split_texts = text_splitter.split_text(doc["text"])

        for chunk in split_texts:
            chunks.append({
                "text": chunk,
                "source_pdf": doc["source_pdf"]
            })

    return chunks


def download_pdf(url):
    """
    Download a PDF from an external URL and return its temporary path.
    """

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:
        tmp_file.write(response.content)

        return tmp_file.name


def load_pdf_from_url(url):
    """
    Download a PDF from an external URL and load it into chunks.

    Args:
        url: Public URL pointing to a PDF.

    Returns:
        A list of dictionaries containing chunk text and source URL.
    """

    tmp_path = download_pdf(url)

    try:
        chunks = load_documents([tmp_path])

        for chunk in chunks:
            chunk["source_pdf"] = url

        return chunks

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)