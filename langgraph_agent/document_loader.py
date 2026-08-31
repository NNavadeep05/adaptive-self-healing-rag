import os
import tempfile

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(files):
    """
    Load multiple local PDF or HTML files and split them into chunks.

    Args:
        files: List of local PDF or HTML file paths.

    Returns:
        A list of dictionaries containing chunk text and source file.
    """

    all_docs = []

    for file_path in files:
        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            loader = PyPDFLoader(file_path, mode="single")
            docs = loader.load()

            for doc in docs:
                all_docs.append({
                    "text": doc.page_content,
                    "source_pdf": os.path.basename(file_path)
                })

        elif extension in [".html", ".htm"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()

            soup = BeautifulSoup(html, "html.parser")

            # Remove HTML elements that do not contain useful document text.
            for tag in soup(["script", "style", "noscript", "ix:header", "ix:hidden"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            all_docs.append({
                "text": text,
                "source_pdf": os.path.basename(file_path)
            })

        else:
            raise ValueError(f"Unsupported file type: {extension}")

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