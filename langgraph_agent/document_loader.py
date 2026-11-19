import os
import tempfile

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph_agent.metadata import chunk_header, filing_metadata
from langgraph_agent.table_text import html_to_text


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _legacy_html_text(html: str) -> str:
    """v1 parsing: every element (including each table cell) on its own line."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "ix:header", "ix:hidden"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _read_text(file_path: str, table_aware: bool) -> str:
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        docs = PyPDFLoader(file_path, mode="single").load()
        return "\n".join(doc.page_content for doc in docs)
    if extension in (".html", ".htm"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        return html_to_text(html) if table_aware else _legacy_html_text(html)
    raise ValueError(f"Unsupported file type: {extension}")


def load_documents(
    files,
    display_names: dict[str, str] | None = None,
    table_aware: bool = True,
    contextual_headers: bool = True,
):
    """
    Load local PDF or HTML files and split them into chunks.

    Args:
        files: Local PDF or HTML file paths.
        display_names: Optional path -> name to record as the source
            (e.g. the original upload name instead of a temp file path).
        table_aware: Keep each HTML table row on one line (v2). False
            reproduces the original v1 parsing.
        contextual_headers: Prefix each chunk with its company / fiscal
            year header (v2), so the chunk is identifiable on its own.

    Returns:
        A list of chunk dicts: text, source_pdf, company, fiscal_year,
        chunk_index.
    """
    display_names = display_names or {}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for file_path in files:
        source = display_names.get(file_path, os.path.basename(file_path))
        metadata = filing_metadata(os.path.basename(source))
        header = chunk_header(metadata, source)
        text = _read_text(file_path, table_aware)

        for index, piece in enumerate(splitter.split_text(text)):
            chunks.append({
                "text": f"{header}\n{piece}" if contextual_headers else piece,
                "source_pdf": source,
                "chunk_index": index,
                **metadata,
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
        return load_documents([tmp_path], display_names={tmp_path: url})

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
