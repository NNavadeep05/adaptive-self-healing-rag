from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(pdf):
    """
    Load a PDF and split it into chunks for efficient retrieval.

    Args:
        pdf: Path to the PDF file.

    Returns:
        A list of text chunks.
    """

    # Load the PDF
    loader = PyPDFLoader(pdf, mode="single")
    docs = loader.load()

    # Split the document into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(docs)

    # Convert LangChain Document objects into plain strings
    chunks = [chunk.page_content for chunk in chunks]

    return chunks


if __name__ == "__main__":
    pdf = "./Guide_AB_Testing.pdf"
    chunks = load_document(pdf)

    print(f"Generated {len(chunks)} chunks from the PDF file.")
    print(chunks)