from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings import get_embedding_model
from uploaded_pdf_loader import extract_uploaded_pdf
from uploaded_pdf_chunker import chunk_uploaded_pdf


VECTORSTORE_ROOT = Path("vectorstores/uploaded")


def create_uploaded_vectorstore(
    pdf_path: str,
) -> Chroma:
    """
    Create a separate Chroma vector store for an uploaded PDF.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_file}"
        )

    print("Extracting PDF...")

    pages = extract_uploaded_pdf(
        str(pdf_file)
    )

    print("Pages extracted:", len(pages))

    print("Creating chunks...")

    chunks = chunk_uploaded_pdf(
        pages,
        source_name=pdf_file.name,
    )

    print("Chunks created:", len(chunks))

    documents = []

    for chunk in chunks:

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                },
            )
        )

    print("Documents prepared:", len(documents))

    print("Loading embedding model...")

    embeddings = get_embedding_model()

    # Create a unique folder for this PDF
    pdf_store_name = pdf_file.stem

    persist_directory = (
        VECTORSTORE_ROOT / pdf_store_name
    )

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Creating Chroma store at:",
        persist_directory,
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(
            persist_directory
        ),
    )

    print("-" * 50)
    print("UPLOADED PDF VECTOR STORE CREATED")
    print("-" * 50)

    print("PDF:", pdf_file.name)
    print("Documents:", len(documents))
    print("Vector store:", persist_directory)

    return vectorstore


def main() -> None:

    pdf_path = "data/peca_2016.pdf"

    create_uploaded_vectorstore(
        pdf_path
    )


if __name__ == "__main__":
    main()