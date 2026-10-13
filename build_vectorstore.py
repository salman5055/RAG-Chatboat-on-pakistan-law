import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHUNKS_PATH = Path("data/law_chunks.json")
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load legal chunks from the JSON file."""
    with chunks_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_documents(chunks: list[dict]) -> list[Document]:
    """Convert legal chunks into LangChain Document objects."""
    documents = []

    for chunk in chunks:
        metadata = {
            "section_number": chunk["section_number"],
            "heading": chunk["heading"],
        }

        if "chunk_number" in chunk:
            metadata["chunk_number"] = chunk["chunk_number"]

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

    return documents


def main() -> None:
    """Create and persist the PECA Chroma vector store."""

    print("Loading legal chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    print("Total chunks:", len(chunks))

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Creating Chroma vector store...")

    documents = create_documents(chunks)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH,
    )

    print("-" * 50)
    print("Vector store created successfully!")
    print("Documents stored:", len(documents))
    print("Location:", VECTORSTORE_PATH)


if __name__ == "__main__":
    main()