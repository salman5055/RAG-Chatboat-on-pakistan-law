from pathlib import Path

from langchain_chroma import Chroma

from embeddings import get_embedding_model


VECTORSTORE_PATH = Path(
    "vectorstores/uploaded/peca_2016"
)


def main() -> None:

    print("=" * 50)
    print("UPLOADED PDF RETRIEVAL TEST")
    print("=" * 50)

    print("\nLoading embedding model...")

    embeddings = get_embedding_model()

    print("Embedding model loaded.")

    print("\nLoading uploaded PDF vector store...")

    vectorstore = Chroma(
        persist_directory=str(
            VECTORSTORE_PATH
        ),
        embedding_function=embeddings,
    )

    print("Vector store loaded.")

    question = "What is cyber stalking under PECA?"

    print("\nQuestion:")
    print(question)

    print("\nRetrieving relevant chunks...")

    documents = vectorstore.similarity_search(
        question,
        k=3,
    )

    print("\n" + "=" * 50)
    print("RETRIEVED DOCUMENTS")
    print("=" * 50)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print(f"\n--- Result {index} ---")

        print(
            "Source:",
            document.metadata.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Page:",
            document.metadata.get(
                "page_number",
                "Unknown",
            ),
        )

        print(
            "Chunk ID:",
            document.metadata.get(
                "chunk_id",
                "Unknown",
            ),
        )

        print("\nContent:")
        print(
            document.page_content[:500]
        )


if __name__ == "__main__":
    main()