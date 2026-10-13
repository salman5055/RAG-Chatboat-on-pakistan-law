from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:

    print("=" * 50)
    print("PECA RETRIEVAL TEST")
    print("=" * 50)

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    print("\nLoading Chroma vector store...")

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
    )

    print("Vector store loaded.")

    question = "What is cyber stalking under PECA?"

    print("\nQuestion:")
    print(question)

    print("\nRetrieving relevant documents...")

    results = vectorstore.similarity_search(
        question,
        k=3,
    )

    print("\n" + "=" * 50)
    print("RETRIEVED DOCUMENTS")
    print("=" * 50)

    for index, document in enumerate(results, start=1):

        print(f"\n--- Result {index} ---")

        print(
            "Section:",
            document.metadata.get(
                "section_number",
                "Unknown"
            ),
        )

        print(
            "Heading:",
            document.metadata.get(
                "heading",
                "Unknown"
            ),
        )

        print("\nContent:")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()