from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    """Test retrieval from the PECA vector store."""

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Loading Chroma vector store...")

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    query = "What is cyber stalking under PECA?"

    print("-" * 50)
    print("Query:", query)
    print("-" * 50)

    documents = retriever.invoke(query)

    print("Retrieved documents:", len(documents))
    print()

    for index, document in enumerate(documents, start=1):
        print(f"Result {index}")
        print("Section:", document.metadata.get("section_number"))
        print("Heading:", document.metadata.get("heading"))
        print("Content:")
        print(document.page_content[:500])
        print("-" * 50)


if __name__ == "__main__":
    main()