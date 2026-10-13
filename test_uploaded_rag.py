from pathlib import Path

from langchain_chroma import Chroma

from embeddings import get_embedding_model
from uploaded_pdf_generator import generate_answer


VECTORSTORE_PATH = Path(
    "vectorstores/uploaded/peca_2016"
)


def main() -> None:

    print("=" * 60)
    print("UPLOADED PDF RAG TEST")
    print("=" * 60)

    # 1. Load embeddings
    print("\n[1] Loading embedding model...")

    embeddings = get_embedding_model()

    # 2. Load uploaded PDF vector store
    print("[2] Loading vector store...")

    vectorstore = Chroma(
        persist_directory=str(
            VECTORSTORE_PATH
        ),
        embedding_function=embeddings,
    )

    # 3. User question
    question = "What is cyber stalking under PECA?"

    print("\n[3] Question:")
    print(question)

    # 4. Retrieve relevant chunks
    print("\n[4] Retrieving documents...")

    documents = vectorstore.similarity_search(
        question,
        k=3,
    )

    print(
        "Retrieved documents:",
        len(documents),
    )

    # 5. Generate answer
    print("\n[5] Generating answer with Gemini...")

    answer = generate_answer(
        question,
        documents,
    )

    # 6. Display result
    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print(
            f"{index}. "
            f"{document.metadata.get('source')} "
            f"— Page "
            f"{document.metadata.get('page_number')}"
        )


if __name__ == "__main__":
    main()