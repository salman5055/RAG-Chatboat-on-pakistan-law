from pathlib import Path

from langchain_chroma import Chroma

from embeddings import get_embedding_model
from uploaded_pdf_generator import generate_answer
from relevance_scorer import calculate_relevance_score


VECTORSTORE_PATH = Path(
    "vectorstores/uploaded/peca_2016"
)

RELEVANCE_THRESHOLD = 95.0


def main() -> None:

    print("=" * 60)
    print("UPLOADED PDF RAG + RELEVANCE TEST")
    print("=" * 60)

    # 1. Load embedding model
    print("\n[1] Loading embedding model...")

    embeddings = get_embedding_model()

    # 2. Load vector store
    print("[2] Loading vector store...")

    vectorstore = Chroma(
        persist_directory=str(
            VECTORSTORE_PATH
        ),
        embedding_function=embeddings,
    )

    # 3. Question
    question = "What is cyber stalking under PECA?"

    print("\n[3] Question:")
    print(question)

    # 4. Retrieval
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
    print("\n[5] Generating answer...")

    answer = generate_answer(
        question,
        documents,
    )

    # 6. Calculate relevance
    print("\n[6] Calculating relevance score...")

    relevance_score = calculate_relevance_score(
        answer,
        documents,
    )

    # 7. Determine status
    if relevance_score >= RELEVANCE_THRESHOLD:
        status = "PASS"
    else:
        status = "BELOW THRESHOLD"

    # 8. Display answer
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    # 9. Display sources
    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page_number",
            "Unknown",
        )

        print(
            f"{index}. {source} — Page {page}"
        )

    # 10. Display relevance
    print("\n" + "=" * 60)
    print("RELEVANCE EVALUATION")
    print("=" * 60)

    print(
        f"Semantic relevance: "
        f"{relevance_score}%"
    )

    print(
        f"Required threshold: "
        f"{RELEVANCE_THRESHOLD}%"
    )

    print(
        f"Status: {status}"
    )


if __name__ == "__main__":
    main()