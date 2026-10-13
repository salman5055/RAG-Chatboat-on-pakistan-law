import json
import time
from pathlib import Path

from langchain_chroma import Chroma
from embeddings import get_embedding_model


VECTORSTORE_PATH = Path("vectorstore")

EVALUATION_QUESTIONS = [
    {
        "question": "What is cyber stalking under PECA?",
        "expected_section": "24",
    },
    {
        "question": "What is cyber terrorism under PECA?",
        "expected_section": "10",
    },
    {
        "question": "What is unauthorized access to information system?",
        "expected_section": "3",
    },
    {
        "question": "What is unauthorized copying or transmission?",
        "expected_section": "4",
    },
    {
        "question": "What is interference with information system?",
        "expected_section": "5",
    },
    {
        "question": "What is electronic forgery?",
        "expected_section": "13",
    },
    {
        "question": "What is electronic fraud?",
        "expected_section": "14",
    },
    {
        "question": "What is identity information?",
        "expected_section": "16",
    },
    {
        "question": "What is unauthorized use of identity information?",
        "expected_section": "16",
    },
    {
        "question": "What is spoofing?",
        "expected_section": "26",
    },
    {
        "question": "What is spamming?",
        "expected_section": "25",
    },
    {
        "question": "What is cyber harassment?",
        "expected_section": "24",
    },
    {
        "question": "What is the punishment for cyber terrorism?",
        "expected_section": "10",
    },
    {
        "question": "What is the punishment for electronic fraud?",
        "expected_section": "14",
    },
    {
        "question": "What is the punishment for cyber stalking?",
        "expected_section": "24",
    },
]


def reciprocal_rank(expected_section: str, documents: list) -> float:
    """
    Calculate Reciprocal Rank for one question.

    If expected section is:
    Rank 1 -> 1.0
    Rank 2 -> 0.5
    Rank 3 -> 0.3333
    Not found -> 0.0
    """

    for rank, document in enumerate(documents, start=1):
        section = document.metadata.get("section_number")

        if section == expected_section:
            return 1 / rank

    return 0.0


def main() -> None:
    print("=" * 60)
    print("PECA RETRIEVAL QUALITY EVALUATION")
    print("=" * 60)

    print("\n[1] Loading embedding model...")
    embeddings = get_embedding_model()

    print("[2] Loading Chroma vector store...")
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_PATH),
        embedding_function=embeddings,
    )

    top1_correct = 0
    top3_correct = 0
    reciprocal_ranks = []
    retrieval_times = []

    results = []

    print("\n[3] Running retrieval evaluation...")
    print("-" * 60)

    for index, item in enumerate(EVALUATION_QUESTIONS, start=1):

        question = item["question"]
        expected_section = item["expected_section"]

        start_time = time.perf_counter()

        documents = vectorstore.similarity_search(
            question,
            k=3,
        )

        end_time = time.perf_counter()

        retrieval_time = end_time - start_time
        retrieval_times.append(retrieval_time)

        retrieved_sections = [
            document.metadata.get("section_number")
            for document in documents
        ]

        # Top-1
        top1_match = (
            len(retrieved_sections) > 0
            and retrieved_sections[0] == expected_section
        )

        # Top-3
        top3_match = expected_section in retrieved_sections

        if top1_match:
            top1_correct += 1

        if top3_match:
            top3_correct += 1

        rr = reciprocal_rank(
            expected_section,
            documents,
        )

        reciprocal_ranks.append(rr)

        if top1_match:
            status = "TOP-1 PASS"
        elif top3_match:
            status = "TOP-3 PASS"
        else:
            status = "FAIL"

        print(
            f"{index:02d}. {status} | "
            f"Expected: {expected_section} | "
            f"Retrieved: {retrieved_sections} | "
            f"Time: {retrieval_time:.4f}s"
        )

        results.append(
            {
                "question_number": index,
                "question": question,
                "expected_section": expected_section,
                "retrieved_sections": retrieved_sections,
                "top1_correct": top1_match,
                "top3_correct": top3_match,
                "reciprocal_rank": round(rr, 4),
                "retrieval_time_seconds": round(
                    retrieval_time,
                    4,
                ),
            }
        )

    total_questions = len(EVALUATION_QUESTIONS)

    top1_accuracy = (
        top1_correct / total_questions
    ) * 100

    top3_accuracy = (
        top3_correct / total_questions
    ) * 100

    mrr = (
        sum(reciprocal_ranks)
        / total_questions
    )

    average_retrieval_time = (
        sum(retrieval_times)
        / total_questions
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY RESULTS")
    print("=" * 60)

    print(
        f"Top-1 Accuracy : {top1_accuracy:.2f}%"
    )

    print(
        f"Top-3 Accuracy : {top3_accuracy:.2f}%"
    )

    print(
        f"MRR            : {mrr:.4f}"
    )

    print(
        f"Average Retrieval Time: "
        f"{average_retrieval_time:.4f}s"
    )

    output_path = Path(
        "evaluation_results/retrieval_quality.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "total_questions": total_questions,
        "top1_accuracy": round(
            top1_accuracy,
            2,
        ),
        "top3_accuracy": round(
            top3_accuracy,
            2,
        ),
        "mrr": round(
            mrr,
            4,
        ),
        "average_retrieval_time_seconds": round(
            average_retrieval_time,
            4,
        ),
        "results": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("Results saved to:")
    print(output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()