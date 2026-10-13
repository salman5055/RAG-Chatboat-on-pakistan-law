import json
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RESULTS_PATH = Path("evaluation_results.json")


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
        "question": "What is unauthorized copying or transmission of data?",
        "expected_section": "4",
    },
    {
        "question": "What is interference with information system or data?",
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
        "question": "What is spoofing under PECA?",
        "expected_section": "26",
    },
    {
        "question": "What is spamming under PECA?",
        "expected_section": "25",
    },
    {
        "question": "What is cyber harassment under PECA?",
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


def load_retriever():
    """Load the PECA Chroma vector store."""

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Loading Chroma vector store...")

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


def evaluate_questions(retriever):
    """Evaluate retrieval quality."""

    results = []

    for index, item in enumerate(
        EVALUATION_QUESTIONS,
        start=1,
    ):
        question = item["question"]
        expected_section = item["expected_section"]

        start_time = time.perf_counter()

        documents = retriever.invoke(question)

        retrieval_time = time.perf_counter() - start_time

        retrieved_sections = [
            document.metadata.get(
                "section_number",
                "Unknown",
            )
            for document in documents
        ]

        passed = expected_section in retrieved_sections

        result = {
            "question_number": index,
            "question": question,
            "expected_section": expected_section,
            "retrieved_sections": retrieved_sections,
            "passed": passed,
            "retrieval_time_seconds": round(
                retrieval_time,
                4,
            ),
        }

        results.append(result)

        status = "PASS" if passed else "FAIL"

        print(
            f"{index:02d}. {status} | "
            f"Expected: Section {expected_section} | "
            f"Retrieved: {retrieved_sections} | "
            f"Time: {retrieval_time:.4f}s"
        )

    return results


def calculate_metrics(results):
    """Calculate evaluation metrics."""

    total = len(results)

    passed = sum(
        result["passed"]
        for result in results
    )

    failed = total - passed

    accuracy = (
        passed / total * 100
        if total > 0
        else 0
    )

    average_time = (
        sum(
            result["retrieval_time_seconds"]
            for result in results
        )
        / total
        if total > 0
        else 0
    )

    return {
        "total_questions": total,
        "passed": passed,
        "failed": failed,
        "retrieval_accuracy_percent": round(
            accuracy,
            2,
        ),
        "average_retrieval_time_seconds": round(
            average_time,
            4,
        ),
        "top_k": 3,
    }


def save_results(results, metrics):
    """Save evaluation results to JSON."""

    output = {
        "evaluation": {
            "description": (
                "PECA RAG retrieval evaluation"
            ),
            "embedding_model": EMBEDDING_MODEL,
            "vector_store": "Chroma",
            "results": results,
            "metrics": metrics,
        }
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print(
        "Evaluation results saved to:",
        RESULTS_PATH,
    )


def print_summary(metrics):
    """Print evaluation summary."""

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        "Total questions:",
        metrics["total_questions"],
    )

    print(
        "Passed:",
        metrics["passed"],
    )

    print(
        "Failed:",
        metrics["failed"],
    )

    print(
        "Retrieval accuracy:",
        f"{metrics['retrieval_accuracy_percent']:.2f}%",
    )

    print(
        "Average retrieval time:",
        f"{metrics['average_retrieval_time_seconds']:.4f}s",
    )

    print(
        "Top-k:",
        metrics["top_k"],
    )

    print("=" * 70)


def main():
    """Run and save the PECA retrieval evaluation."""

    print("=" * 70)
    print("PECA RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    retriever = load_retriever()

    print()
    print("Running evaluation...")
    print("-" * 70)

    results = evaluate_questions(retriever)

    metrics = calculate_metrics(results)

    print_summary(metrics)

    save_results(
        results,
        metrics,
    )


if __name__ == "__main__":
    main()