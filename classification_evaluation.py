import json
from pathlib import Path

from langchain_chroma import Chroma
from embeddings import get_embedding_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


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


def main() -> None:
    print("=" * 60)
    print("PECA RAG CLASSIFICATION EVALUATION")
    print("=" * 60)

    print("\n[1] Loading embedding model...")
    embeddings = get_embedding_model()

    print("[2] Loading Chroma vector store...")
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_PATH),
        embedding_function=embeddings,
    )

    y_true = []
    y_pred = []

    results = []

    print("\n[3] Running evaluation...")
    print("-" * 60)

    for index, item in enumerate(EVALUATION_QUESTIONS, start=1):

        question = item["question"]
        expected_section = item["expected_section"]

        documents = vectorstore.similarity_search(
            question,
            k=3,
        )

        predicted_section = None

        if documents:
            predicted_section = documents[0].metadata.get(
                "section_number"
            )

        y_true.append(expected_section)
        y_pred.append(predicted_section)

        is_correct = predicted_section == expected_section

        results.append(
            {
                "question_number": index,
                "question": question,
                "expected_section": expected_section,
                "predicted_section": predicted_section,
                "correct": is_correct,
            }
        )

        status = "PASS" if is_correct else "FAIL"

        print(
            f"{index:02d}. {status} | "
            f"Expected: {expected_section} | "
            f"Predicted: {predicted_section}"
        )

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    output_path = Path("evaluation_results/classification_results.json")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "total_questions": len(EVALUATION_QUESTIONS),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=4,
        )

    print("=" * 60)
    print("Results saved to:")
    print(output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()